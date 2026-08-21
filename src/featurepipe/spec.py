"""The feature specification. One definition, three implementations.

**The real problem, taken from a real repository.**
`nse-realtime-screener` documents this about itself:

> *"The model trains on trades harvested from historical bars, which carry no
> tick data. So `ltq_*`, `bid_ask_spread_pct`, `order_book_imbalance` and
> `depth_ratio` are **zero in training but populated live**."*

That is train/serve skew, and it is the most expensive bug in applied ML because
nothing errors. Offline metrics look fine, the model ships, and it behaves
differently in production because it is being fed features that were computed a
different way — or, in that case, not computed at all.

The fix is not "be careful". The fix is to make the feature definition a single
artifact that every path must satisfy, and then **test that the paths agree**.

**Why this file has no aggregation code in it.** A specification that is also an
implementation is not a specification — it is just the first implementation, and
the others get compared against whichever happened to be written first. This
file declares *what* each feature is: its window, its aggregation, its boundary
convention, and what it does when the window is empty. `streaming.py`,
`batch.py` and `spark_batch.py` each compute it their own way, and
`parity.py` asserts all three agree.

**Boundary convention, stated once because it is where implementations diverge.**
A window of length W ending at time T covers:

    (T - W, T]        left-open, right-closed

A tick exactly at `T - W` is OUT. A tick exactly at `T` is IN. Pandas'
`rolling`, Spark's `window()` and a hand-written deque each default to
*different* conventions, and a one-tick disagreement at a boundary is enough to
make two implementations of "the same" feature produce different numbers on the
same data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: All times are epoch SECONDS as floats, in event time. Never processing time.
#: The distinction is the single most important one in this repository - see
#: `Aggregation.EVENT_TIME_NOTE`.
Seconds = float


@dataclass(frozen=True)
class WindowSpec:
    """A time window, with its boundary convention made explicit."""

    seconds: Seconds

    def contains(self, event_time: Seconds, window_end: Seconds) -> bool:
        """Left-open, right-closed: (end - seconds, end].

        Written as a function rather than left to each implementation, because
        this is exactly the line where implementations silently diverge. A tick
        at the boundary belongs to one window in pandas and the neighbouring
        one in Spark unless somebody decides.
        """
        return (window_end - self.seconds) < event_time <= window_end


@dataclass(frozen=True)
class FeatureSpec:
    """One feature: what it aggregates, over what window, and its empty value."""

    name: str
    #: The tick field it reads.
    source: str
    #: "mean" | "sum" | "count" | "last" | "max" | "min"
    aggregation: str
    window: Optional[WindowSpec]
    #: What the feature is when the window contains no ticks.
    #:
    #: NOT zero by default, and that matters more than it looks. Zero is a
    #: *value*: it says "the average trade size was nothing", which is a claim
    #: about the market. "No ticks in the window" is an absence of evidence. A
    #: model trained on rows where absence was encoded as 0.0 learns that 0.0
    #: means quiet - and then in production a genuine zero and a missing feed
    #: are indistinguishable to it.
    empty_value: Optional[float] = None
    description: str = ""

    EVENT_TIME_NOTE = (
        "Every window here is over EVENT time - when the exchange stamped the "
        "tick - never processing time. Keyed on arrival, a replay of the same "
        "data produces different features, which makes the pipeline "
        "irreproducible and every backtest meaningless."
    )


@dataclass(frozen=True)
class DerivedSpec:
    """A feature computed from other features rather than from ticks."""

    name: str
    inputs: Tuple[str, ...]
    fn: Callable[[Dict[str, Optional[float]]], Optional[float]]
    description: str = ""


def _safe_ratio(values: Dict[str, Optional[float]], numerator: str,
                denominator: str) -> Optional[float]:
    """A ratio that propagates absence instead of inventing 1.0.

    The predecessor repo defaults this ratio to 1.0 when the denominator is
    missing, which reads as "no change" - a confident, wrong statement. If
    either side is unknown the ratio is unknown, and the model should be told
    that rather than handed a plausible number.
    """
    top, bottom = values.get(numerator), values.get(denominator)
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom


#: The window lengths from the original brief. 2m vs 5m is the comparison it
#: names explicitly, which is why it is the headline feature here.
TWO_MIN = WindowSpec(120.0)
FIVE_MIN = WindowSpec(300.0)
TWENTY_MIN = WindowSpec(1200.0)

FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("ltq_avg_2m", "ltq", "mean", TWO_MIN,
                description="2-minute average trade size"),
    FeatureSpec("ltq_avg_5m", "ltq", "mean", FIVE_MIN,
                description="5-minute average trade size"),
    FeatureSpec("ltq_avg_20m", "ltq", "mean", TWENTY_MIN,
                description="20-minute average trade size"),
    FeatureSpec("ltq_sum_2m", "ltq", "sum", TWO_MIN,
                empty_value=0.0,
                description="2-minute traded quantity. Zero IS meaningful here "
                            "- no trades means zero volume - so unlike the "
                            "averages this one legitimately defaults to 0."),
    FeatureSpec("tick_count_2m", "ltq", "count", TWO_MIN,
                empty_value=0.0,
                description="ticks in the last 2 minutes"),
    FeatureSpec("ltp_last", "ltp", "last", None,
                description="most recent traded price"),
    FeatureSpec("ltp_avg_5m", "ltp", "mean", FIVE_MIN,
                description="5-minute average traded price"),
    FeatureSpec("ltp_max_5m", "ltp", "max", FIVE_MIN,
                description="5-minute high"),
    FeatureSpec("ltp_min_5m", "ltp", "min", FIVE_MIN,
                description="5-minute low"),
)

DERIVED: Tuple[DerivedSpec, ...] = (
    DerivedSpec("ltq_ratio_2_5", ("ltq_avg_2m", "ltq_avg_5m"),
                lambda v: _safe_ratio(v, "ltq_avg_2m", "ltq_avg_5m"),
                "trade-size ratio, 2m vs 5m - the comparison the brief names "
                "explicitly. Above 1.0 means trade size is stepping in now."),
    DerivedSpec("ltq_spike_2_20", ("ltq_avg_2m", "ltq_avg_20m"),
                lambda v: _safe_ratio(v, "ltq_avg_2m", "ltq_avg_20m"),
                "2m trade size against the 20m baseline"),
    DerivedSpec("ltp_range_5m", ("ltp_max_5m", "ltp_min_5m"),
                lambda v: (None if v.get("ltp_max_5m") is None
                           or v.get("ltp_min_5m") is None
                           else v["ltp_max_5m"] - v["ltp_min_5m"]),
                "5-minute high-low range"),
)

FEATURES_BY_NAME: Dict[str, FeatureSpec] = {f.name: f for f in FEATURES}
DERIVED_BY_NAME: Dict[str, DerivedSpec] = {d.name: d for d in DERIVED}

ALL_NAMES: Tuple[str, ...] = tuple(
    [f.name for f in FEATURES] + [d.name for d in DERIVED]
)

#: The longest window any feature needs. A streaming implementation cannot
#: discard a tick older than this, and a batch implementation cannot produce a
#: trustworthy value until this much history exists - the "cold start" that
#: makes the first rows of a backtest quietly different from the rest.
MAX_WINDOW_S: Seconds = max(f.window.seconds for f in FEATURES if f.window)


def aggregate(values: Sequence[float], how: str) -> Optional[float]:
    """The one place an aggregation is defined.

    Returns None for an empty input rather than 0.0 or NaN. The caller applies
    the feature's `empty_value`, so the decision about what absence means lives
    in the SPEC and not in three separate implementations that will drift.
    """
    if not values:
        return None
    if how == "mean":
        return sum(values) / len(values)
    if how == "sum":
        return float(sum(values))
    if how == "count":
        return float(len(values))
    if how == "last":
        return float(values[-1])
    if how == "max":
        return float(max(values))
    if how == "min":
        return float(min(values))
    raise ValueError(f"unknown aggregation '{how}'")


def apply_derived(base: Dict[str, Optional[float]]
                  ) -> Dict[str, Optional[float]]:
    """Add the derived features to a row of base features."""
    out = dict(base)
    for spec in DERIVED:
        out[spec.name] = spec.fn(out)
    return out


def empty_row() -> Dict[str, Optional[float]]:
    """A feature row with no ticks behind it at all."""
    base = {f.name: f.empty_value for f in FEATURES}
    return apply_derived(base)


def describe() -> str:
    lines = [FeatureSpec.EVENT_TIME_NOTE, "",
             f"{'feature':<16} {'window':>8} {'agg':>6} {'empty':>7}  description"]
    lines.append("-" * 100)
    for spec in FEATURES:
        window = f"{spec.window.seconds:.0f}s" if spec.window else "-"
        empty = "None" if spec.empty_value is None else f"{spec.empty_value:g}"
        lines.append(f"{spec.name:<16} {window:>8} {spec.aggregation:>6} "
                     f"{empty:>7}  {spec.description}")
    for spec in DERIVED:
        lines.append(f"{spec.name:<16} {'derived':>8} {'-':>6} {'None':>7}  "
                     f"{spec.description}")
    return "\n".join(lines)
