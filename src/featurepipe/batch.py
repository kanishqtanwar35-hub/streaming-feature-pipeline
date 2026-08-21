"""The training path: the same features, recomputed from stored history.

This is the reference implementation. It has every advantage the streaming path
does not — the whole session in memory, sorted by event time, no watermark, no
decision about when a window is finished — and that is exactly why it is the
wrong thing to trust on its own.

**The trap this file exists to expose.** A batch job over complete history
computes the *true* value of every window. The streaming job computes what was
knowable at the time. Those differ whenever a tick arrived late, and the batch
number is the one that ends up in the training set.

So the model learns from features that were never available at decision time.
Offline metrics look good, live performance does not match, and nothing errors.
That is train/serve skew, and it is what
[`nse-realtime-screener`](https://github.com/kanishqtanwar35-hub/nse-realtime-screener)
documents about itself.

`replay=True` reproduces the streaming path's information limits inside the
batch job — it is the fix, and `parity.py` measures the difference between the
two modes so the cost of getting it wrong is a number rather than an argument.

Pure Python, no pandas. The reference implementation should have as few moving
parts as possible: if it needed a dataframe library, a version bump in that
library could change what "correct" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from featurepipe.spec import (
    FEATURES,
    Seconds,
    aggregate,
    apply_derived,
    empty_row,
)
from featurepipe.ticks import Tick, deduplicate, in_event_order


@dataclass
class BatchFeatures:
    """Recompute features from a stored tick history."""

    ticks: List[Tick]
    #: When True, a tick is only visible at times at or after it ARRIVED,
    #: reproducing what the streaming path could actually know.
    #:
    #: False is the naive batch job - and the naive batch job is the bug.
    replay: bool = False

    def __post_init__(self) -> None:
        # Deduplicate here too. At-least-once delivery means the stored history
        # contains redeliveries, and a batch job that does not remove them
        # double-counts exactly the ticks the streaming consumer skipped -
        # producing a disagreement that looks like a windowing bug.
        self.ticks = in_event_order(deduplicate(self.ticks))
        self._by_symbol: Dict[str, List[Tick]] = {}
        for tick in self.ticks:
            self._by_symbol.setdefault(tick.symbol, []).append(tick)

    def symbols(self) -> List[str]:
        return sorted(self._by_symbol)

    def _visible(self, symbol: str, at: Seconds) -> List[Tick]:
        ticks = self._by_symbol.get(symbol, [])
        if not self.replay:
            return ticks
        # The information limit: a tick that had not arrived yet was not
        # available to compute with, however true it is now.
        return [t for t in ticks if t.arrival_time <= at]

    def features_at(self, symbol: str, at: Seconds
                    ) -> Dict[str, Optional[float]]:
        ticks = self._visible(symbol, at)
        if not ticks:
            return empty_row()

        row: Dict[str, Optional[float]] = {}
        for spec in FEATURES:
            if spec.window is None:
                values = [getattr(t, spec.source)
                          for t in ticks if t.event_time <= at]
            else:
                start = at - spec.window.seconds
                # (at - window, at] - the spec's convention, spelled out here
                # rather than delegated, so the two implementations can be read
                # against each other line by line.
                values = [getattr(t, spec.source) for t in ticks
                          if start < t.event_time <= at]

            computed = aggregate(values, spec.aggregation)
            row[spec.name] = spec.empty_value if computed is None else computed

        return apply_derived(row)

    def rows(self, symbol: str, times: Sequence[Seconds]
             ) -> List[Dict[str, Optional[float]]]:
        return [{"symbol": symbol, "event_time": at,
                 **self.features_at(symbol, at)} for at in times]

    def table(self, times: Sequence[Seconds]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for symbol in self.symbols():
            out.extend(self.rows(symbol, times))
        return out


def to_parquet_like(rows: Sequence[Dict[str, object]], path) -> None:
    """Persist a feature table as newline-delimited JSON.

    NDJSON rather than Parquet on purpose: Parquet means pyarrow, and the
    feature store is not where a heavy dependency earns its place. The format
    is swappable; what matters is that training and serving read the SAME
    stored rows rather than each recomputing from ticks - which is the
    structural fix for skew that no amount of careful coding replaces.
    """
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_parquet_like(path) -> List[Dict[str, object]]:
    import json
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
