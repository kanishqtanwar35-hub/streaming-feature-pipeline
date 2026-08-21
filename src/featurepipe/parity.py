"""Do the paths agree? Measured per feature, per watermark.

This is the deliverable. Everything else in the repository exists so that this
comparison can be made honestly.

**What is being compared**

  `streaming`      what the serving path could compute, live, under a watermark
  `batch_naive`    what a batch job over complete history computes - the
                   number that ends up in the training set
  `batch_replay`   the batch job constrained to what was knowable at the time

`streaming` vs `batch_naive` is the **skew**: two implementations of the same
feature, same data, different answers, no error anywhere.

`streaming` vs `batch_replay` is the **correctness check**: with the same
information, the two implementations must agree exactly. If they do not, one of
them has a bug — a boundary convention, a deduplication miss, an eviction that
fired early — and the parity harness is what finds it.

Both comparisons matter and they say different things. Reporting only the second
would make the pipeline look perfect while training on numbers that were never
available. Reporting only the first would blame the watermark for what might be
an off-by-one in a window.

**Tolerance is exact for the replay comparison.** Two implementations given the
same ticks and the same window definition should produce bit-identical floats,
because both are summing the same values in the same order. A tolerance there
would hide precisely the boundary bugs the comparison exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from featurepipe.batch import BatchFeatures
from featurepipe.spec import ALL_NAMES, Seconds
from featurepipe.streaming import DEFAULT_WATERMARK_S, StreamingFeatures
from featurepipe.ticks import Tick, evaluation_times, statistics

#: Exact. See the module docstring.
EXACT = 0.0


@dataclass
class FeatureDelta:
    """Disagreement on one feature, across all evaluated instants."""

    name: str
    compared: int = 0
    disagreements: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    #: One concrete example, because a rate without an instance is not
    #: debuggable.
    worst_example: Optional[Tuple[Seconds, Optional[float], Optional[float]]] = None
    #: Cases where one side has a value and the other has None. Counted apart
    #: from numeric disagreement: "we computed a different number" and "one of
    #: us had no data at all" are different failures with different fixes.
    presence_mismatch: int = 0

    @property
    def rate(self) -> float:
        return self.disagreements / self.compared if self.compared else 0.0


@dataclass
class ParityReport:
    label: str
    tolerance: float
    features: Dict[str, FeatureDelta] = field(default_factory=dict)
    rows_compared: int = 0

    @property
    def agree(self) -> bool:
        return all(d.disagreements == 0 and d.presence_mismatch == 0
                   for d in self.features.values())

    @property
    def worst(self) -> List[FeatureDelta]:
        return sorted(self.features.values(), key=lambda d: -d.rate)

    def summary(self) -> str:
        broken = [d for d in self.worst if d.disagreements or d.presence_mismatch]
        if not broken:
            return (f"{self.label}: AGREE on all {len(self.features)} features "
                    f"across {self.rows_compared} rows")
        lines = [f"{self.label}: {len(broken)} of {len(self.features)} features "
                 f"disagree across {self.rows_compared} rows",
                 f"  {'feature':<16} {'rate':>7} {'max abs':>10} {'max rel':>9}"]
        for delta in broken:
            lines.append(f"  {delta.name:<16} {delta.rate:>6.1%} "
                         f"{delta.max_abs:>10.4f} {delta.max_rel:>8.1%}")
        return "\n".join(lines)


def _compare_value(delta: FeatureDelta, at: Seconds,
                   left: Optional[float], right: Optional[float],
                   tolerance: float) -> None:
    delta.compared += 1

    if (left is None) != (right is None):
        delta.presence_mismatch += 1
        delta.disagreements += 1
        if delta.worst_example is None:
            delta.worst_example = (at, left, right)
        return

    if left is None and right is None:
        return

    absolute = abs(left - right)
    if absolute <= tolerance:
        return

    relative = absolute / abs(right) if right else float("inf")
    delta.disagreements += 1
    if absolute > delta.max_abs:
        delta.max_abs = absolute
        delta.max_rel = relative
        delta.worst_example = (at, left, right)


def compare(left_rows: Sequence[Dict[str, object]],
            right_rows: Sequence[Dict[str, object]],
            label: str, tolerance: float = EXACT) -> ParityReport:
    """Compare two feature tables, row for row."""
    report = ParityReport(label=label, tolerance=tolerance)
    for name in ALL_NAMES:
        report.features[name] = FeatureDelta(name)

    index = {(r["symbol"], r["event_time"]): r for r in right_rows}
    for row in left_rows:
        other = index.get((row["symbol"], row["event_time"]))
        if other is None:
            continue
        report.rows_compared += 1
        for name in ALL_NAMES:
            _compare_value(report.features[name], float(row["event_time"]),
                           row.get(name), other.get(name), tolerance)
    return report


# ---------------------------------------------------------------------------

@dataclass
class Run:
    ticks: List[Tick]
    times: List[Seconds]
    streaming_rows: List[Dict[str, object]]
    naive_rows: List[Dict[str, object]]
    replay_rows: List[Dict[str, object]]
    stream_stats: object
    tick_stats: Dict[str, float]


def run(ticks: Sequence[Tick], watermark_s: Seconds = DEFAULT_WATERMARK_S,
        every_s: Seconds = 30.0) -> Run:
    """Compute all three tables over the same ticks."""
    ticks = list(ticks)
    times = evaluation_times(ticks, every_s=every_s)

    # consume(), not offer_all() + rows(). The latter queries the finished
    # buffer, which by then holds ticks that had not arrived at the time being
    # asked about - the streaming path would be answering with information it
    # did not have, and parity would fail against a correct batch job.
    stream = StreamingFeatures(watermark_s=watermark_s)
    streaming_rows = stream.consume(ticks, times)

    naive = BatchFeatures(ticks, replay=False)
    replay = BatchFeatures(ticks, replay=True)

    symbols = sorted({t.symbol for t in ticks})
    naive_rows, replay_rows = [], []
    for symbol in symbols:
        naive_rows.extend(naive.rows(symbol, times))
        replay_rows.extend(replay.rows(symbol, times))

    return Run(ticks=ticks, times=times, streaming_rows=streaming_rows,
               naive_rows=naive_rows, replay_rows=replay_rows,
               stream_stats=stream.stats, tick_stats=statistics(ticks))


def skew(result: Run, tolerance: float = 1e-9) -> ParityReport:
    """streaming vs the naive batch job. The train/serve skew."""
    return compare(result.streaming_rows, result.naive_rows,
                   "streaming vs batch (naive)", tolerance)


def correctness(result: Run) -> ParityReport:
    """streaming vs the replayed batch job. Must be EXACT."""
    return compare(result.streaming_rows, result.replay_rows,
                   "streaming vs batch (replay)", EXACT)


def watermark_sweep(ticks: Sequence[Tick],
                    watermarks: Sequence[Seconds] = (0.0, 5.0, 15.0, 30.0,
                                                     45.0, 60.0, 120.0),
                    every_s: Seconds = 30.0) -> List[Dict[str, float]]:
    """What each watermark setting costs, measured rather than argued.

    Two columns move in opposite directions and there is no setting that
    optimises both:

      `dropped_pct`   ticks discarded for arriving too late. Every one is a
                      difference between what training saw and what serving
                      saw.
      `latency_s`     how long a window must wait before it can be emitted.

    A signal that decays in seconds cannot afford a 120-second watermark, and a
    pipeline whose features must match training cannot afford to drop 4% of its
    ticks. Picking a number without this table is guessing.
    """
    rows = []
    for watermark in watermarks:
        result = run(ticks, watermark_s=watermark, every_s=every_s)
        divergence = skew(result)
        broken = sum(1 for d in divergence.features.values() if d.disagreements)
        worst = max((d.rate for d in divergence.features.values()), default=0.0)
        rows.append({
            "watermark_s": watermark,
            "dropped": float(result.stream_stats.dropped_late),
            "dropped_pct": result.stream_stats.drop_rate * 100.0,
            "features_disagreeing": float(broken),
            "worst_feature_rate_pct": worst * 100.0,
            "latency_s": watermark,
        })
    return rows


def format_sweep(rows: Sequence[Dict[str, float]]) -> str:
    lines = [f"{'watermark':>10} {'latency':>8} {'dropped':>8} {'dropped%':>9} "
             f"{'features':>9} {'worst%':>8}",
             "-" * 60]
    for row in rows:
        lines.append(
            f"{row['watermark_s']:>9.0f}s {row['latency_s']:>7.0f}s "
            f"{row['dropped']:>8.0f} {row['dropped_pct']:>8.2f}% "
            f"{row['features_disagreeing']:>9.0f} "
            f"{row['worst_feature_rate_pct']:>7.1f}%")
    return "\n".join(lines)
