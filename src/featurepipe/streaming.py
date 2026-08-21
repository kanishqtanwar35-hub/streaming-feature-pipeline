"""The serving path: features computed incrementally, from a live stream.

This is the implementation that has to be *right* under conditions the batch job
never faces — it sees ticks in arrival order, it cannot see the future, and it
has to decide when a window is finished without ever being certain.

**The watermark is the whole design.** A window ending at T cannot be emitted
the instant the clock passes T, because a tick stamped before T may still be in
flight. So the consumer tracks the greatest event time it has seen and declares
everything older than `max_event_time - watermark_s` complete.

That is a *bet*, and the trade is explicit:

  **Short watermark** → features are available sooner, and late ticks are
  dropped. The streaming path then computes a smaller average than the batch
  path over identical data. That is train/serve skew, arriving quietly.

  **Long watermark** → features are correct but late. For a signal that decays
  in seconds, a correct answer that arrives a minute afterwards is not a
  correct answer.

There is no setting that avoids the trade. `parity.py` measures what each
setting costs, which is the only honest way to pick one.

**Dropped ticks are counted, not swallowed.** A pipeline that silently discards
late data reports healthy while its features quietly diverge from training. The
count is the metric that catches it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from featurepipe.spec import (
    DERIVED,
    FEATURES,
    MAX_WINDOW_S,
    Seconds,
    aggregate,
    apply_derived,
)
from featurepipe.ticks import Tick

#: How long to wait for late data before declaring a window complete.
#:
#: 45s is chosen against the generator's own p99 lateness rather than picked:
#: `ticks.statistics()` reports it, and a watermark below the p99 drops a
#: measurable fraction of ticks. Picking it without measuring is the same
#: mistake as a folklore PSI threshold.
DEFAULT_WATERMARK_S: Seconds = 45.0


@dataclass
class StreamStats:
    """What the consumer did. Every number here is a real operational metric."""

    seen: int = 0
    accepted: int = 0
    duplicates: int = 0
    #: Ticks that arrived after their window was already declared complete.
    #: THE number to alert on: it is the direct measure of train/serve skew
    #: being introduced.
    dropped_late: int = 0
    out_of_order: int = 0
    max_event_time: Seconds = 0.0
    max_lateness_seen: Seconds = 0.0

    @property
    def drop_rate(self) -> float:
        return self.dropped_late / self.seen if self.seen else 0.0

    def summary(self) -> str:
        return (f"{self.seen} ticks, {self.accepted} accepted, "
                f"{self.duplicates} duplicate, {self.dropped_late} dropped late "
                f"({self.drop_rate:.2%}), {self.out_of_order} out of order, "
                f"max lateness {self.max_lateness_seen:.1f}s")


@dataclass
class SymbolState:
    """The per-symbol buffer. One deque, bounded by the longest window."""

    symbol: str
    ticks: Deque[Tick] = field(default_factory=deque)

    def add(self, tick: Tick) -> None:
        """Insert keeping EVENT-time order.

        Appending in arrival order and sorting later would be simpler and
        wrong: every window query would have to re-sort, and the cost is
        O(n log n) per query on the hot path. Ticks are nearly ordered, so a
        backward linear scan touches one or two elements in practice.
        """
        if not self.ticks or tick.event_time >= self.ticks[-1].event_time:
            self.ticks.append(tick)
            return

        buffer: List[Tick] = []
        while self.ticks and self.ticks[-1].event_time > tick.event_time:
            buffer.append(self.ticks.pop())
        self.ticks.append(tick)
        while buffer:
            self.ticks.append(buffer.pop())

    def evict(self, before: Seconds) -> None:
        """Drop ticks no window can still need."""
        while self.ticks and self.ticks[0].event_time <= before:
            self.ticks.popleft()

    def window(self, end: Seconds, seconds: Seconds) -> List[Tick]:
        """Ticks in (end - seconds, end]. The spec's boundary convention."""
        start = end - seconds
        return [t for t in self.ticks if start < t.event_time <= end]


class StreamingFeatures:
    """Incremental feature computation over a live tick stream."""

    def __init__(self, watermark_s: Seconds = DEFAULT_WATERMARK_S,
                 deduplicate: bool = True,
                 retain_s: Optional[Seconds] = None):
        self.watermark_s = watermark_s
        self.deduplicate = deduplicate
        #: How much history to keep. Must exceed the longest window plus the
        #: watermark, or a late tick arrives to find its window already evicted
        #: - a bug that looks exactly like a watermark that is too short.
        self.retain_s = retain_s if retain_s is not None \
            else MAX_WINDOW_S + watermark_s + 60.0

        self.state: Dict[str, SymbolState] = {}
        self.stats = StreamStats()
        self._seen_keys: Set[Tuple[str, int]] = set()

    # -- ingestion ----------------------------------------------------------

    @property
    def watermark(self) -> Seconds:
        """Everything at or below this event time is declared complete."""
        return self.stats.max_event_time - self.watermark_s

    def offer(self, tick: Tick) -> bool:
        """Ingest one tick. False if it was rejected, and the reason is counted."""
        self.stats.seen += 1
        self.stats.max_lateness_seen = max(self.stats.max_lateness_seen,
                                           tick.lateness)

        if self.deduplicate:
            if tick.key in self._seen_keys:
                self.stats.duplicates += 1
                return False
            self._seen_keys.add(tick.key)

        # Late beyond the watermark: its window has already been reported, so
        # accepting it would silently change a value somebody has already acted
        # on. Rejected and COUNTED - this is the skew metric.
        if self.stats.max_event_time and tick.event_time <= self.watermark:
            self.stats.dropped_late += 1
            return False

        state = self.state.get(tick.symbol)
        if state is None:
            state = self.state[tick.symbol] = SymbolState(tick.symbol)

        if state.ticks and tick.event_time < state.ticks[-1].event_time:
            self.stats.out_of_order += 1

        state.add(tick)
        self.stats.accepted += 1
        self.stats.max_event_time = max(self.stats.max_event_time,
                                        tick.event_time)
        state.evict(self.stats.max_event_time - self.retain_s)
        return True

    def offer_all(self, ticks: Iterable[Tick]) -> StreamStats:
        for tick in ticks:
            self.offer(tick)
        return self.stats

    # -- querying -----------------------------------------------------------

    def features_at(self, symbol: str, at: Seconds
                    ) -> Dict[str, Optional[float]]:
        """The feature row for one symbol at one event time."""
        state = self.state.get(symbol)
        if state is None:
            from featurepipe.spec import empty_row
            return empty_row()

        row: Dict[str, Optional[float]] = {}
        for spec in FEATURES:
            if spec.window is None:
                # An unwindowed feature is "the most recent value at or before
                # `at`" - NOT the most recent value the consumer holds. Using
                # the latter would leak the future into a historical query and
                # make replay disagree with live.
                candidates = [t for t in state.ticks if t.event_time <= at]
                values = [getattr(t, spec.source) for t in candidates]
            else:
                values = [getattr(t, spec.source)
                          for t in state.window(at, spec.window.seconds)]

            computed = aggregate(values, spec.aggregation)
            row[spec.name] = spec.empty_value if computed is None else computed

        return apply_derived(row)

    def rows(self, symbol: str, times: Sequence[Seconds]
             ) -> List[Dict[str, Optional[float]]]:
        """Query the CURRENT buffer at several event times.

        Only correct once the stream has finished, or for `at` values at the
        head of the stream. For a historical row use `consume()` - see the
        warning there.
        """
        return [{"symbol": symbol, "event_time": at,
                 **self.features_at(symbol, at)} for at in times]

    def consume(self, ticks: Iterable[Tick], emit_times: Sequence[Seconds]
                ) -> List[Dict[str, Optional[float]]]:
        """Ingest in arrival order, emitting a feature row as the clock passes
        each time in `emit_times`.

        **This is the only correct way to produce historical rows, and getting
        it wrong is a bug worth understanding.**

        The obvious approach - offer the whole stream, then call `features_at`
        for each past time - is wrong, and it is wrong in the direction that
        makes everything look fine. By the end of the stream the buffer holds
        ticks that had not arrived at `at`, so the query answers with
        information the consumer did not have then. Parity against a correctly
        replayed batch job then FAILS, and the natural reading is "the batch job
        is broken" when in fact the streaming query saw the future.

        It is the same mistake as grounding a camera detection against the
        robot's current pose instead of the pose when the frame was taken: a
        retrospective query into live state silently borrows later knowledge.

        So the row for time T is emitted after every tick with
        `arrival_time <= T` has been ingested and before any tick that arrived
        later - which is exactly what a real consumer does, because it has no
        choice.
        """
        rows: List[Dict[str, Optional[float]]] = []
        pending = sorted(emit_times)
        index = 0

        for tick in sorted(ticks, key=lambda t: (t.arrival_time, t.seq)):
            while index < len(pending) and pending[index] < tick.arrival_time:
                rows.extend(self._emit(pending[index]))
                index += 1
            self.offer(tick)

        while index < len(pending):
            rows.extend(self._emit(pending[index]))
            index += 1
        return rows

    def _emit(self, at: Seconds) -> List[Dict[str, Optional[float]]]:
        return [{"symbol": symbol, "event_time": at,
                 **self.features_at(symbol, at)}
                for symbol in sorted(self.state)]

    def symbols(self) -> List[str]:
        return sorted(self.state)
