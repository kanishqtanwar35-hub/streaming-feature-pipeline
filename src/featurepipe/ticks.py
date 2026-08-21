"""Ticks, and the two timestamps that cause every streaming bug.

Every tick carries **two** times, and conflating them is the root of most
train/serve skew:

  `event_time`   — when the exchange stamped the trade. The truth.
  `arrival_time` — when our process saw it. An accident of the network, the
                   broker's batching, a GC pause, a reconnect.

A batch job sorts by `event_time` and sees a clean, ordered history. A streaming
job sees `arrival_time` order, which is neither sorted nor complete. Compute a
"2-minute average" over the second and you get a different number from the
first — on the same data — and nothing anywhere reports an error.

The generator below produces the three arrival pathologies that actually happen,
because a parity test against perfectly ordered synthetic data proves nothing:

  **Out of order.** Two ticks stamped 3 ms apart arrive in the wrong order.
  Routine, and mostly harmless.

  **Late.** A tick arrives seconds after its event time, usually in a burst
  after a reconnect. This is the one that breaks windows: by the time it lands,
  the window it belongs to may already have been emitted.

  **Duplicate.** The same tick delivered twice, because at-least-once delivery
  is what a broker actually guarantees. A consumer that does not deduplicate
  double-counts it into every aggregate.

Deterministic given a seed, and seeded with `hashlib` rather than the builtin
`hash()` - Python randomises string hashing per process, so a `hash()`-based
generator is not reproducible across runs and the resulting test flakes about
one run in three.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from featurepipe.spec import Seconds


@dataclass(frozen=True)
class Tick:
    """One trade print."""

    symbol: str
    #: Exchange timestamp. Everything is computed against this.
    event_time: Seconds
    #: When this process saw it. Used ONLY for lateness accounting, never for
    #: windowing - see the module docstring.
    arrival_time: Seconds
    ltp: float
    ltq: float
    #: Broker sequence number. The deduplication key, because two genuinely
    #: distinct trades can share a symbol, a timestamp, a price and a size.
    seq: int = 0

    @property
    def lateness(self) -> Seconds:
        return self.arrival_time - self.event_time

    @property
    def key(self) -> Tuple[str, int]:
        return (self.symbol, self.seq)

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol, "event_time": self.event_time,
            "arrival_time": self.arrival_time, "ltp": self.ltp,
            "ltq": self.ltq, "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Tick":
        return cls(
            symbol=str(payload["symbol"]),
            event_time=float(payload["event_time"]),
            arrival_time=float(payload.get("arrival_time",
                                           payload["event_time"])),
            ltp=float(payload["ltp"]), ltq=float(payload["ltq"]),
            seq=int(payload.get("seq", 0)),
        )


@dataclass(frozen=True)
class StreamProfile:
    """How badly behaved the feed is.

    Defaults are deliberately mild - a well-behaved session. The demo turns
    them up, because a parity test that only ever sees clean data proves the
    implementations agree on the easy case.
    """

    #: Probability a tick arrives out of order relative to its neighbour.
    out_of_order_rate: float = 0.10
    #: Probability a tick is LATE by a meaningful amount.
    late_rate: float = 0.04
    #: How late, in seconds, when it is late. 30s is a short reconnect.
    max_lateness_s: Seconds = 30.0
    #: Probability of an at-least-once redelivery.
    duplicate_rate: float = 0.02
    #: Mean seconds between ticks for one symbol.
    mean_gap_s: Seconds = 1.5


SYMBOLS: Tuple[str, ...] = ("IDEA", "YESBANK", "PNB", "SUZLON")

#: Rough starting prices, in the brief's Rs 30-500 band except IDEA which is a
#: penny stock - kept because a cheap, heavily traded symbol exercises the
#: large-ltq / small-ltp corner that the ratio features care about.
START_PRICE: Dict[str, float] = {
    "IDEA": 7.5, "YESBANK": 21.0, "PNB": 104.0, "SUZLON": 58.0,
}


def generate(symbol: str = "IDEA", minutes: float = 40.0,
             start_time: Seconds = 1_700_000_000.0,
             profile: StreamProfile = StreamProfile(),
             seed: int = 0) -> List[Tick]:
    """A session of ticks for one symbol, in ARRIVAL order.

    Returned in arrival order on purpose: that is what a consumer sees, and
    handing back event-ordered data would quietly do the streaming
    implementation's hardest job for it.
    """
    rng = random.Random(f"{symbol}:{seed}")
    ticks: List[Tick] = []

    price = START_PRICE.get(symbol, 100.0)
    now = start_time
    seq = 0
    horizon = start_time + minutes * 60.0

    while now < horizon:
        now += rng.expovariate(1.0 / profile.mean_gap_s)
        if now >= horizon:
            break
        seq += 1

        # A random walk with occasional size bursts - the pattern the 2m-vs-5m
        # ratio exists to detect.
        price = max(0.05, price * (1.0 + rng.gauss(0.0, 0.0006)))
        burst = rng.random() < 0.06
        quantity = float(max(1, int(rng.lognormvariate(6.2 if burst else 5.0,
                                                       0.8))))

        arrival = now
        if rng.random() < profile.late_rate:
            arrival = now + rng.uniform(1.0, profile.max_lateness_s)
        elif rng.random() < profile.out_of_order_rate:
            arrival = now + rng.uniform(0.001, 0.4)

        ticks.append(Tick(symbol, now, arrival, round(price, 2), quantity, seq))

        if rng.random() < profile.duplicate_rate:
            # At-least-once redelivery: identical seq, later arrival.
            ticks.append(Tick(symbol, now, arrival + rng.uniform(0.05, 2.0),
                              round(price, 2), quantity, seq))

    ticks.sort(key=lambda t: t.arrival_time)
    return ticks


def generate_multi(symbols: Sequence[str] = SYMBOLS,
                   seed: int = 0, **kwargs) -> List[Tick]:
    """Ticks for several symbols, interleaved in arrival order.

    Each symbol gets its own seed offset so they are independent streams rather
    than the same walk under different names - which would make a per-symbol
    grouping bug invisible, because every group would agree by construction.
    """
    out: List[Tick] = []
    for index, symbol in enumerate(symbols):
        out.extend(generate(symbol, seed=seed + index, **kwargs))
    out.sort(key=lambda t: t.arrival_time)
    return out


def in_event_order(ticks: Iterable[Tick]) -> List[Tick]:
    """Sorted by event time. What a batch job gets to start from."""
    return sorted(ticks, key=lambda t: (t.event_time, t.seq))


def deduplicate(ticks: Iterable[Tick]) -> List[Tick]:
    """First occurrence of each (symbol, seq), in the order given.

    At-least-once is what a broker guarantees, so exactly-once has to be built
    on the consumer side. Keeping the FIRST occurrence rather than the last is
    the right choice: a redelivery is the same trade, and preferring the later
    copy would move its arrival time for no reason.
    """
    seen = set()
    out = []
    for tick in ticks:
        if tick.key in seen:
            continue
        seen.add(tick.key)
        out.append(tick)
    return out


def statistics(ticks: Sequence[Tick]) -> Dict[str, float]:
    """What the feed actually did. Reported, not assumed."""
    if not ticks:
        return {}

    deduped = deduplicate(ticks)
    lateness = sorted(t.lateness for t in deduped)
    out_of_order = sum(
        1 for a, b in zip(deduped, deduped[1:]) if b.event_time < a.event_time
    )
    return {
        "ticks": float(len(ticks)),
        "unique": float(len(deduped)),
        "duplicates": float(len(ticks) - len(deduped)),
        "out_of_order": float(out_of_order),
        "out_of_order_pct": out_of_order / max(1, len(deduped) - 1) * 100.0,
        "max_lateness_s": lateness[-1],
        "p99_lateness_s": lateness[int(len(lateness) * 0.99) - 1],
        "median_lateness_s": lateness[len(lateness) // 2],
        "span_minutes": (deduped[-1].event_time - deduped[0].event_time) / 60.0,
    }


def evaluation_times(ticks: Sequence[Tick], every_s: Seconds = 30.0,
                     warmup_s: Optional[Seconds] = None) -> List[Seconds]:
    """The instants at which features are compared.

    `warmup_s` skips the cold start. Every window is partial before the stream
    has been running for its full length, so the first rows are legitimately
    different from the rest - and comparing implementations there measures the
    cold start rather than the implementations. Defaults to the longest window
    in the spec.
    """
    from featurepipe.spec import MAX_WINDOW_S

    if not ticks:
        return []
    warmup = MAX_WINDOW_S if warmup_s is None else warmup_s
    ordered = in_event_order(deduplicate(ticks))
    start = ordered[0].event_time + warmup
    end = ordered[-1].event_time

    times, now = [], start
    while now <= end:
        times.append(now)
        now += every_s
    return times
