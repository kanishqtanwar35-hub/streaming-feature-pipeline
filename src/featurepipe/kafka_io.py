"""Kafka transport. Optional — the pipeline is testable without a broker.

Nothing else in `featurepipe` imports this module. The streaming logic takes an
iterable of `Tick`, so the broker is a *transport detail*, and keeping it that
way is what lets the parity harness run in a few milliseconds with no
infrastructure at all.

That is not a testing convenience. A pipeline whose windowing logic can only be
exercised by standing up a broker is one where the windowing logic does not get
exercised.

**What Kafka is actually contributing here**, beyond being on the CV:

  **Partitioning by symbol.** Ordering in Kafka is per-partition, not per-topic.
  Key by symbol and every tick for one symbol lands on one partition in order;
  key by nothing and a symbol's ticks scatter across partitions and arrive
  interleaved, which turns a mild out-of-order problem into a severe one. This
  is the single most consequential line in the file.

  **Consumer offsets as the replay mechanism.** Reprocessing a session means
  seeking to an offset, not re-downloading a file.

  **At-least-once delivery, made concrete.** The broker will redeliver on a
  rebalance or a crashed commit. `StreamingFeatures` deduplicates on
  `(symbol, seq)` for exactly that reason, and the generator produces
  duplicates so the path is tested rather than assumed.

Run a local broker with `docker compose -f docker/compose.yaml up -d` — KRaft,
single node, no ZooKeeper, about 285 MB of RAM and no cost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from featurepipe.ticks import Tick

DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_TOPIC = "ticks"


def _require_kafka():
    """Import the client lazily, with an actionable message if it is absent."""
    try:
        from kafka import KafkaConsumer, KafkaProducer      # noqa: F401
        return KafkaProducer, KafkaConsumer
    except ImportError as error:                            # pragma: no cover
        raise RuntimeError(
            "kafka-python-ng is not installed. It is an OPTIONAL dependency: "
            "the feature logic, the parity harness and the whole test suite "
            "run without a broker. Install it with `pip install "
            "kafka-python-ng` and start one with "
            "`docker compose -f docker/compose.yaml up -d`."
        ) from error


@dataclass
class TickProducer:
    """Publishes ticks, keyed by symbol."""

    bootstrap_servers: str = DEFAULT_BOOTSTRAP
    topic: str = DEFAULT_TOPIC
    _producer: object = None
    _futures: list = field(default_factory=list)

    def __enter__(self) -> "TickProducer":
        KafkaProducer, _ = _require_kafka()
        self._producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # THE line that matters. Ordering in Kafka is per-partition; keying
            # by symbol puts each symbol's ticks on one partition, in order.
            # Without it, a symbol's ticks scatter across partitions and the
            # consumer sees them interleaved - a mild out-of-order problem
            # becomes a severe one, and the watermark has to grow to cover it.
            key_serializer=lambda k: k.encode("utf-8"),
            acks="all",
            linger_ms=5,
            # retries defaults to ZERO. On a freshly created topic the first
            # sends routinely hit NOT_LEADER_FOR_PARTITION while leadership is
            # still settling, and with no retries those records are dropped -
            # permanently, and without raising, because the error goes into a
            # future nobody reads.
            retries=10,
            # With retries enabled, more than one in-flight request can reorder
            # a partition: request 2 succeeds while request 1 is being retried.
            # Ordering within a symbol is the entire reason for keying by
            # symbol, so it is not something to trade away for throughput here.
            max_in_flight_requests_per_connection=1,
            request_timeout_ms=30_000,
        )
        self._futures = []
        return self

    def __exit__(self, *exc) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()

    def send(self, tick: Tick) -> None:
        """Queue one tick. The returned future is KEPT - see `send_all`."""
        self._futures.append(
            self._producer.send(self.topic, key=tick.symbol,
                                value=tick.to_dict()))

    def send_all(self, ticks: Iterable[Tick]) -> int:
        """Publish, then VERIFY every record was acknowledged.

        **`producer.send()` is fire-and-forget.** It returns a future, and a
        delivery failure lands in that future and nowhere else. Not checking it
        is the reason this repository lost 42 of 3004 ticks with `acks="all"`
        and a `flush()` and no error anywhere: `flush()` waits for the batches
        to be *attempted*, not for them to have *succeeded*.

        Silently dropping 1.4% of a feed is the worst failure a feature
        pipeline can have. Every window is slightly wrong, nothing errors, and
        it surfaces weeks later as a model that underperforms for no visible
        reason - which is exactly the train/serve skew this repository exists
        to measure, arriving through the transport instead of the windowing.

        So the futures are collected and resolved, and a failure raises.
        """
        count = 0
        for tick in ticks:
            self.send(tick)
            count += 1

        self._producer.flush()

        failures = []
        for future in self._futures:
            try:
                future.get(timeout=30)
            except Exception as error:                      # noqa: BLE001
                failures.append(error)
        self._futures = []

        if failures:
            raise RuntimeError(
                f"{len(failures)} of {count} records were not acknowledged; "
                f"first error: {failures[0]!r}. Producing without checking the "
                f"futures would have lost them silently."
            )
        return count


@dataclass
class TickConsumer:
    """Reads ticks back, in partition order."""

    bootstrap_servers: str = DEFAULT_BOOTSTRAP
    topic: str = DEFAULT_TOPIC
    group_id: Optional[str] = None
    #: Milliseconds of silence before the iterator stops. Finite by default so
    #: a demo terminates; a real consumer runs forever.
    timeout_ms: int = 10_000
    from_beginning: bool = True

    def __iter__(self) -> Iterator[Tick]:
        _, KafkaConsumer = _require_kafka()
        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="earliest" if self.from_beginning else "latest",
            consumer_timeout_ms=self.timeout_ms,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            # Offsets are committed by the CALLER, after the tick has actually
            # been folded into the feature state. Auto-commit would acknowledge
            # a tick the consumer had merely received, so a crash between
            # receipt and processing loses it silently - at-most-once dressed
            # up as at-least-once.
            enable_auto_commit=False,
        )
        try:
            for message in consumer:
                yield Tick.from_dict(message.value)
        finally:
            consumer.close()

    def drain(self) -> List[Tick]:
        return list(self)


def ensure_topic(bootstrap_servers: str = DEFAULT_BOOTSTRAP,
                 topic: str = DEFAULT_TOPIC, partitions: int = 3,
                 wait_s: float = 15.0) -> bool:
    """Create the topic if absent, and WAIT until it is usable. True if created.

    Three partitions rather than one so the per-symbol keying above is actually
    doing something. With one partition ordering is global and the keying would
    be untested.

    **The wait is not defensive padding - it fixes real message loss.**
    `create_topics` is asynchronous: it returns once the controller has accepted
    the request, not once every broker can serve the partitions. Producing
    immediately afterwards cost 42 of 3004 ticks in this repository's own demo -
    silently, with `acks="all"` and a `flush()`, because the records were
    rejected before they were ever assigned a partition.

    Silent loss of 1.4% of a feed is the worst possible failure for a feature
    pipeline: every window is slightly wrong, nothing errors, and the drift only
    shows up as a model that underperforms for no visible reason.

    So this polls until the topic and all its partitions have leaders, and
    raises if they never do. A create call that has not converged is not a
    created topic.
    """
    import time

    try:
        from kafka.admin import KafkaAdminClient, NewTopic
    except ImportError:                                     # pragma: no cover
        _require_kafka()
        raise

    admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    created = False
    try:
        if topic not in admin.list_topics():
            admin.create_topics([NewTopic(name=topic, num_partitions=partitions,
                                          replication_factor=1)])
            created = True

        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                described = admin.describe_topics([topic])
            except Exception:                               # noqa: BLE001
                described = []
            for entry in described:
                found = entry.get("partitions") or []
                if len(found) >= partitions and all(
                        part.get("leader", -1) >= 0 for part in found):
                    return created
            time.sleep(0.25)

        raise RuntimeError(
            f"topic '{topic}' did not become usable within {wait_s}s. "
            f"Producing anyway would lose records silently."
        )
    finally:
        admin.close()


def round_trip(ticks: Sequence[Tick],
               bootstrap_servers: str = DEFAULT_BOOTSTRAP,
               topic: str = DEFAULT_TOPIC) -> List[Tick]:
    """Publish and read back. Used by the demo to prove the transport works.

    The returned order is partition order, NOT the order sent - which is the
    point. A consumer never sees a global ordering, and any logic that assumes
    one is broken the moment a topic has more than one partition.
    """
    ensure_topic(bootstrap_servers, topic)
    with TickProducer(bootstrap_servers, topic) as producer:
        producer.send_all(ticks)
    return TickConsumer(bootstrap_servers, topic).drain()
