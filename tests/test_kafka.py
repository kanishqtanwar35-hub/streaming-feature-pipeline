"""Kafka integration. Skipped automatically when no broker is running.

The whole point of keeping the transport out of the feature logic is that these
tests are the ONLY ones that need infrastructure - 39 others do not.
"""

import time

import pytest

from featurepipe.batch import BatchFeatures
from featurepipe.parity import compare
from featurepipe.streaming import StreamingFeatures
from featurepipe.ticks import (
    SYMBOLS,
    StreamProfile,
    deduplicate,
    evaluation_times,
    generate_multi,
)

pytest.importorskip("kafka", reason="kafka-python-ng is an optional dependency")

BOOTSTRAP = "localhost:9092"


def _broker_available() -> bool:
    from kafka.admin import KafkaAdminClient
    try:
        admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP,
                                 request_timeout_ms=3000)
        admin.close()
        return True
    except Exception:                                       # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.kafka,
    pytest.mark.skipif(
        not _broker_available(),
        reason="no broker; start one with "
               "`docker compose -f docker/compose.yaml up -d`"),
]


@pytest.fixture(scope="module")
def ticks():
    return generate_multi(SYMBOLS[:3], minutes=25,
                          profile=StreamProfile(late_rate=0.06,
                                                max_lateness_s=45.0,
                                                out_of_order_rate=0.12,
                                                duplicate_rate=0.03))


@pytest.fixture
def topic():
    return f"featurepipe_test_{int(time.time() * 1000)}"


def test_nothing_is_lost_in_transit(ticks, topic):
    """The bug this repository actually hit.

    `create_topics` is asynchronous - it returns when the controller ACCEPTS the
    request, not when the partitions have leaders. Producing immediately
    afterwards lost 42 of 3004 ticks, silently, with acks="all" and a flush().

    Silent loss of 1.4% of a feed is the worst failure a feature pipeline can
    have: every window is slightly wrong, nothing errors, and it surfaces only
    as a model that underperforms for no visible reason.
    """
    from featurepipe.kafka_io import round_trip

    received = round_trip(ticks, BOOTSTRAP, topic)
    sent_keys = {t.key for t in ticks}
    received_keys = {t.key for t in received}

    assert sent_keys - received_keys == set(), "ticks were lost in transit"
    assert len(received) == len(ticks)


def test_ensure_topic_waits_for_partitions_to_have_leaders(topic):
    from featurepipe.kafka_io import ensure_topic

    assert ensure_topic(BOOTSTRAP, topic, partitions=3) is True
    # Idempotent: a second call finds it and returns False without recreating.
    assert ensure_topic(BOOTSTRAP, topic, partitions=3) is False


def test_features_are_identical_through_kafka(ticks, topic):
    """The transport must not change the answer.

    Kafka reorders across partitions, so this is a real check on whether the
    streaming path's event-time handling actually tolerates arrival order -
    rather than quietly depending on the order the generator produced.
    """
    from featurepipe.kafka_io import round_trip

    received = round_trip(ticks, BOOTSTRAP, topic)
    times = evaluation_times(ticks, every_s=60.0)
    assert times, "the session must be longer than the longest window"

    worst = max(t.lateness for t in deduplicate(ticks))
    watermark = worst + 5.0

    through_kafka = StreamingFeatures(watermark_s=watermark).consume(received, times)
    replayed = BatchFeatures(ticks, replay=True)
    reference = []
    for symbol in replayed.symbols():
        reference.extend(replayed.rows(symbol, times))

    report = compare(through_kafka, reference, "kafka vs replayed batch", 1e-9)
    assert report.agree, report.summary()


def test_a_symbols_ticks_stay_on_one_partition(ticks, topic):
    """Ordering in Kafka is per-partition. Keying by symbol is what keeps one
    symbol's ticks in order; without it they scatter and a mild out-of-order
    problem becomes a severe one."""
    from kafka import KafkaConsumer

    from featurepipe.kafka_io import TickProducer, ensure_topic

    ensure_topic(BOOTSTRAP, topic, partitions=3)
    with TickProducer(BOOTSTRAP, topic) as producer:
        producer.send_all(ticks)

    consumer = KafkaConsumer(topic, bootstrap_servers=BOOTSTRAP,
                             auto_offset_reset="earliest",
                             consumer_timeout_ms=15000,
                             enable_auto_commit=False)
    partitions = {}
    try:
        for message in consumer:
            symbol = message.key.decode("utf-8")
            partitions.setdefault(symbol, set()).add(message.partition)
    finally:
        consumer.close()

    for symbol, found in partitions.items():
        assert len(found) == 1, f"{symbol} spread across partitions {found}"
