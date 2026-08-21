"""One definition, three implementations, and the laws that relate them.

No broker, no JVM, no third-party package. That is deliberate: a pipeline whose
windowing logic can only be exercised by standing up infrastructure is one where
the windowing logic does not get exercised.
"""

import math

import pytest

from featurepipe.batch import BatchFeatures, read_parquet_like, to_parquet_like
from featurepipe.parity import compare, correctness, run, skew, watermark_sweep
from featurepipe.spec import (
    ALL_NAMES,
    DERIVED_BY_NAME,
    FEATURES_BY_NAME,
    MAX_WINDOW_S,
    WindowSpec,
    aggregate,
    apply_derived,
    empty_row,
)
from featurepipe.streaming import StreamingFeatures
from featurepipe.ticks import (
    StreamProfile,
    Tick,
    deduplicate,
    evaluation_times,
    generate,
    generate_multi,
    in_event_order,
    statistics,
)

NOISY = StreamProfile(late_rate=0.08, max_lateness_s=90.0,
                      out_of_order_rate=0.15, duplicate_rate=0.03)


def tick(event_time, ltq=100.0, ltp=10.0, seq=0, arrival=None, symbol="X"):
    return Tick(symbol, float(event_time),
                float(event_time if arrival is None else arrival),
                ltp, ltq, seq)


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------

def test_the_window_is_left_open_right_closed():
    """The line where implementations silently diverge. Pandas' rolling,
    Spark's window() and a hand-written deque each default to a DIFFERENT
    convention, and a one-tick disagreement at a boundary is enough to make two
    implementations of the same feature produce different numbers."""
    window = WindowSpec(120.0)
    assert not window.contains(880.0, 1000.0), "the left edge is OUT"
    assert window.contains(880.1, 1000.0)
    assert window.contains(1000.0, 1000.0), "the right edge is IN"
    assert not window.contains(1000.1, 1000.0)


def test_an_empty_average_is_none_not_zero():
    """Zero is a VALUE - it claims the average trade size was nothing. Absence
    of ticks is not a claim about the market, and a model trained on rows where
    absence was encoded as 0.0 cannot distinguish a quiet market from a dead
    feed."""
    assert FEATURES_BY_NAME["ltq_avg_2m"].empty_value is None
    assert empty_row()["ltq_avg_2m"] is None


def test_an_empty_sum_legitimately_is_zero():
    """Unlike the average: no trades really does mean zero volume."""
    assert FEATURES_BY_NAME["ltq_sum_2m"].empty_value == 0.0
    assert empty_row()["ltq_sum_2m"] == 0.0


def test_aggregate_returns_none_for_nothing():
    for how in ("mean", "sum", "count", "last", "max", "min"):
        assert aggregate([], how) is None


def test_aggregate_rejects_an_unknown_aggregation():
    with pytest.raises(ValueError):
        aggregate([1.0], "median")


def test_a_ratio_propagates_absence_instead_of_inventing_one():
    """The predecessor defaults this to 1.0 when the denominator is missing,
    which reads as 'no change' - a confident, wrong statement. If either side is
    unknown the ratio is unknown."""
    assert apply_derived({"ltq_avg_2m": 5.0, "ltq_avg_5m": None})["ltq_ratio_2_5"] is None
    assert apply_derived({"ltq_avg_2m": None, "ltq_avg_5m": 5.0})["ltq_ratio_2_5"] is None
    assert apply_derived({"ltq_avg_2m": 5.0, "ltq_avg_5m": 0.0})["ltq_ratio_2_5"] is None


def test_the_ratio_is_computed_when_both_sides_exist():
    assert apply_derived({"ltq_avg_2m": 6.0,
                          "ltq_avg_5m": 3.0})["ltq_ratio_2_5"] == 2.0


def test_the_longest_window_is_what_retention_must_cover():
    assert MAX_WINDOW_S == 1200.0


# ---------------------------------------------------------------------------
# The feed
# ---------------------------------------------------------------------------

def test_the_generator_is_reproducible():
    a = generate("IDEA", minutes=10, seed=3)
    b = generate("IDEA", minutes=10, seed=3)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_different_seeds_give_different_streams():
    a = generate("IDEA", minutes=10, seed=1)
    b = generate("IDEA", minutes=10, seed=2)
    assert [t.event_time for t in a] != [t.event_time for t in b]


def test_ticks_come_back_in_ARRIVAL_order():
    """What a consumer sees. Returning event-ordered data would quietly do the
    streaming implementation's hardest job for it."""
    ticks = generate("IDEA", minutes=15, profile=NOISY)
    arrivals = [t.arrival_time for t in ticks]
    assert arrivals == sorted(arrivals)


def test_the_feed_actually_contains_the_pathologies():
    """A parity test against perfectly ordered synthetic data proves nothing."""
    stats = statistics(generate("IDEA", minutes=30, profile=NOISY))
    assert stats["duplicates"] > 0
    assert stats["out_of_order"] > 0
    assert stats["max_lateness_s"] > 10.0


def test_event_order_differs_from_arrival_order():
    ticks = generate("IDEA", minutes=20, profile=NOISY)
    assert [t.seq for t in ticks] != [t.seq for t in in_event_order(ticks)]


def test_deduplicate_keeps_the_first_occurrence():
    """A redelivery is the same trade; preferring the later copy would move its
    arrival time for no reason."""
    first = tick(100.0, seq=1, arrival=100.0)
    second = tick(100.0, seq=1, arrival=105.0)
    assert deduplicate([first, second]) == [first]


def test_multiple_symbols_are_independent_streams():
    """Same walk under different names would make a per-symbol grouping bug
    invisible, because every group would agree by construction."""
    ticks = generate_multi(("IDEA", "PNB"), minutes=10)
    idea = [t.event_time for t in ticks if t.symbol == "IDEA"]
    pnb = [t.event_time for t in ticks if t.symbol == "PNB"]
    assert idea and pnb and idea != pnb


def test_evaluation_times_skip_the_cold_start():
    """Every window is partial before the stream has run for its full length,
    so comparing implementations there measures the cold start rather than the
    implementations."""
    ticks = generate("IDEA", minutes=40)
    times = evaluation_times(ticks)
    assert times
    assert times[0] >= in_event_order(ticks)[0].event_time + MAX_WINDOW_S


# ---------------------------------------------------------------------------
# The streaming path
# ---------------------------------------------------------------------------

def test_a_simple_window_is_computed_correctly():
    stream = StreamingFeatures(watermark_s=1000.0)
    for i, quantity in enumerate([10.0, 20.0, 30.0], start=1):
        stream.offer(tick(1000.0 + i, ltq=quantity, seq=i))
    row = stream.features_at("X", 1003.0)
    assert row["ltq_avg_2m"] == pytest.approx(20.0)
    assert row["ltq_sum_2m"] == pytest.approx(60.0)
    assert row["tick_count_2m"] == 3.0


def test_a_tick_on_the_left_edge_is_excluded():
    stream = StreamingFeatures(watermark_s=1000.0)
    stream.offer(tick(880.0, ltq=999.0, seq=1))     # exactly at end - 120
    stream.offer(tick(950.0, ltq=10.0, seq=2))
    assert stream.features_at("X", 1000.0)["ltq_avg_2m"] == pytest.approx(10.0)


def test_a_tick_on_the_right_edge_is_included():
    stream = StreamingFeatures(watermark_s=1000.0)
    stream.offer(tick(1000.0, ltq=42.0, seq=1))
    assert stream.features_at("X", 1000.0)["ltq_avg_2m"] == pytest.approx(42.0)


def test_duplicates_are_rejected_and_counted():
    stream = StreamingFeatures(watermark_s=1000.0)
    assert stream.offer(tick(1000.0, seq=7)) is True
    assert stream.offer(tick(1000.0, seq=7)) is False
    assert stream.stats.duplicates == 1
    assert stream.stats.accepted == 1


def test_a_late_tick_beyond_the_watermark_is_dropped_and_COUNTED():
    """A pipeline that silently discards late data reports healthy while its
    features quietly diverge from training. The count is the metric that
    catches it."""
    stream = StreamingFeatures(watermark_s=30.0)
    stream.offer(tick(1000.0, seq=1))
    assert stream.offer(tick(900.0, seq=2)) is False
    assert stream.stats.dropped_late == 1
    assert stream.stats.drop_rate > 0


def test_a_late_tick_inside_the_watermark_is_accepted():
    stream = StreamingFeatures(watermark_s=200.0)
    stream.offer(tick(1000.0, seq=1))
    assert stream.offer(tick(900.0, seq=2)) is True
    assert stream.stats.out_of_order == 1


def test_out_of_order_ticks_are_stored_in_event_order():
    stream = StreamingFeatures(watermark_s=1000.0)
    for event_time, seq in [(1000.0, 1), (900.0, 2), (950.0, 3)]:
        stream.offer(tick(event_time, seq=seq))
    stored = [t.event_time for t in stream.state["X"].ticks]
    assert stored == sorted(stored)


def test_an_unwindowed_feature_does_not_see_the_future():
    """`ltp_last` at time T must be the last price at or before T, not the last
    price the consumer happens to hold. Otherwise a historical query leaks the
    future and replay disagrees with live."""
    stream = StreamingFeatures(watermark_s=1000.0)
    stream.offer(tick(1000.0, ltp=10.0, seq=1))
    stream.offer(tick(1100.0, ltp=99.0, seq=2))
    assert stream.features_at("X", 1050.0)["ltp_last"] == pytest.approx(10.0)


def test_an_unknown_symbol_gives_the_empty_row():
    assert StreamingFeatures().features_at("NOPE", 1000.0) == empty_row()


def test_retention_covers_the_longest_window_plus_the_watermark():
    """A late tick arriving to find its window already evicted is a bug that
    looks exactly like a watermark that is too short."""
    stream = StreamingFeatures(watermark_s=45.0)
    assert stream.retain_s > MAX_WINDOW_S + 45.0


# ---------------------------------------------------------------------------
# The batch path
# ---------------------------------------------------------------------------

def test_the_naive_batch_job_sees_everything():
    ticks = [tick(1000.0, ltq=10.0, seq=1, arrival=1000.0),
             tick(1010.0, ltq=20.0, seq=2, arrival=9999.0)]
    row = BatchFeatures(ticks, replay=False).features_at("X", 1020.0)
    assert row["ltq_avg_2m"] == pytest.approx(15.0)


def test_the_replayed_batch_job_respects_arrival_time():
    ticks = [tick(1000.0, ltq=10.0, seq=1, arrival=1000.0),
             tick(1010.0, ltq=20.0, seq=2, arrival=9999.0)]
    row = BatchFeatures(ticks, replay=True).features_at("X", 1020.0)
    assert row["ltq_avg_2m"] == pytest.approx(10.0), \
        "the late tick had not arrived yet"


def test_the_batch_job_deduplicates_too():
    """A batch job that keeps redeliveries double-counts exactly the ticks the
    streaming consumer skipped, producing a disagreement that looks like a
    windowing bug."""
    ticks = [tick(1000.0, ltq=10.0, seq=1), tick(1000.0, ltq=10.0, seq=1)]
    assert BatchFeatures(ticks).features_at("X", 1010.0)["tick_count_2m"] == 1.0


def test_the_feature_store_round_trips(tmp_path):
    ticks = generate("IDEA", minutes=30)
    rows = BatchFeatures(ticks).rows("IDEA", evaluation_times(ticks))
    path = tmp_path / "features.ndjson"
    to_parquet_like(rows, path)
    assert read_parquet_like(path) == rows


def test_the_feature_store_uses_lf_endings(tmp_path):
    path = tmp_path / "f.ndjson"
    to_parquet_like([{"symbol": "X", "event_time": 1.0}], path)
    assert b"\r\n" not in path.read_bytes()


# ---------------------------------------------------------------------------
# The laws
# ---------------------------------------------------------------------------

def test_querying_a_finished_buffer_is_NOT_the_same_as_streaming():
    """The bug this repository nearly shipped.

    Offering the whole stream and then querying past times answers with ticks
    that had not arrived then - the query sees the future. Parity against a
    correctly replayed batch job then fails, and the natural reading is 'the
    batch job is broken'.

    Same mistake as grounding a camera detection against the robot's CURRENT
    pose instead of the pose when the frame was taken.
    """
    ticks = generate("IDEA", minutes=40, profile=NOISY)
    times = evaluation_times(ticks, every_s=60.0)

    retrospective = StreamingFeatures(watermark_s=1000.0)
    retrospective.offer_all(ticks)
    wrong = retrospective.rows("IDEA", times)

    right = StreamingFeatures(watermark_s=1000.0).consume(ticks, times)
    replayed = BatchFeatures(ticks, replay=True).rows("IDEA", times)

    assert compare(right, replayed, "correct", 0.0).agree
    assert not compare(wrong, replayed, "retrospective", 0.0).agree


def test_streaming_and_replayed_batch_agree_EXACTLY():
    """Same ticks, same window definition, so bit-identical. A tolerance here
    would hide precisely the boundary bugs this comparison exists to catch."""
    ticks = generate("IDEA", minutes=45, profile=NOISY)
    worst = max(t.lateness for t in deduplicate(ticks))
    result = run(ticks, watermark_s=worst + 1.0)
    report = correctness(result)
    assert report.agree, report.summary()


def test_parity_holds_IF_AND_ONLY_IF_the_watermark_covers_max_lateness():
    """The law. Below the feed's actual maximum lateness the consumer drops
    ticks, and once a tick is dropped NO batch job can reproduce the streaming
    output - the pipeline is lossy, and the loss is silent."""
    ticks = generate("IDEA", minutes=45, profile=NOISY)
    worst = max(t.lateness for t in deduplicate(ticks))

    assert correctness(run(ticks, watermark_s=worst + 1.0)).agree
    assert not correctness(run(ticks, watermark_s=worst - 10.0)).agree


def test_skew_exists_even_when_nothing_is_dropped():
    """The subtle half, and the more common one. A live system has not RECEIVED
    late data yet, while the batch job has. The watermark controls drops; it
    cannot control that."""
    ticks = generate("IDEA", minutes=45, profile=NOISY)
    worst = max(t.lateness for t in deduplicate(ticks))
    result = run(ticks, watermark_s=worst + 1.0)

    assert result.stream_stats.dropped_late == 0
    assert not skew(result).agree, \
        "training would use numbers serving never had"


def test_a_longer_watermark_never_drops_more():
    rows = watermark_sweep(generate("IDEA", minutes=40, profile=NOISY),
                           watermarks=(0.0, 30.0, 60.0, 120.0))
    dropped = [r["dropped"] for r in rows]
    assert dropped == sorted(dropped, reverse=True)


def test_the_sweep_reports_both_sides_of_the_trade():
    rows = watermark_sweep(generate("IDEA", minutes=30, profile=NOISY),
                           watermarks=(0.0, 60.0))
    assert rows[0]["dropped"] > rows[-1]["dropped"], "short drops more"
    assert rows[0]["latency_s"] < rows[-1]["latency_s"], "long waits longer"


def test_every_feature_is_compared():
    result = run(generate("IDEA", minutes=35, profile=NOISY))
    assert set(correctness(result).features) == set(ALL_NAMES)
    assert len(ALL_NAMES) == 12


def test_multiple_symbols_stay_separate_end_to_end():
    """A grouping bug would show as one symbol's ticks leaking into another's
    windows."""
    ticks = generate_multi(("IDEA", "PNB", "SUZLON"), minutes=35)
    worst = max(t.lateness for t in deduplicate(ticks))
    result = run(ticks, watermark_s=worst + 1.0)
    assert correctness(result).agree
    assert len({r["symbol"] for r in result.streaming_rows}) == 3
