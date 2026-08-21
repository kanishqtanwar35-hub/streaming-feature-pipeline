"""Command line entry point.

    python -m featurepipe.cli spec        the feature definitions
    python -m featurepipe.cli feed        what the synthetic feed does
    python -m featurepipe.cli parity      the headline: do the paths agree?
    python -m featurepipe.cli sweep       what each watermark setting costs
    python -m featurepipe.cli boundary    when replay parity holds, exactly
    python -m featurepipe.cli kafka       publish and consume through a broker

Everything except `kafka` runs with no broker, no JVM and no third-party
package beyond pytest.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from featurepipe import spec
from featurepipe.parity import (
    correctness,
    format_sweep,
    run,
    skew,
    watermark_sweep,
)
from featurepipe.ticks import (
    StreamProfile,
    deduplicate,
    generate,
    generate_multi,
    statistics,
)


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _profile(args) -> StreamProfile:
    return StreamProfile(
        late_rate=args.late_rate,
        max_lateness_s=args.max_lateness,
        out_of_order_rate=args.out_of_order,
        duplicate_rate=args.duplicates,
    )


def _ticks(args):
    if args.symbols > 1:
        from featurepipe.ticks import SYMBOLS
        return generate_multi(SYMBOLS[:args.symbols], minutes=args.minutes,
                              profile=_profile(args))
    return generate("IDEA", minutes=args.minutes, profile=_profile(args))


def cmd_spec(args) -> int:
    print(spec.describe())
    print()
    print(f"longest window: {spec.MAX_WINDOW_S:.0f}s - a streaming consumer "
          f"cannot discard a tick older than this, and a batch job has no "
          f"trustworthy value until this much history exists.")
    return 0


def cmd_feed(args) -> int:
    ticks = _ticks(args)
    stats = statistics(ticks)
    print("The synthetic feed reproduces the three arrival pathologies that")
    print("actually happen. A parity test against perfectly ordered data")
    print("proves nothing.")
    print()
    for key, value in stats.items():
        print(f"  {key:<20} {value:>10.2f}")
    return 0


def cmd_parity(args) -> int:
    ticks = _ticks(args)
    result = run(ticks, watermark_s=args.watermark)

    print(result.stream_stats.summary())
    print()

    check = correctness(result)
    print(check.summary())
    print("  ^ streaming vs a batch job constrained to the same information.")
    print("    Must be EXACT: same ticks, same window definition, so any")
    print("    disagreement is a bug in one implementation, not a trade-off.")
    print()

    divergence = skew(result)
    print(divergence.summary())
    print("  ^ streaming vs a NAIVE batch job over complete history - the")
    print("    numbers that end up in the training set. This gap is")
    print("    train/serve skew, and nothing anywhere reports an error.")

    if result.stream_stats.dropped_late == 0 and not divergence.agree:
        print()
        print("    Note: zero ticks were DROPPED, and the paths still disagree.")
        print("    Skew does not require dropping anything - a live system")
        print("    simply has not received late data yet, while the batch job")
        print("    has. The watermark controls drops; it cannot control that.")

    return 0 if check.agree else 1


def cmd_sweep(args) -> int:
    ticks = _ticks(args)
    print("Two columns move in opposite directions and no setting optimises")
    print("both. Picking a watermark without this table is guessing.")
    print()
    print(format_sweep(watermark_sweep(ticks)))
    print()
    print("dropped%  every dropped tick is a difference between what training")
    print("          saw and what serving saw")
    print("latency   how long a window must wait before it can be emitted")
    return 0


def cmd_boundary(args) -> int:
    """Find the watermark at which replay parity starts to hold."""
    ticks = _ticks(args)
    worst = max(t.lateness for t in deduplicate(ticks))
    print(f"maximum lateness in this feed: {worst:.2f}s")
    print()

    candidates = sorted({0.0, 15.0, 30.0, 45.0, 60.0,
                         round(worst - 5, 1), round(worst - 1, 1),
                         round(worst, 1), round(worst + 1, 1), 180.0})
    print(f"{'watermark':>10} {'dropped':>8}  replay parity")
    print("-" * 40)
    first_exact = None
    for watermark in candidates:
        if watermark < 0:
            continue
        result = run(ticks, watermark_s=watermark)
        exact = correctness(result).agree
        if exact and first_exact is None:
            first_exact = watermark
        print(f"{watermark:>9.1f}s {result.stream_stats.dropped_late:>8} "
              f" {'EXACT' if exact else 'mismatch'}")

    print()
    print(f"Replay parity holds from {first_exact:.1f}s, and the feed's")
    print(f"maximum lateness is {worst:.2f}s. That is the law:")
    print()
    print("  parity holds if and only if the watermark >= actual max lateness")
    print()
    print("Below it the consumer drops ticks, and once a tick is dropped NO")
    print("batch job can reproduce the streaming output. The pipeline is")
    print("lossy, and the loss is silent.")
    return 0


def cmd_kafka(args) -> int:
    from featurepipe.kafka_io import round_trip

    ticks = _ticks(args)
    print(f"publishing {len(ticks)} ticks to {args.topic} at {args.bootstrap}...")
    try:
        received = round_trip(ticks, args.bootstrap, args.topic)
    except Exception as error:                              # noqa: BLE001
        print(f"\nbroker unavailable: {error}", file=sys.stderr)
        print("start one with: docker compose -f docker/compose.yaml up -d",
              file=sys.stderr)
        return 2

    print(f"consumed {len(received)} ticks back")

    sent_keys = {t.key for t in ticks}
    got_keys = {t.key for t in received}
    print(f"  unique keys sent {len(sent_keys)}, received {len(got_keys)}")
    print(f"  lost: {len(sent_keys - got_keys)}")

    reordered = sum(1 for a, b in zip(received, received[1:])
                    if b.event_time < a.event_time)
    print(f"  out of event order after the round trip: {reordered}")
    print()
    print("Order is per-PARTITION, never per-topic. Ticks are keyed by symbol")
    print("so each symbol lands on one partition in order; without the key")
    print("they scatter and a mild out-of-order problem becomes a severe one.")

    from featurepipe.parity import compare
    from featurepipe.streaming import StreamingFeatures
    from featurepipe.ticks import evaluation_times
    from featurepipe.batch import BatchFeatures

    times = evaluation_times(received, every_s=60.0)
    if times:
        stream = StreamingFeatures(watermark_s=args.watermark)
        through_kafka = stream.consume(received, times)
        direct = StreamingFeatures(watermark_s=args.watermark).consume(ticks, times)
        report = compare(through_kafka, direct, "through kafka vs direct", 1e-9)
        print()
        print(report.summary())
    return 0


def main(argv=None) -> int:
    _stdout_utf8()
    parser = argparse.ArgumentParser(
        prog="featurepipe",
        description="one feature definition, three implementations, tested")
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--symbols", type=int, default=1)
    parser.add_argument("--watermark", type=float, default=45.0)
    parser.add_argument("--late-rate", type=float, default=0.08)
    parser.add_argument("--max-lateness", type=float, default=90.0)
    parser.add_argument("--out-of-order", type=float, default=0.15)
    parser.add_argument("--duplicates", type=float, default=0.03)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("spec").set_defaults(func=cmd_spec)
    sub.add_parser("feed").set_defaults(func=cmd_feed)
    sub.add_parser("parity").set_defaults(func=cmd_parity)
    sub.add_parser("sweep").set_defaults(func=cmd_sweep)
    sub.add_parser("boundary").set_defaults(func=cmd_boundary)

    kafka = sub.add_parser("kafka")
    kafka.add_argument("--bootstrap", default="localhost:9092")
    kafka.add_argument("--topic", default="ticks")
    kafka.set_defaults(func=cmd_kafka)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
