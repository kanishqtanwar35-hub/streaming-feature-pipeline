"""The third implementation: the same features in PySpark.

Optional, like Kafka. Nothing else imports this module.

**Why a third implementation at all.** Two implementations that agree might
both be wrong in the same way — they were written by the same person, from the
same reading of the spec, on the same afternoon. A third, written against a
completely different execution model (a distributed query planner rather than a
Python loop), is a genuinely independent check on whether the *specification*
is unambiguous.

It found one thing immediately, which is the reason the spec spells out its
window convention: Spark's `window()` produces `[start, end)` — left-closed,
right-**open** — while `spec.WindowSpec.contains` defines `(end - w, end]`.
Left as defaults, the two implementations disagree by exactly one tick whenever
one lands on a boundary, and the disagreement looks like a rounding error.
The range join below implements the spec's convention explicitly rather than
using `window()`.

**Running it.** PySpark needs a JVM, and this machine has no Java — `pyspark`
installs fine and then dies with `JAVA_GATEWAY_EXITED`. So locally it runs in
Docker:

    docker run --rm -v "$PWD:/work" -w /work apache/spark:latest \\
        /opt/spark/bin/spark-submit src/featurepipe/spark_batch.py

On CI it runs natively, because the Ubuntu runners ship a JDK. Both paths are
free.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from featurepipe.spec import DERIVED, FEATURES, Seconds
from featurepipe.ticks import Tick


def _session(app_name: str = "featurepipe"):
    from pyspark.sql import SparkSession

    session = (SparkSession.builder
               .master("local[*]")
               .appName(app_name)
               # The UI is a web server nobody looks at in a batch job, and on
               # a constrained machine it is a port conflict waiting to happen.
               .config("spark.ui.enabled", "false")
               # Default 200 shuffle partitions on a dataset this size means
               # 200 mostly-empty tasks and more scheduling than computation.
               .config("spark.sql.shuffle.partitions", "8")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    return session


def compute(ticks: Sequence[Tick], times: Sequence[Seconds],
            session=None) -> List[Dict[str, object]]:
    """Feature rows for every (symbol, time), computed in Spark.

    A range join rather than `window()`. `window()` tumbles or slides on a fixed
    grid with `[start, end)` boundaries; the spec asks for a window ending at an
    arbitrary evaluation instant, right-closed. Forcing `window()` to do that
    means offsetting the grid and then living with the wrong boundary - so the
    join states the condition directly and the SQL reads like the spec.
    """
    from pyspark.sql import functions as F

    owns_session = session is None
    session = session or _session()

    try:
        tick_rows = [
            (t.symbol, float(t.event_time), float(t.arrival_time),
             float(t.ltp), float(t.ltq), int(t.seq))
            for t in ticks
        ]
        ticks_df = session.createDataFrame(
            tick_rows,
            ["symbol", "event_time", "arrival_time", "ltp", "ltq", "seq"],
        )
        # Deduplicate on (symbol, seq), exactly as the streaming consumer does.
        # At-least-once delivery means the stored history contains
        # redeliveries, and a batch job that keeps them double-counts precisely
        # the ticks the consumer skipped.
        ticks_df = ticks_df.dropDuplicates(["symbol", "seq"])

        symbols = [r["symbol"] for r in
                   ticks_df.select("symbol").distinct().collect()]
        eval_df = session.createDataFrame(
            [(s, float(t)) for s in symbols for t in times],
            ["symbol", "at"],
        )

        joined = eval_df.join(ticks_df, on="symbol", how="left")

        aggregations = []
        for spec in FEATURES:
            if spec.window is None:
                condition = F.col("event_time") <= F.col("at")
            else:
                # (at - w, at] - the spec's convention, written out. Spark's
                # window() would give [start, end) and disagree by one tick at
                # every boundary.
                condition = ((F.col("event_time") > F.col("at") - spec.window.seconds)
                             & (F.col("event_time") <= F.col("at")))

            value = F.when(condition, F.col(spec.source))
            if spec.aggregation == "mean":
                aggregations.append(F.avg(value).alias(spec.name))
            elif spec.aggregation == "sum":
                aggregations.append(F.sum(value).alias(spec.name))
            elif spec.aggregation == "count":
                aggregations.append(F.count(value).cast("double").alias(spec.name))
            elif spec.aggregation == "max":
                aggregations.append(F.max(value).alias(spec.name))
            elif spec.aggregation == "min":
                aggregations.append(F.min(value).alias(spec.name))
            elif spec.aggregation == "last":
                # "last" means the most recent by EVENT time, which is a
                # max-by, not Spark's last() - that returns the last row in
                # whatever order the partition happened to produce.
                aggregations.append(
                    F.max(F.when(condition,
                                 F.struct(F.col("event_time"),
                                          F.col(spec.source)))
                          ).getField(spec.source).alias(spec.name))
            else:
                raise ValueError(f"unknown aggregation '{spec.aggregation}'")

        grouped = joined.groupBy("symbol", "at").agg(*aggregations)

        rows: List[Dict[str, object]] = []
        for record in grouped.collect():
            row: Dict[str, object] = {"symbol": record["symbol"],
                                      "event_time": record["at"]}
            for spec in FEATURES:
                value = record[spec.name]
                if value is None:
                    row[spec.name] = spec.empty_value
                elif spec.aggregation == "count":
                    # count() returns 0 for an empty window, not null, so the
                    # spec's empty_value has to be applied by hand.
                    row[spec.name] = float(value)
                else:
                    row[spec.name] = float(value)
            for spec in DERIVED:
                row[spec.name] = spec.fn(row)          # type: ignore[arg-type]
            rows.append(row)

        rows.sort(key=lambda r: (r["symbol"], r["event_time"]))
        return rows
    finally:
        if owns_session:
            session.stop()


def main() -> int:
    """Run the Spark implementation and compare it with the pure-Python one.

    This is the entry point `spark-submit` calls. It prints a parity verdict,
    and exits non-zero if the two disagree.
    """
    import sys
    sys.path.insert(0, "src")

    from featurepipe.batch import BatchFeatures
    from featurepipe.parity import compare
    from featurepipe.ticks import evaluation_times, generate

    ticks = generate("IDEA", minutes=30, seed=0)
    times = evaluation_times(ticks, every_s=60.0)

    reference = BatchFeatures(ticks, replay=False)
    reference_rows = reference.rows("IDEA", times)
    spark_rows = compute(ticks, times)

    # A small tolerance here, unlike the streaming/replay comparison. Spark
    # sums in a partition-dependent order and floating-point addition is not
    # associative, so bit-identical results are not something either
    # implementation can promise. 1e-9 is far below any value that matters and
    # far above the accumulation error.
    report = compare(spark_rows, reference_rows, "spark vs pure python", 1e-9)
    print(report.summary())
    return 0 if report.agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
