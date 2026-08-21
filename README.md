# Streaming Feature Pipeline

**One feature definition. Three implementations. A test that they agree.**

Train/serve skew is the most expensive bug in applied ML because nothing errors:
the training job and the serving job compute "the same" feature slightly
differently, offline metrics look fine, and the model behaves differently in
production for no visible reason.

This measures it, explains exactly where it comes from, and proves the fix.

**Status:** 43 tests. 39 of them need no broker, no JVM and no third-party
package at all.

---

## The real problem, from a real repository

[`nse-realtime-screener`](https://github.com/kanishqtanwar35-hub/nse-realtime-screener)
documents this about itself:

> *"The model trains on trades harvested from historical bars, which carry no
> tick data. So `ltq_*`, `bid_ask_spread_pct`, `order_book_imbalance` and
> `depth_ratio` are **zero in training but populated live**."*

That is a documented, unfixed train/serve skew in a working project. This
repository is the fix, generalised: make the feature definition a single
artifact every path must satisfy, then **test that the paths agree**.

---

## The result

```bash
python -m featurepipe.cli parity
```

```
1705 ticks, 1571 accepted, 50 duplicate, 0 dropped late (0.00%),
  128 out of order, max lateness 88.2s

streaming vs batch (replay): AGREE on all 12 features across 40 rows
  ^ same information, so this must be EXACT. Any disagreement is a bug.

streaming vs batch (naive): 8 of 12 features disagree across 40 rows
  feature             rate    max abs   max rel
  ltq_avg_2m        22.5%     4.9608     2.0%
  ltq_sum_2m        22.5%   610.0000     3.3%
  ltq_ratio_2_5     22.5%     0.0121     1.3%
  ...
  ^ this gap is train/serve skew, and nothing anywhere reports an error.

  Note: zero ticks were DROPPED, and the paths still disagree.
```

**That last line is the point most treatments miss.** Skew does not require
losing anything. A live system simply has not *received* late data yet, while a
batch job over complete history has. The watermark controls drops; it cannot
control that.

### Three implementations, independently written

| implementation | execution model | verified |
|---|---|---|
| `streaming.py` | incremental, arrival order, watermarked | — |
| `batch.py` | pure Python over sorted history | exact match vs streaming |
| `spark_batch.py` | PySpark 4.2, distributed query planner | `spark vs pure python: AGREE on all 12 features` |

Two implementations that agree might both be wrong the same way — same author,
same reading of the spec, same afternoon. A third written against a completely
different execution model is an independent check on whether the *specification*
is unambiguous.

It earned its place immediately: Spark's `window()` is `[start, end)` —
left-closed, **right-open** — while the spec says `(end - w, end]`. Left as
defaults they disagree by exactly one tick at every boundary, and it looks like
a rounding error. `spark_batch.py` implements the spec's convention explicitly.

---

## The law

```bash
python -m featurepipe.cli boundary
```

```
maximum lateness in this feed: 88.24s

 watermark  dropped  replay parity
      0.0s      131  mismatch
     45.0s       51  mismatch
     83.2s        4  mismatch
     87.2s        2  mismatch
     88.2s        0  EXACT
    180.0s        0  EXACT
```

> **Replay parity holds if and only if the watermark ≥ the feed's actual
> maximum lateness.**

Below it the consumer drops ticks, and **once a tick is dropped no batch job can
reproduce the streaming output**. The pipeline is lossy, and the loss is silent.

`test_parity_holds_IF_AND_ONLY_IF_the_watermark_covers_max_lateness` asserts
both directions.

### And what that costs

```bash
python -m featurepipe.cli sweep
```

| watermark | latency | dropped | dropped % | features disagreeing |
|---|---|---|---|---|
| 0s | 0s | 131 | 7.68% | 10 |
| 30s | 30s | 74 | 4.34% | 8 |
| 60s | 60s | 34 | 1.99% | 8 |
| 120s | 120s | 0 | 0.00% | 8 |

Two columns move in opposite directions and **no setting optimises both**. A
signal that decays in seconds cannot afford a 120-second watermark; a pipeline
whose features must match training cannot afford to drop 4% of its ticks.
Picking a number without this table is guessing — the same argument
[`model-drift-monitor`](https://github.com/kanishqtanwar35-hub/model-drift-monitor)
makes about PSI thresholds.

---

## Where the numbers actually come from

Every window is over **event time** — when the exchange stamped the tick — never
processing time. Keyed on arrival, a replay of the same data produces different
features, which makes the pipeline irreproducible and every backtest
meaningless.

The synthetic feed reproduces the three arrival pathologies that actually
happen, because a parity test against perfectly ordered data proves nothing:

- **out of order** — routine, mostly harmless
- **late** — arrives after its window may already have been emitted
- **duplicate** — at-least-once is what a broker actually guarantees

Design decisions that are decisions:

- **The window is `(end - w, end]`.** Stated once, in the spec, because pandas'
  `rolling`, Spark's `window()` and a hand-written deque each default to a
  *different* convention.
- **An empty average is `None`, not `0.0`.** Zero is a *value* — it claims the
  average trade size was nothing. A model trained where absence was encoded as
  zero cannot distinguish a quiet market from a dead feed. An empty *sum* is
  legitimately `0.0`, and the spec says which is which per feature.
- **A ratio with a missing side is `None`.** The predecessor defaults it to
  `1.0`, which reads as "no change" — a confident, wrong statement.
- **Dropped ticks are counted, not swallowed.** `dropped_late` is the direct
  measure of skew being introduced, and it is the number to alert on.

---

## Quickstart

```bash
git clone https://github.com/kanishqtanwar35-hub/streaming-feature-pipeline
cd streaming-feature-pipeline
export PYTHONPATH=src

pytest -q                              # 39 tests, zero dependencies
python -m featurepipe.cli spec         # the feature definitions
python -m featurepipe.cli feed         # what the synthetic feed does
python -m featurepipe.cli parity       # do the paths agree?
python -m featurepipe.cli sweep        # what each watermark costs
python -m featurepipe.cli boundary     # the law, demonstrated
```

**With Kafka** (free, one container, ~285 MB):

```bash
docker compose -f docker/compose.yaml up -d
pip install kafka-python-ng
python -m featurepipe.cli kafka        # publish, consume, compare
pytest -q                              # now 43 tests
```

**With Spark.** It needs a JVM. On a machine without Java, `pyspark` installs
fine and then dies with `JAVA_GATEWAY_EXITED` — so run it in Docker:

```bash
docker run --rm -e PYTHONPATH=/work/src -v "$PWD:/work" -w /work \
  apache/spark:latest /opt/spark/bin/spark-submit src/featurepipe/spark_batch.py
# -> spark vs pure python: AGREE on all 12 features across 10 rows
```

On CI it runs natively, because the Ubuntu runners ship a JDK. Both paths free.

---

## Two bugs worth reading about

### The streaming path was answering with information it did not have

The parity check failed on 10 of 12 features and the obvious reading was "the
batch job is broken". It was not.

The harness offered the *entire* stream to the consumer and then queried past
times. By then the buffer held ticks that had not arrived at the time being
asked about — so the streaming path answered with the future, and disagreed with
a correctly replayed batch job.

A real consumer has no such option: it emits a row for time T when the clock
passes T, using only what has arrived. `consume()` does that, and
`test_querying_a_finished_buffer_is_NOT_the_same_as_streaming` pins both
behaviours so the wrong one cannot come back.

It is the same mistake as grounding a camera detection against a robot's
*current* pose instead of the pose when the frame was taken. A retrospective
query into live state silently borrows later knowledge.

### 42 of 3004 ticks vanished between the producer and the broker

With `acks="all"` and a `flush()`, and no error anywhere. Two causes, both
worth knowing:

1. **`create_topics` is asynchronous.** It returns when the controller accepts
   the request, not when the partitions have leaders. `ensure_topic` now polls
   until every partition has one, and raises if they never do.
2. **`producer.send()` is fire-and-forget.** It returns a future, and a delivery
   failure lands in that future and nowhere else. `flush()` waits for batches to
   be *attempted*, not to have *succeeded*. Worse, `retries` defaults to **0**,
   so a transient `NOT_LEADER_FOR_PARTITION` on a new topic drops the record
   permanently.

`send_all` now collects every future, resolves it, and raises on failure —
turning silent loss into a loud error. `retries=10` with
`max_in_flight_requests_per_connection=1`, because retries with multiple
in-flight requests reorder a partition, and per-symbol ordering is the entire
reason for keying by symbol.

Losing 1.4% of a feed silently is the worst failure a feature pipeline can have:
every window is slightly wrong, nothing errors, and it surfaces weeks later as a
model that underperforms for no visible reason. Which is train/serve skew again,
arriving through the transport instead of the windowing.

---

## Limitations, stated plainly

- **The feed is synthetic.** Deliberately: the parity law needs a known maximum
  lateness to be demonstrable, and no real feed tells you its own. The numbers
  characterise the *method*, not any real market.
- **Single-node everything.** One Kafka broker, `local[*]` Spark. The
  correctness properties hold at any scale; the *performance* claims here are
  none.
- **No state backend.** The streaming consumer keeps its window buffer in
  memory, so a restart cold-starts. Production wants RocksDB or a changelog
  topic, and the eviction/retention logic here is what such a backend would key
  off.
- **Offsets are never committed.** The consumer is read-only for the demo. A
  real deployment commits *after* folding a tick into state — never before, and
  never via auto-commit, which acknowledges a tick that was merely received.
- **No schema registry.** Ticks are JSON. Avro or Protobuf with a registry is
  what stops a producer change from silently breaking a consumer, and that is a
  real gap.
- **The feature store is NDJSON.** Parquet means pyarrow, and the store is not
  where a heavy dependency earns its place. The format is swappable; what
  matters is that training and serving read the *same stored rows*.
- **No Databricks or Snowflake.** Both have trials that expire. Kafka and Spark
  in Docker are free forever, so that is what this uses.

## Roadmap

1. Commit offsets after state updates, and demonstrate exactly-once across a
   deliberate consumer crash.
2. A changelog topic for the window buffer, so a restart is warm.
3. Avro plus a schema registry, and a test that an incompatible producer change
   is rejected rather than silently mis-parsed.
4. Wire the feature store into `nse-realtime-screener`'s training path, which
   closes the skew it documents at the source.

## Licence

MIT.
