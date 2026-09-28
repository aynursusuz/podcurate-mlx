# Throughput without changing the filters

The performance changes remove repeated I/O and bookkeeping. Model checkpoints,
precision, decoder settings, alignment algorithms, selection bounds and training
sample rates are unchanged. They do not establish calibrated quality or a
million-record throughput guarantee.

## What is reused

| Work | Behavior |
|---|---|
| Analysis PCM and ASR | The same bounded float32 array feeds signal measurements and ASR; it is decoded once. |
| Explicit source/reference intervals | Known end times avoid a redundant duration probe. Cached reference intervals are reused only while checking the reference content hash. |
| Prepared diarization | Validated activity metadata is reused without decoding the clip again. Source, audio adapter and model identities must still match. |
| Stage results | Each completed stage still commits its result and merged record together. The extra ASR rewrite is removed. |
| Cached scoring | Unchanged records are not rewritten. Later stages materialize only pending records; source and model identity checks still run. |
| Export resume | Existing grouping work is reused. Initially verified FLAC files need matching content hashes and stored metadata to reuse verification. |

There is no corpus-sized PCM cache: a stage retains at most a bounded clip's audio
instead of materializing millions of waveforms on disk or in RAM. Models still load
one at a time. Complete resumes still validate model identities and source hashes;
large collections of unique source files can make those checks expensive.

SQLite synchronization and the per-completed-stage commit boundary are preserved.
An interruption can require recomputing unfinished work, but should not repeat
successfully committed inference. Export still reopens and fully decodes each new
FLAC, and detects changed source/completed output files on resume.

## Reproduce the measurements

Measured on the M4 Pro / 24 GB machine, against commit `948f5da`.
[Recorded results](performance-results.json) include source fingerprints and raw timings.

| CPU I/O test: 200 repeated 1.25 s records, stub ASR + quality | Before | After |
|---|---:|---:|
| Initial scoring, median of two runs | 25.533 s | 10.721 s |
| Cached resume, median of two runs | 5.015 s | 0.047 s |
| Initial FFmpeg decode calls | 600 | 400 |
| Initial ffprobe calls | 401 | 1 |
| Cached-resume record writes | 200 | 0 |

All four runs produced the same canonical output digest. These improvements measure
orchestration/I/O with stub models, not MLX inference speed. The unusually short
resume uses one shared source file; unique-source hashing still costs time.

Separately, two runs per version executed real models on **16 FLEURS clips across
en/zh/ja/tr**. All **80 stage-metric results** matched exactly across all four runs.
Both profiles retained 7 accepts / 9 reviews under the runner's explicitly
permissive transport-test policy. All **14 FLAC files and their export metadata**
were byte-identical across versions. These are software-equivalence checks, not
human quality calibration.

The second, sequential real-model comparison took **38.267 s before / 36.633 s
after** for the entire validation workflow, including resume/retry/export checks.
The first comparison took 43.952 s / 39.129 s; its after-run overlapped ordinary
CPU unit tests, so it is retained as evidence rather than used as a clean speed
estimate. Models were already downloaded. This small cohort does not establish
million-record throughput.

The CPU benchmark below uses a repeated generated WAV and **stub neural stages**.
It measures process starts, database writes and orchestration, not speech-model
throughput. It records timings, peak process RSS, source hashes and canonical output
digests. Use a fresh output directory for every run.

```bash
python scripts/benchmark_io.py --records 200 --out benchmarks/io-run-1
```

For a code comparison, invoke the same script with `PYTHONPATH` pointing at each
checkout's `src`. Alternate versions to reduce cache/order bias. Compare canonical
output digests as well as elapsed time; the canonical digest excludes stage timings
and fixture-directory prefixes.

Real neural checks use the four-language FLEURS runner described in
[validation.md](validation.md). Compare stage metrics, selection decisions and
exported FLAC hashes across versions. Timing is separate from numerical/output
equivalence and from human quality calibration.

A separate scale check preseeds deterministic stub results for a large manifest,
then invokes the real cached scoring path in a fresh child process:

| One-million-row cached-resume check: ASR + quality stubs | Measured |
|---|---:|
| Records resumed / errors | 1,000,000 / 0 |
| Traversal wall time | 69.772 s |
| Fresh-process peak RSS | 45.38 MiB |
| Source checks / unique source hashes | 1,000,000 / 1 |
| Model calls / FFmpeg decodes / record rewrites | 0 / 0 / 0 |
| Fixture preparation / post-run digest verification | 31.312 s / 6.003 s |
| Fixture disk usage | 2.930 GB |

The exact stored-record digest was unchanged. This was **one run on preseeded
stub results referencing one generated tone**; it proves the tested cache traversal
and integrity behavior, not the cost of inference or hashing one million distinct
audio files. The RSS value excludes the seeding process and real MLX models.
Filesystem pages may remain cached after seeding; this is not a cold-disk test.
Process RSS also excludes the operating system's file cache.

```bash
python scripts/benchmark_resume.py --records 1000000 \
  --out benchmarks/cached-million --disk-budget-gb 6
```

This creates a multi-gigabyte SQLite fixture. All IDs share one generated tone;
the test measures manifest traversal, cached-result handling and source checks,
not a million unique files or speech-model calls. Seeding time, measured traversal,
verification time and fresh-process peak RSS are reported separately. The runner
checks that cached record bytes remain identical and no inference/decode or record
rewrite occurs. Disk guards stop it if the configured budget is approached.

## Large runs

Start with a representative pilot containing the actual languages, durations,
podcast/synthetic mix and optional speaker references. Measure each stage and
storage growth before estimating the full corpus. A repeated-file CPU benchmark
does not reproduce reading millions of distinct files or long compressed episodes.

Use one scoring process/writer per database on the 24 GB machine; increasing the
number of MLX processes also duplicates model memory. RAM buffers are bounded, but
the manifest, SQLite audit records and exported audio still grow with the corpus.
Keep the SQLite files and their source inputs available for resume.

Keep selection and export global if manually splitting scoring work: independently
exporting shards does not provide cross-shard duplicate or speaker/source split
protection. The current CLI does not merge independently scored databases.

After an orchestration-code update, start a new score database as required by the
pinned run identity. Successful stages are reused within the same code/model run.
Prepared manifests remain usable when their audio adapter identity still matches.
This export update also needs a new output directory because cached verification
now pins the verifier and FFmpeg/ffprobe versions.
