# Long recording and interruption validation

Executed on 2026-09-28 on an Apple M4 Pro with 24 GiB unified memory, using
MLX 0.32.2 and mlx-audio 0.5.6. This validates the **prepare** stage: FFmpeg
streaming decode, native MLX Silero VAD, native MLX Nemotron diarization,
SQLite persistence and JSONL publication. ASR, alignment, speaker embeddings
and DNSMOS are not included in these timings.

The input was **660 seconds (11 minutes)** of repeated public FLEURS clips in
English, Mandarin, Japanese and Turkish, converted to mono 16 kHz PCM16.
It is a constructed duration/buffering fixture, **not a natural podcast or a
quality evaluation**. The source dataset revision was
[`70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
(CC-BY-4.0). The generated waveform's SHA-256 was
`54f0ef8d521f79b12e8fe2ed61551ee88d58756d96eb595597df170887fc08bb`.

Models were loaded locally from the immutable
[Silero revision](https://huggingface.co/mlx-community/silero-vad/tree/7bc17f22d3c0451bd3a6cd71e759b009271ff49a)
and [Nemotron revision](https://huggingface.co/mlx-community/Nemotron-3-Diarization/tree/59ed2dbfc1346dcea9d423c71306a3a2499c568f).
Heavy inference was serialized with a process lock. Each row below is a fresh
Python process. Timings exclude waiting for that lock and include model loading
and prepare work. They are single observations, not sustained throughput claims.

| Run | Input duration | Wall time | Process peak RSS | MLX tracked peak |
|---|---:|---:|---:|---:|
| Short comparison | 120 s | 6.217 s | 546.25 MiB | 636.83 MiB |
| Uninterrupted | 660 s | 12.981 s | 553.69 MiB | 640.00 MiB |
| Replay after termination | 660 s | 16.162 s | 549.67 MiB | 640.09 MiB |
| Already complete | 660 s | 2.671 s | 524.02 MiB | 189.63 MiB |

RSS is macOS `resource.getrusage(...).ru_maxrss`; the MLX value is
`mx.get_peak_memory()` after resetting its counter. These are different memory
accounting measures and must not be added. RSS does not include the FFmpeg child
process. Neither measure is total machine memory or a guarantee for arbitrary
inputs. Transparent instrumentation observed at most **320,000 input samples
(20 s)** per decoder block. Persistent neural state arrays peaked at **1,280 B**
for Silero and **2,267,652 B** for Nemotron; this excludes weights, temporary
activations and Python object overhead. Long-file preparation yielded 135
segments and 25 diarization output chunks. Probabilities are retained on disk.

## Real interruption and replay

The process received **SIGKILL** after its third diarization output chunk had
been committed. SQLite then contained three probability chunks, an incomplete
episode and zero published segments. No final JSONL existed.

A fresh process replayed the incomplete episode. Its 135-segment output was
**byte-identical** to uninterrupted preparation: 272,738 bytes, SHA-256
`c7026599bd7efaef5d936ffcdc70a6ff9ba9361e5c36f03246a5edda617453f4`.
A further completed-run invocation decoded zero blocks and preserved those
bytes. This establishes replay correctness for this fixture; unfinished episodes
are recomputed rather than resumed from serialized neural state.

## Boundary and capacity checks

The validation exposed and fixed a frame-clipping bug: when a VAD interval began
between diarization frame boundaries, combining activity into shifted bins could
turn sequential speakers into false overlap. Aggregation now intersects original
frame intervals exactly. Regression tests cover sequential speakers, partially
clipped true overlap, eight sequential channels, and an episode using all eight
channels whose individual VAD spans contain only one speaker. Capacity is recorded
at source level; it is a warning, not proof of an additional speaker.

A separate real voiced fixture contained **32,073 samples**, leaving a 73-sample
partial final frame. The original end sample remained 32,073. Its unscored tail
produced `diarization_status: review`, reason `unscored_diarization_tail`,
`unscored_tail_samples: 73`, null speaker/overlap scores and `boundary_cut: true`.
No speech activity was invented for that tail, and no source audio was discarded.

Four offline regression tests pass in `tests/test_stream_runtime.py`. Audio,
weights, per-run telemetry, SQLite state and detailed results remain outside
this repository. Speaker-model parity and overlap limitations are documented in
[speaker-validation.md](speaker-validation.md).
