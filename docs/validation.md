# Validation — 2026-09-28

**52 tests passed; Ruff passed.** Tests cover Turkish casing, Chinese/Japanese CER,
policy validation, missing/non-finite scores, real FFmpeg reads, SQLite per-record commits,
resume, changed/moved sources, duplicate IDs, incomplete runs, selection errors, exact PCM
deduplication, VAD block-boundary review, and model-adapter contracts. Neural outputs in
unit tests are mocked. GitHub CI runs these tests on Linux with FFmpeg, without model weights.

## Real inference on the target machine

Apple M4 Pro, 24 GiB, macOS arm64; Python 3.12. Packages used:
`mlx 0.32.2`, `mlx-whisper 0.4.3`, `mlx-audio 0.5.6`, `onnxruntime 1.30.0`.

Test input: a 5.510204-second English sample generated with the default macOS `say` voice:

> This is a short English speech sample. We are testing a local audio filtering pipeline.

The same file appeared twice under different record IDs. This is a deliberately small
integration fixture, not a benchmark or quality-calibration set.

| Component | Actual check | Result |
|---|---|---|
| MLX Whisper tiny | Transcribe, score 2 records, resume 2, select with a smoke policy | Passed; transcript contained errors |
| MLX Whisper large-v3-turbo (default) | Transcribe, score 2 records, resume 2, select | Passed; transcript matched this reference |
| MLX Silero VAD | Prepare a manifest from the source audio | Passed; 2 intervals |
| DNSMOS P.835 | Execute the real ONNX model with CPUExecutionProvider | Passed; finite SIG/BAK/OVRL |
| Final CLI | Default Whisper + DNSMOS score; resume; select; VAD prepare | Passed |

Selection retained one otherwise-eligible record and rejected its exact PCM duplicate.
The smoke policy checks syntax/flow, not perceptual quality. No human MOS was collected.

Pinned model snapshots:

- `mlx-community/whisper-tiny`: `78c52ab98ca87f570bc57ad852e15ef7060f9f76`.
- `mlx-community/whisper-large-v3-turbo`: `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`.
- `mlx-community/silero-vad`: `7bc17f22d3c0451bd3a6cd71e759b009271ff49a`.
- DNSMOS model: [Microsoft source snapshot](https://raw.githubusercontent.com/microsoft/DNS-Challenge/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS/sig_bak_ovr.onnx),
  SHA-256 `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`.

## Not established

No Chinese, Japanese, or Turkish speech inference evaluation; no real podcast cohort;
no human quality labels; no calibrated acceptance thresholds; no downstream TTS/ASR
training experiment; no large-corpus throughput, peak-memory, or long-file boundary
quality benchmark. Supported input languages and passing software tests must not be
reported as these missing evaluations.
