# Validation — 2026-09-28

The complete default pipeline ran with real MLX models on **Apple M4 Pro, 14 CPU cores,
24 GiB unified memory**, in a fresh Python 3.12 environment installed with
`uv sync --locked --python 3.12 --extra all --extra dev`. Models were already downloaded locally.
Machine-readable measurements, model identities, fixture hashes and export hashes are in
[validation-results.json](validation-results.json).

## Software checks

**107 tests passed, 5 opt-in model tests skipped** in the default local suite; Ruff passed. The vendored ECAPA model preserves the
upstream final blank line so its source checksum remains exact. The five real-model tests were also run separately with their
required cached weights/reference dependencies: DNSMOS 3/3; alignment 2/2.
The final combined run, with those checks enabled, passed **112/112 tests** in 42.96 s. Additional
real ECAPA and diarization comparisons are documented in their validation notes.

Regression coverage includes independent stage resume/retry, changed sources/references,
stale human labels, exact-PCM deduplication, sample boundaries, speaker changes/overlap,
wrong references, Turkish characters, repeated/omitted text, forced word cuts, transitive
split grouping and actual FLAC frame/sample-rate validation. GitHub CI uses Linux,
FFmpeg and the locked base/dev dependencies; GPU checks are separate.

## Four-language end-to-end run

16 original FLEURS test recordings (4 each in en/zh/ja/tr), **188.76 seconds** in total,
from dataset revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
These are read-speech fixtures run through the podcast manifest branch, not a natural
podcast cohort or synthetic-generator evaluation. Same-recording/different-recording
reference controls were explicitly labeled; speaker identity was not inferred as truth.

| Check | Observed result |
|---|---|
| ASR | 16 successful actual inferences |
| Native MLX DNSMOS / Nemotron / ECAPA | 16 successful inferences per stage |
| Qwen en/zh/ja / Turkish CTC | 12 / 4 completed inferences |
| Add remaining stages after ASR | ASR result bytes and count unchanged |
| Resume all stages | 0 recomputed, 16 resumed, 0 errors |
| Explicit error injection in an isolated DB copy | 1 quality stage recomputed; 15 records resumed; ASR unchanged |
| Human-review sample | 8 deterministic records; every human label remains null |
| Separate TTS and ASR selection | Each: 7 accept, 9 review, 0 reject, 0 error under a permissive **transport-test-only** policy |
| Actual export | 7 FLAC files/profile, **24 kHz**, fully reopened and frame-checked; portable metadata |
| Export resume | FLAC and metadata bytes unchanged; no group crosses splits |

The 7 accepts are software transport results, **not calibrated training-data acceptance**.
The 9 review records failed structural alignment requirements. For Qwen's 12 fixtures,
9 contain nonpositive-duration units (en 4/4, zh 2/4, ja 3/4). An independent wrapper
check matched upstream positive spans in all 12. There were 7 zero-duration pairs before
upstream timestamp correction and 14 after it; the cause cannot be attributed solely
to the 80 ms timestamp class scale. No confidence or corrected duration was invented.
See [alignment evidence](alignment-validation.md).

## Measured time and memory

The final cached-model run took **32.860 seconds** including stage loading, scoring,
resume/retry checks, selection and both exports. ASR-only phase: **14.359 s**; adding
remaining stages: **11.647 s**. An earlier run of the same fixtures took 76.320 s; the
results do not establish a repeatable throughput estimate for a larger corpus.

- Peak process RSS: **2,249,080,832 bytes** (~2.095 GiB).
- Peak MLX allocator accounting: **2,483,589,010 bytes** (~2.313 GiB).

These are different process-lifetime high-water measurements. They must not be added
or reported as total physical device memory. Downloads and human review are excluded.

A separate **660-second** constructed FLEURS concatenation produced 135 prepare segments
in **12.981 s**. Peak RSS was 580,583,424 bytes; peak MLX accounting 671,088,726 bytes.
A real `SIGKILL` followed by replay produced byte-identical JSONL. A voiced final partial
frame of 73 samples was preserved and marked for review. See
[stream-validation.md](stream-validation.md) for offsets, bounded buffers and replay evidence.

## Conversion parity is separate from quality

- [Turkish Wav2Vec2](alignment-validation.md): four original-PyTorch comparisons;
  maximum logits absolute error **0.0002803803**, all frame argmax values equal.
- [ECAPA](speaker-validation.md): 16 original-SpeechBrain comparisons;
  minimum embedding cosine **0.9999985079**, maximum absolute embedding error **0.0003869832**.
- [DNSMOS](dnsmos-validation.md): 16 speech clips plus silence/noise controls;
  maximum calibrated score difference **6.24e-7**, intermediate tensor difference **6.68e-6**.

Nemotron detected a balanced two-recording mixture but missed the quieter voice in a
roughly 13:1 level-imbalanced mixture. These checks are not a DER or overlap recall study.
There is no original FP32 Qwen numerical-parity claim.

## Reproduce

```bash
uv sync --locked --python 3.12 --extra all --extra dev
source .venv/bin/activate
python scripts/download_fleurs.py --out work/fleurs
# Cache the pinned models first. This runner intentionally requires local weights.
python scripts/validate_pipeline.py \
  --fixtures work/fleurs/manifest.jsonl --out work/validation \
  --hf-cache /path/to/huggingface/hub \
  --dnsmos-model /path/to/sig_bak_ovr.onnx
```

Original-framework parity commands and dependencies are in the stage-specific notes.
No user-distribution human labels, calibrated quality thresholds, natural podcast
cohort evaluation or downstream TTS/ASR training improvement were established.
