# Speaker backends and measured validation

`NemotronDiarizer` runs the native MLX implementation in `mlx-audio==0.5.6`.
Its default [checkpoint](https://huggingface.co/mlx-community/Nemotron-3-Diarization/tree/59ed2dbfc1346dcea9d423c71306a3a2499c568f)
is pinned to `59ed2dbfc1346dcea9d423c71306a3a2499c568f` (OpenMDW-1.1).
It emits eight recording-local speaker activity channels every 160 samples
at 16 kHz. Channels are independently thresholded at the recorded default 0.5;
this is an activity decision threshold, not a calibrated quality guarantee.
Overlap ratio is the duration with at least two active channels divided by the
entire input duration. Eight observed channels trigger a capacity warning;
this does not prove a ninth speaker exists. See the [upstream implementation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/vad/models/nemotron_diarization/nemotron_diarization.py).

`stream(blocks)` preserves one state across bounded PCM blocks and flushes once.
It does not load a complete recording. The offline preset buffers lookahead;
empty intermediate outputs are expected. Each emitted dictionary contains
`start_sample`, `probabilities[frames, 8]`, and `frame_samples=160`.
A new stream resets speaker identity. Resume replays the recording; the adapter
does not serialize the upstream cache. The final sub-frame tail can be shorter
than 160 samples and is excluded from frame coverage. `close()` releases model
references and clears the MLX cache.

`ECAPASpeaker` uses [the pinned MLX conversion](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX/tree/e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77)
of [SpeechBrain ECAPA VoxCeleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286).
Reviewed model/frontend source is vendored under Apache-2.0, with the full license
in `ecapa_license.py`; remote Python files are not executed. The NumPy frontend
uses 80 SpeechBrain mel filters, periodic Hamming/400-point FFT, 160-sample hop,
zero padding and sentence mean normalization. The network runs in MLX evaluation
mode and returns normalized 192-dimensional embeddings. Input is at most 30 s;
the source's validated feature range is 10–3001 frames.

Reference similarity is cosine similarity, not identity probability. Without a
reference it is `null`. A cached embedding is accepted explicitly.
`speaker_consistency` is this project's diagnostic: mean cosine between adjacent,
disjoint 3 s windows, retaining a final remainder only when at least 1 s. For a
2–4 s clip with only one retained window, two equal halves are used. Under 2 s,
consistency is unavailable and status is `review`. `speaker_status=ok` means the
metrics were computed; policy thresholds decide selection. Silence, content and
window length affect consistency. It does not certify one speaker.

## Executed checks — 2026-09-28

Hardware: Apple M4 Pro, 24 GiB. Runtime: MLX 0.32.2, mlx-audio 0.5.6, NumPy 2.5.3.
Original-reference inference used SpeechBrain 1.1.0 / Torch 2.14.0 on CPU in a
separate environment. Heavy model inference was serialized. Model files were
checked against immutable source SHA-256 metadata.

- **ECAPA parity:** 16 real [FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  test clips, four each in English, Mandarin, Japanese and Turkish. Independent
  SpeechBrain Fbank, sentence normalization and the original ECAPA checkpoint
  were compared with the vendored frontend and MLX embeddings. Minimum embedding
  cosine: **0.999998507888092**; maximum embedding absolute error:
  **0.00038698315620422363**; maximum frontend absolute error:
  **0.001049041748046875**. Comparison gates were cosine >0.999 and frontend
  absolute error <0.002. These are implementation parity checks, not verification
  accuracy or language-specific threshold calibration.
- **Reference handling:** identical 6 s audio produced cosine 1.0; an English
  clip compared with a different Japanese recording produced 0.07898771208072872.
  The recordings' speaker identities were not independently annotated.
- **Diarization:** one real clip per language ran successfully. A 52 s constructed
  recording emitted 5,200 × 8 scores with contiguous offsets. Feeding component
  blocks versus 317-sample blocks produced maximum absolute difference **0.0**.
- **Constructed speaker/overlap controls:** sequential English/Japanese recordings
  produced two channels. Their 6 s RMS-balanced mixture produced two channels and
  overlap ratio **0.665**. With clean introductions, the mixed tail's overlap
  fraction was **0.5667**. An unnormalized mixture, whose input RMS levels differed
  by about 13×, missed the quieter voice and reported zero overlap. These are
  controlled functional observations, not labeled DER or overlap precision/recall.
- Eight offline adapter tests passed, covering state continuity, final flushing,
  offsets, multi-label overlap, bounded input and unavailable reference handling.

Validation audio, weights and detailed machine-readable results remain outside
the repository. No throughput, arbitrary-length memory benchmark, speaker
identification accuracy, or production acceptance threshold is claimed.
