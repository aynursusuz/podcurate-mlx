# Runtime contract

All default neural inference is native MLX: Whisper, Silero, Nemotron, Qwen ForcedAligner,
Turkish Wav2Vec2, ECAPA and DNSMOS. Python 3.12 / native Apple Silicon is the validated
runtime. The reference-comparison environments are separate from this inference path.

- **Audio:** FFmpeg continuously decodes podcasts into bounded 16 kHz mono float32 blocks.
  Analysis leaves original files intact. Export explicitly preserves or selects a training
  sample rate and verifies the actual FLAC frames by reopening the file.
- **Sample-rate provenance:** Probe, decode, streaming and export select the first audio
  stream (`0:a:0`). Scored rows retain the file's `source_sample_rate` and the 16 kHz
  `analysis_sample_rate`. Optional source-rate policy bounds apply before export and are
  independent of DNSMOS. Scoring the primary audio and exporting print conversion warnings
  to stderr once per rate pair per invocation; export persists warnings and conversion counts.
  `preserve` warns on mixed output rates. A higher output rate does not establish wider
  source bandwidth; effective-bandwidth detection is not implemented.
- **State:** Silero consumes 512-sample windows with recurrent state. Nemotron retains
  speaker state across feed calls and finalizes once. Neither restores partial hidden
  state from disk; an incomplete episode is replayed from its beginning. Completed episodes
  retain source/model identities and stable sample-based IDs.
- **Models:** Scoring loads and releases one stage at a time. SQL stores per-stage results;
  adding a new stage does not invalidate successful ASR. Full snapshot commits, conversion
  checksums, adapter code hashes and package versions identify model outputs.
- **Japanese:** MeCab dictionary/Viterbi tokenization runs on CPU without a neural network;
  Qwen alignment runs on MLX. This deliberately differs from upstream nagisa/DyNet
  tokenization. Returned coverage describes the chosen reference units, not confidence.
- **Turkish:** Original checkpoint conversion uses PyTorch once. Acoustic emissions run
  on MLX; a bounded CTC Viterbi trellis uses NumPy. The CTC path score is a log-score,
  not a probability of transcription correctness.
- **DNSMOS:** ONNX parses the original weight container at conversion time. Production
  does not use ONNX Runtime. The numerical comparison does, explicitly.
- **Speaker metrics:** ECAPA BatchNorm uses evaluation mode. Only the reviewed, attributed
  vendored model/frontend code runs; no model-repository Python is executed dynamically.
- **Scale:** Source checksums and split-group state live in SQLite. Process RAM does not
  grow with the manifest or all decoded episode samples. Probability chunks persist on disk.
  Models themselves and bounded activations still consume unified memory.

See [validation](validation.md), [stream validation](stream-validation.md),
[alignment](alignment-validation.md), [speakers](speaker-validation.md), and
[DNSMOS](dnsmos-validation.md) for measured results and their scope. No hardware speed,
quality, or distribution-wide claim follows merely from this architecture.

Sample-rate auditing changes the pinned audio/orchestration code identity. Score databases
created before this change require a new `--out` database; they are never rewritten.
Old scores can still be selected, but a newly required source-rate bound produces review
when that measurement is missing. Export refuses selections lacking source-rate provenance:
rescore and select first. Export metadata uses `flac-groups-v2`; earlier exports need a new
output directory. Preparation now pins the audio adapter too; regenerate earlier prepared
manifests into a new output before reusing diarization. This prevents old automatic stream
selection from mixing with the explicit first-stream rule. Within this version, prepare,
score and export resume remain supported.
