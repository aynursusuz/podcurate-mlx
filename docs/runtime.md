# Runtime evidence and limits

Verified on 2026-09-28 using upstream source, package metadata, and a real local
smoke test on the target M4 Pro with 24 GiB memory. A 5.51-second English sample
generated with macOS `say` passed MLX Whisper tiny and default large-v3-turbo
score/select/resume, MLX Silero VAD preparation, and DNSMOS CPU inference. Two duplicate records yielded one
acceptance and one exact-duplicate rejection. This verifies integration, not
multilingual quality or throughput; the tiny transcript contained errors, while
large-v3-turbo matched this sample's reference transcript.

## MLX Whisper

[MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) implements
the neural network with MLX. Its tokenizer lists English, Chinese, Japanese, and
Turkish. This does not establish equal accuracy across these languages.
The PyPI `mlx-whisper` 0.4.3 metadata includes **torch** as a dependency; an MLX
inference backend does not imply a torch-free installation.

Upstream `transcribe()` accepts a local model path but does not expose a model
revision argument. This adapter resolves a Hugging Face snapshot once, records
its 40-character commit SHA, and transcribes using the local snapshot path.

Upstream file decoding collects the complete ffmpeg output, and preprocessing
constructs the spectrogram for the complete input. Internal 30-second inference
windows are not streaming file I/O. This adapter accepts only mono float PCM
already sampled at 16 kHz, with a maximum duration of 30 seconds.

Language is explicitly supplied. Accordingly, `detected_language` and
`language_probability` are null. Decoder statistics are duration-weighted means
over finite, positive-duration returned segments. They are not MOS scores or
calibrated hallucination probabilities; missing statistics remain null.

## Optional VAD

[MLX Audio](https://github.com/Blaizzy/mlx-audio) is Prince Canuma's community
project, not an Apple-maintained package. The inspected source revision is
`4ab7e6f7dedd69a136cfaa318c5dc8aed5119446`. Its Silero port provides
`get_speech_timestamps()` and a streaming `feed()` API requiring 512 samples at
16 kHz. Preparation decodes bounded windows and flags segments touching internal
window boundaries. Cross-window VAD state and forced alignment are not implemented.

Upstream also contains Sortformer diarization and Qwen3 forced alignment ports.
These are not implemented here. The inspected forced-aligner language list does
not establish Turkish support. Official DNSMOS runs through ONNX Runtime;
official NISQA and UTMOS22 implementations use PyTorch. None is silently treated
as an MLX backend.

## Memory

MLX uses lazy evaluation and unified CPU/GPU memory. Dataset size is not a batch
size, and physical RAM is not reserved GPU memory. Keep inference serial and
measure actual peak memory before changing concurrency. No batch size or speed
claim has been established for this machine.
