# Third-party sources

The project license does not relicense model weights or datasets.

| Component | Pinned source | Upstream terms |
|---|---|---|
| Whisper large-v3-turbo MLX | `mlx-community/whisper-large-v3-turbo@a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` | [Model card](https://huggingface.co/mlx-community/whisper-large-v3-turbo) |
| Silero MLX | `mlx-community/silero-vad@7bc17f22d3c0451bd3a6cd71e759b009271ff49a` | [Model card](https://huggingface.co/mlx-community/silero-vad) |
| Nemotron | `mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f` | [OpenMDW-1.1 model terms](https://huggingface.co/nvidia/Nemotron-3-Diarization) |
| Qwen ForcedAligner | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit@0e1a68e91d815300c7c9754b2a7639378b23db15` | [Apache-2.0 model card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) |
| Turkish Wav2Vec2 | `m3hrdadfi/wav2vec2-large-xlsr-turkish@8699cf317b8f9a834ddf192608324ae5bbd191f0` | [Apache-2.0 model card](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-turkish) |
| ECAPA MLX and vendored implementation | `aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX@e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77` | [Apache-2.0](https://huggingface.co/aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX); attribution/license retained in vendored modules |
| DNSMOS P.835 | Source commit and model SHA in [conversion validation](dnsmos-validation.md) | [DNS-Challenge root CC-BY-4.0](https://github.com/microsoft/DNS-Challenge/blob/master/LICENSE) |
| Japanese tokenization | `fugashi==1.5.2`, `unidic-lite==1.0.8` | [fugashi MIT / MeCab BSD](https://github.com/polm/fugashi), [UniDic Lite dictionary BSD](https://github.com/polm/unidic-lite) |
| FLEURS validation clips | `google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd` | [CC-BY-4.0 dataset card](https://huggingface.co/datasets/google/fleurs); audio is not committed here |

MLX, mlx-whisper and mlx-audio package releases and transitive dependencies are locked in `uv.lock`. Model hashes and adapter source hashes are saved in each scoring database, alongside resolved checkpoint commits. Conversion code runs locally; no remote Python is executed for ECAPA.
