# podcurate-mlx

Local podcast and synthetic-speech curation on Apple Silicon: **prepare → score → calibrate → select → export**. English, Mandarin Chinese, Japanese and Turkish. Default neural inference runs on **MLX**, one model at a time; FFmpeg, SQLite and numerical processing run on CPU.

## Filtering at a glance

🟩 **Required** for the profile · 🟦 **Policy-controlled**: use only when explicitly bounded.
These colors describe selection rules, not measured quality or model confidence.
All default scoring stages run for both profiles; their acceptance requirements differ.

| Check | Model or measurement | TTS profile | ASR profile |
|---|---|---|---|
| **Audio integrity** | FFmpeg; duration, silence and PCM hash | 🟩 Valid audio; silence excluded | 🟩 Valid audio; silence excluded |
| **Source sample rate** | FFprobe reads the original audio stream; conversion warnings remain separate from quality scores | 🟦 Set `source_sample_rate` bounds for the training task | 🟦 Same source-rate bounds; export upsampling cannot change this metric |
| **Transcript agreement** | [MLX Whisper](src/podcurate_mlx/backends.py); CER for all four languages, WER for en/tr | 🟩 Reference-agreement bounds, or decoder bounds without a reference | 🟩 Same text checks |
| **Text–audio alignment** | [MLX Qwen](src/podcurate_mlx/alignment.py) for en/zh/ja; MLX Wav2Vec2 CTC for tr | 🟩 Valid intervals and reference-unit coverage | 🟩 Same alignment checks |
| **Speakers and overlap** | [MLX Nemotron](src/podcurate_mlx/speakers.py); speaker activity and simultaneous speech | 🟩 One detected speaker, overlap bound; capacity saturation requires review | 🟦 Speaker changes and overlap alone do not reject audio |
| **Speaker consistency / reference** | [MLX ECAPA](src/podcurate_mlx/speakers.py); within-recording consistency and optional reference cosine | 🟩 Consistency bound; similarity bound when a reference exists | 🟦 Apply bounds if required by the policy |
| **Acoustic quality** | [Native MLX DNSMOS P.835](src/podcurate_mlx/quality.py); SIG, BAK and OVRL | 🟩 Explicit OVRL bound | 🟦 Noise score alone is not an automatic rejection rule |
| **Segment boundaries** | Stateful VAD preparation; forced-cut and unscored-tail flags | 🟩 Flagged boundaries cannot be accepted | 🟩 Same boundary protection |
| **Duplicates and data splits** | Exact analysis-PCM deduplication; source/speaker/reference groups | 🟩 Deduplicate accepted records; keep export groups together | 🟩 Same export protections |

Bounds are chosen separately for each **profile × language × podcast/synthetic source**.
A reference-free speaker score does not verify target identity. Nemotron has eight
source-local speaker channels; unknown speakers have no speaker-disjoint guarantee.

| Decision | Meaning | Exported? |
|---|---|---|
| 🟢 **`accept`** | Required checks and the supplied policy pass; the record is not an accepted PCM duplicate | Yes |
| 🟡 **`review`** | For example: missing required measurements/bounds, invalid alignment, or a flagged cut | No |
| 🔴 **`reject`** | For example: an explicit bound fails, silence, empty transcript, or an accepted PCM duplicate | No |
| 🟣 **`error`** | For example: decoding/inference failure or a changed source/reference | No |

These are decision examples, not a priority order: multiple problems can coexist.
The exact rules live in [metrics.py](src/podcurate_mlx/metrics.py); every selection writes
`decisions.jsonl` with its reasons. Missing required measurements never produce acceptance.

## Install

Native Apple Silicon, Python 3.12 and FFmpeg. Model downloads happen on first use. Source audio is never overwritten.

```bash
brew install ffmpeg uv
git clone https://github.com/aynursusuz/podcurate-mlx.git
cd podcurate-mlx
uv sync --locked --python 3.12 --extra all --extra dev
source .venv/bin/activate
podcurate-mlx --help
```

`uv.lock` fixes Python dependencies; model commits and conversion hashes are recorded in each run. PyTorch is installed for the initial Turkish checkpoint conversion and upstream packaging; default acoustic inference uses MLX. Original-framework comparisons are separate development checks.

## Input

One JSON object per line; paths resolve relative to the manifest. Each scored interval is at most 30 seconds. Supported language codes: `en`, `zh`, `ja`, `tr`.

```json
{"id":"generated-1","audio":"audio/1.wav","language":"tr","kind":"synthetic","reference_text":"Bugün hava çok güzel.","source_id":"generation-batch-1","speaker_id":"known-voice-1","reference_audio":"references/voice-1.wav"}
{"id":"episode-1","audio":"episodes/1.mp3","language":"en","kind":"podcast","source_id":"episode-1"}
```

Synthetic speech requires the text supplied to the generator. `speaker_id` and `reference_audio` are optional. A reference longer than 30 seconds uses its first 30 seconds; `reference_start` / `reference_end` can select another bounded interval. References should contain the intended speaker.

Prepared intervals have integer `start_sample`, `end_sample` and `timebase_hz`; seconds-based `start` / `end` remain supported. Known speaker IDs must mean the same person across files. Episode-local diarization labels are never treated as global speaker IDs.

## Commands

**Choose your starting point:** long, untranscribed podcasts start at `prepare`;
already bounded clips (up to 30 seconds), including synthetic speech, start at `score`.
Both then use `calibrate → select → export`. Selection needs your own `policy.json`;
scoring alone does not require quality thresholds.

Prepare long, untranscribed podcast episodes (omit reference text):

```bash
podcurate-mlx prepare episodes.jsonl --out segments.jsonl
```

FFmpeg reads bounded blocks. Silero recurrent state and Nemotron speaker state persist across blocks. Forced 30-second cuts receive `boundary_cut` and require review. A sidecar SQLite database retains completed episodes; rerun the same command to replay an unfinished episode from its start and publish deterministic intervals. Preserve this sidecar for resume.

Score prepared podcast segments or already short synthetic clips:

```bash
podcurate-mlx score segments.jsonl --out runs/scores.sqlite
# Same command resumes; only failed stages are retried with this flag:
podcurate-mlx score segments.jsonl --out runs/scores.sqlite --retry-errors
```

The default stages are MLX Whisper ASR, native MLX DNSMOS, Nemotron diarization/overlap, ECAPA speaker metrics, Qwen alignment for en/zh/ja and MLX Turkish Wav2Vec2 CTC alignment. They run sequentially. Successful results commit per stage and record. `--stages asr,quality` can start a partial run; adding other stages later reuses successful ASR. Partial runs cannot pass a profile requiring missing stages. Source, manifest or existing stage identity changes require a fresh database. Use one writer per database. Version 0.1 databases remain intact; start a fresh database for 0.2.

Create a repeatable human-review sample stratified by language and podcast/synthetic source:

```bash
podcurate-mlx calibrate runs/scores.sqlite --out review --per-group 20 --seed 42
```

Listen to the source intervals in `review/review.jsonl`, set `human_label` to `accept` or `reject`, and retain IDs, hashes and input metadata. Choose separate bounds for each language/source/profile. No universal quality cutoffs are supplied. Policy shape (placeholders must be replaced with finite numbers):

```text
{"profiles":{"tts":{"languages":{"tr":{"synthetic":{
  "reference_cer":{"max":YOUR_LIMIT},
  "overlap_ratio":{"max":YOUR_LIMIT},
  "speaker_consistency":{"min":YOUR_LIMIT},
  "speaker_similarity":{"min":YOUR_LIMIT},
  "dnsmos_ovrl":{"min":YOUR_LIMIT}
}}}},"asr":{"languages":{"tr":{"synthetic":{
  "reference_cer":{"max":YOUR_LIMIT}
}}}}}}
```

For podcasts without reference text, include `avg_logprob` and `compression_ratio` bounds. ASR agreement and decoder scores remain proxies; they cannot establish that an untranscribed podcast is correct. CER works across the four languages; space-delimited WER is provided only for en/tr. Digits/readings/scripts are not silently normalized into guessed words.

Evaluate chosen policies against human labels, then select independently:

```bash
podcurate-mlx calibrate runs/scores.sqlite --out evaluated \
  --labels review/review.jsonl --policy policy.json --profile tts
podcurate-mlx select runs/scores.sqlite --policy policy.json --profile tts --out selected-tts
podcurate-mlx select runs/scores.sqlite --policy policy.json --profile asr --out selected-asr
```

TTS requires one detected speaker, structurally valid alignment, intact boundaries, acoustic and speaker measurements, and explicit bounds. Reference similarity applies only when a reference exists; without one, consistency does not prove target identity. ASR checks text/alignment; noise and speaker changes alone are not rejection rules. Missing required measurements produce review/error, never acceptance. Exact decoded-PCM duplicates are removed only among otherwise accepted records.

Export actual, reopened-and-verified FLAC files and portable JSONL metadata:

```bash
podcurate-mlx export selected-tts --out corpus-tts --sample-rate 24000 --seed 42
podcurate-mlx export selected-asr --out corpus-asr --sample-rate preserve --seed 42
```

The training sample rate is explicit: a 16 kHz analysis copy never silently becomes training audio. Default split fractions are 90%/5%/5%, configurable with `--train` / `--validation`; small grouped corpora need not approximate these fractions. Shared sources, known speakers, reference sources and exact PCM copies are transitively grouped before splitting. **Unknown speakers have no speaker-disjoint guarantee.** Export can resume, and refuses changed source or completed output files.

### Sample rates and quality

Scoring records `source_sample_rate` from the file and `analysis_sample_rate` (16000 Hz).
Scoring the primary audio and exporting print conversion warnings to stderr, once per rate pair
per invocation. JSON results on stdout remain machine-readable. All audio operations use
the first audio stream (`0:a:0`), so probing, scoring and export describe the same signal.

| Conversion | Warning / consequence |
|---|---|
| 🟨 **16 → 48 kHz** | Upsampling does not restore missing frequency detail; the higher output rate is not evidence of higher quality. |
| 🟨 **48 → 24 kHz** | Downsampling limits bandwidth to below 12 kHz. Choose the rate required by your training model. |
| 🟦 **48 → 16 kHz analysis** | Models assess the analysis view, with bandwidth below 8 kHz; original training audio remains available. |
| 🟨 **Mixed rates with `preserve`** | Export warns that the corpus contains multiple rates; choose an explicit rate if your trainer requires uniform input. |

To exclude sources below a task-specific rate, add `"source_sample_rate":{"min":24000}`
to the relevant profile/language/source bounds. **24000 is an example requirement, not a
universal quality threshold.** Missing values require review; out-of-bound values reject.
The bound applies before export, so converting a 16 kHz source to 48 kHz cannot satisfy it.

Export metadata keeps `source_sample_rate`, `analysis_sample_rate`, output `sample_rate`
and `sample_rate_warnings`. `run.json` contains conversion counts and mixed-output-rate
warnings, including on resume. `source_timebase_hz` remains the coordinate system for
segment boundaries, which can differ from the file's sample rate.

Sample rate alone is not a perceptual quality score. These checks do not detect audio
that was already upsampled before ingestion or measure its effective spectral bandwidth.
DNSMOS scores are not penalized by an invented conversion formula. Conversion uses
[FFmpeg's resampler](https://ffmpeg.org/ffmpeg-resampler.html).

**Updating an existing run:** regenerate prepared manifests and scores with new `--out`
paths, then select and export into a new directory. Earlier results lack the first-stream
provenance needed by this version; see [compatibility details](docs/runtime.md).

## Code and documentation

The CLI exposes five commands. To follow an input through the code:

- [cli.py](src/podcurate_mlx/cli.py) parses commands; [pipeline.py](src/podcurate_mlx/pipeline.py) validates manifests and writes selection decisions.
- [preparing.py](src/podcurate_mlx/preparing.py) handles continuous episodes; [audio.py](src/podcurate_mlx/audio.py) bounds audio reads.
- [stages.py](src/podcurate_mlx/stages.py) loads models sequentially and stores resumable stage results in SQLite.
- [metrics.py](src/podcurate_mlx/metrics.py) defines text metrics and profile rules; [curation.py](src/podcurate_mlx/curation.py) samples human review records and exports grouped FLAC datasets.
- Model adapters are linked in the filtering table above. The attributed `ecapa_*` modules retain upstream model/frontend code and license text.
- [tests/](tests/) contains regression and opt-in model checks; [scripts/](scripts/) contains the FLEURS downloader and reproducible end-to-end runner.

## Evidence and limits

- [Runtime, preprocessing and resume behavior](docs/runtime.md)
- [Research and source mapping](docs/research.md)
- [DNSMOS numerical equivalence](docs/dnsmos-validation.md)
- [ECAPA and diarization validation](docs/speaker-validation.md)
- [Alignment validation](docs/alignment-validation.md)
- [Long-recording memory and interruption checks](docs/stream-validation.md)
- [End-to-end checks and reproduction](docs/validation.md) · [Recorded results and model identities](docs/validation-results.json)

Numerical parity is separate from quality calibration. FLEURS checks exercise real speech in the four languages; they do not measure performance on the user's podcasts or speech generators. Nemotron has eight recording-local speaker channels and can miss low-level overlapping speech. Qwen does not provide an alignment confidence score; none is invented. Turkish does not use Qwen's unsupported language path.

```bash
pytest -q
ruff check src tests scripts
```

Ordinary CI uses deterministic fixtures and model mocks, without GPU downloads. Opt-in real-model checks and measured hardware results are documented separately. Project code is MIT; models, vendored code and datasets retain their [upstream terms](docs/third-party.md).
