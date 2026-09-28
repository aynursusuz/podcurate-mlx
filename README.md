# podcurate-mlx

Local speech-data scoring and filtering on Apple Silicon. Small Python CLI, MLX Whisper,
JSONL inputs, resumable SQLite scores, explicit selection policies.

Works with **podcast segments and synthetic speech** labeled `en`, `zh`, `ja`, or `tr`.
These are supported input languages, **not a claim of validated quality in all four languages**.
This first release is a filtering core, not a complete raw-podcast-to-TTS corpus pipeline.

## Install

Native Apple Silicon Python 3.11+ and FFmpeg are required for MLX inference.

```bash
brew install ffmpeg
git clone https://github.com/aynursusuz/podcurate-mlx.git
cd podcurate-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[mlx]'
```

Whisper inference uses MLX. Upstream `mlx-whisper` also declares PyTorch as a dependency;
installation is not PyTorch-free. Optional VAD uses the community `mlx-audio` MLX port.
Optional DNSMOS uses **ONNX Runtime on CPU**, not MLX.

## 1. Prepare an input manifest

One JSON object per line. Paths resolve relative to the manifest. IDs must be unique.
Use an ISO Whisper language code: `en`, `zh`, `ja`, `tr`.

```json
{"id":"generated-1","audio":"audio/1.wav","kind":"synthetic","language":"tr","reference_text":"Bugün hava çok güzel."}
{"id":"podcast-1","audio":"episodes/1.mp3","kind":"podcast","language":"en","start":12.3,"end":25.1}
```

Synthetic clips require the text given to the generator. Each scoring span must be at most
30 seconds. Audio is read into an unnormalized 16 kHz mono analysis view; original files
are never modified. Preserve generator/prompt/speaker/source information as extra input
fields: it is carried through to the audit output.

Long, untranscribed podcasts can use optional VAD proposals:

```bash
pip install -e '.[vad]'
podcurate-mlx prepare episodes.jsonl --out segments.jsonl \
  --revision 7bc17f22d3c0451bd3a6cd71e759b009271ff49a
```

The episode manifest has `id`, `audio`, and `language`, without `reference_text`.
Prepare runs VAD in bounded 30-second blocks. Segments touching an internal block edge
are marked `boundary_cut` and always go to review during selection. VAD does **not**
identify speakers or remove overlapping speech. This command proposes intervals;
it does not certify TTS training examples.

## 2. Score with MLX

```bash
podcurate-mlx score segments.jsonl --out runs/podcast.sqlite \
  --revision a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb
```

Default model: `mlx-community/whisper-large-v3-turbo`. The commit above pins its weights.
The first invocation downloads them. Use `--offline` once cached. To use another model,
pass both `--model` and its `--revision`. Without a commit, `main` is resolved and its
actual commit is recorded; a changed commit requires a new run database.

Run the same command to resume. Completed records are committed individually. Manifest,
model, code, FFmpeg/dependency version, or source-content/location changes are refused for that run.
Keep source files unchanged while a command is running. One model
process handles one bounded segment at a time. SQLite stores corpus-wide results on disk.

Scores include duration, level, full-scale sample ratio, transcript, decoder diagnostics,
and exact decoded-PCM hash. With reference text, they include **CER for all four languages**
and **space-delimited WER for English/Turkish**. CER/WER are ASR agreement proxies.
Numbers are not expanded, Chinese scripts are not converted, and Japanese readings are
not normalized: inspect these effects when calibrating. Language is supplied, not detected.

Optional acoustic quality:

```bash
pip install -e '.[quality]'
podcurate-mlx score segments.jsonl --out runs/quality.sqlite \
  --revision a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb \
  --dnsmos-model models/sig_bak_ovr.onnx
```

Supply the **non-personalized** model from the
[Microsoft DNSMOS reference directory](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS).
Its SHA-256 is saved. DNSMOS predicts perceptual scores; it does not establish transcript
correctness, speaker identity, or naturalness for every language and synthesis system.

## 3. Select using measured thresholds

No universal quality thresholds are supplied. First score and listen to a stratified
sample from each language, source, and generator. Choose thresholds from that review.
Policy shape (replace the placeholders with finite numbers):

```text
{"languages":{"tr":{"reference_cer":{"max":YOUR_MEASURED_LIMIT},
                     "dnsmos_ovrl":{"min":YOUR_MEASURED_LIMIT}}}}
```

For a syntax-only run, `examples/policy.smoke.json` checks positive duration. It is **not
a quality filter**. Never treat its output as curated training data.

```bash
podcurate-mlx select runs/quality.sqlite --policy policy.json --out runs/selection-v1
```

Outputs:

- `accepted.jsonl`: records passing the specified policy, with source paths and spans.
- `decisions.jsonl`: every record, scores, and `accept` / `reject` / `review` / `error` reasons.
- `summary.json`: counts, the exact policy, and model/runtime identity.
- `_SUCCESS`: selection completed; `_FAILED` marks an incomplete selection.

Missing requested metrics and uncalibrated languages go to review. Silent audio and empty
transcripts are rejected, except that cut-boundary candidates go to review first.
Unavailable or changed source files produce errors at selection. Exact decoded-PCM
duplicates among otherwise accepted records
keep the first manifest entry. Similar text alone never causes removal. Selection requires
a fully completed scoring pass. No source audio is deleted or copied.
The CLI returns a nonzero exit status if scoring/selection contains errors. To retry error
records after correcting the input or environment, start a new database.

## Scope and evidence

Implemented: bounded audio reads, optional MLX VAD, MLX ASR, reference-text agreement,
optional CPU DNSMOS, disk-backed audit/resume, exact PCM deduplication, separate per-language
policies. No throughput or downstream training-quality improvement is claimed.

Still needed for a complete TTS corpus: speaker diarization/overlap detection, independent
alignment and boundary checking, speaker-reference consistency, music/separation assessment,
cross-episode identity and train/test leakage checks. Reference-free podcast ASR can be wrong
even with high confidence. Review rejected samples as well as accepted ones.

[Research and limitations (Türkçe)](docs/research.md) · [Runtime notes](docs/runtime.md)

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
```

Unit tests use model mocks. Model inference verification and its limits are recorded in
[validation](docs/validation.md). Library tests can run without MLX; actual inference needs
a supported MLX runtime. The code is MIT-licensed; upstream models retain their own terms.
