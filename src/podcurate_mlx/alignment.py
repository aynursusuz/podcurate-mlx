"""Independent forced alignment, with acoustic model inference on MLX.

Qwen predicts timestamps; Turkish Wav2Vec2 emits framewise CTC logits. Their
scores are deliberately not interchangeable. No confidence is invented for Qwen.
Japanese text segmentation uses non-neural MeCab/UniDic Viterbi on CPU instead
of upstream nagisa. The Turkish encoder is reused from mlx-audio (MIT);
both default model checkpoints declare Apache-2.0. See docs/alignment-validation.md.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

from .backends import _COMMIT, _audio, _optional_module, _package_version

QWEN_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
QWEN_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
TURKISH_MODEL = "m3hrdadfi/wav2vec2-large-xlsr-turkish"
TURKISH_REVISION = "8699cf317b8f9a834ddf192608324ae5bbd191f0"
_QWEN_LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
_CONVERSION_VERSION = "wav2vec2-fp32-v1"


def _snapshot(model_id, revision, local_files_only, cache_dir, patterns=None):
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise ValueError("Alignment model revision must be a full immutable commit SHA")
    hub = _optional_module("huggingface_hub", "alignment")
    path = Path(hub.snapshot_download(
        repo_id=model_id, revision=revision, local_files_only=local_files_only,
        cache_dir=cache_dir, allow_patterns=patterns,
    )).resolve()
    if path.name.lower() != revision.lower():
        raise RuntimeError("Alignment checkpoint did not resolve to the requested commit")
    return path


def _result(reasons=(), alignment=None, coverage=None, score=None, **extra):
    reasons = list(dict.fromkeys(reasons))
    return {
        "alignment": alignment or [],
        "alignment_status": "review" if reasons else "ok",
        "alignment_reasons": reasons,
        "alignment_coverage": coverage,
        "alignment_score": score,
        **extra,
    }


def _reference(text):
    if not isinstance(text, str):
        raise TypeError("reference_text must be a string")
    return unicodedata.normalize("NFC", text).strip()


def _release(obj):
    obj._model = None
    gc.collect()
    obj._mx.clear_cache()


class MeCabJapaneseUnits:
    """Non-neural morphological units; not upstream nagisa-equivalent tokens.

    fugashi is an MIT wrapper of BSD MeCab. unidic-lite packages UniDic 2.1.2
    under BSD. Explicit dictionary/rule paths prevent local MeCab user settings
    from silently changing the selected dictionary.
    """

    def __init__(self, clean_token):
        fugashi = _optional_module("fugashi", "alignment")
        unidic = _optional_module("unidic_lite", "alignment")
        directory = Path(unidic.DICDIR)
        if '"' in str(directory):
            raise ValueError("UniDic dictionary path contains an unsupported quote")
        self._tagger = fugashi.Tagger(f'-r /dev/null -d "{directory}"')
        self._clean = clean_token
        digest = hashlib.sha256()
        for name in ("sys.dic", "unk.dic", "matrix.bin", "char.bin", "dicrc"):
            digest.update(name.encode())
            with (directory / name).open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        self.identity = {
            "backend": "mecab_unidic_lite_viterbi", "neural": False,
            "fugashi_version": _package_version("fugashi"),
            "unidic_lite_version": _package_version("unidic-lite"),
            "dictionary_sha256": digest.hexdigest(),
            "coverage_unit": "mecab_morphological_units",
        }

    def __call__(self, text):
        return [cleaned for word in self._tagger(text)
                if (cleaned := self._clean(word.surface))]


class QwenAligner:
    """MLX Qwen timestamp prediction for en/zh/ja; Turkish is rejected."""

    def __init__(self, model_id=QWEN_MODEL, revision=QWEN_REVISION,
                 local_files_only=False, *, cache_dir=None):
        self._mx = _optional_module("mlx.core", "alignment")
        stt = _optional_module("mlx_audio.stt", "alignment")
        path = _snapshot(model_id, revision, local_files_only, cache_dir)
        self._model = stt.load(str(path), strict=True)
        self._model.eval()
        self._japanese = MeCabJapaneseUnits(self._model.aligner_processor.clean_token)
        # Per-instance processor injection: no upstream files/global classes are patched.
        self._model.aligner_processor.tokenize_japanese = self._japanese
        supported = set(self._model.get_supported_languages() or [])
        if not {v.lower() for v in _QWEN_LANGUAGES.values()} <= supported:
            self.close()
            raise ValueError("Qwen checkpoint does not declare the expected en/zh/ja support")
        self.identity = {
            "backend": "mlx-qwen3-forced-aligner", "model_id": model_id,
            "revision": revision, "package_version": _package_version("mlx-audio"),
            "mlx_version": _package_version("mlx"), "languages": ["en", "zh", "ja"],
            "sample_rate": 16000, "max_audio_seconds": 30,
            "score_kind": None, "coverage_kind": "returned_reference_units",
            "timestamp_quantum_seconds": self._model.config.timestamp_segment_time / 1000,
            "timestamp_postprocessing": "mlx_audio_fix_timestamp_then_span_validation",
            "japanese_text_segmentation": self._japanese.identity,
        }

    def __call__(self, audio_16k, reference_text, language):
        if language not in _QWEN_LANGUAGES:
            raise ValueError("Qwen forced alignment supports en/zh/ja only; tr is unsupported")
        if self._model is None:
            raise RuntimeError("Aligner is closed")
        audio, text = _audio(audio_16k), _reference(reference_text)
        if not text:
            return _result(["empty_reference"], coverage=0.0)
        if not np.any(audio):
            return _result(["silent_audio"], coverage=0.0)
        # Bound adversarially long references before tokenization/model allocation.
        if len(text) > 4096:
            return _result(["reference_too_long"])
        language_name = _QWEN_LANGUAGES[language]
        expected, _ = self._model.aligner_processor.encode_timestamp(text, language_name)
        if not expected:
            return _result(["empty_normalized_reference"], coverage=0.0)
        if len(expected) > 512:
            return _result(["reference_too_long"])
        result = self._model.generate(audio=audio, text=text, language=language_name)
        items, reasons, invalid_units = [], [], []
        duration, previous_end = len(audio) / 16000, 0.0
        for item in result.items:
            start, end = float(item.start_time), float(item.end_time)
            if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                reasons.append("invalid_alignment_timestamp")
                invalid_units.append({
                    "text": item.text,
                    "start": start if np.isfinite(start) else None,
                    "end": end if np.isfinite(end) else None,
                    "reason": "zero_duration" if start == end else "invalid_span",
                })
                continue
            if start < previous_end:
                reasons.append("nonmonotonic_alignment")
            if end > duration:
                reasons.append("alignment_outside_audio")
            items.append({"text": item.text, "start": start, "end": end})
            previous_end = end
        if [i["text"] for i in items] != expected:
            reasons.append("reference_units_not_fully_aligned")
        coverage = min(len(items) / len(expected), 1.0)
        return _result(reasons, items, coverage, None,
                       alignment_invalid_units=invalid_units,
                       alignment_units_total=len(expected))

    def close(self):
        self._japanese = None
        _release(self)


def normalize_turkish_reference(text: str) -> str:
    """Keep lexical letters, preserve Turkish I/İ, remove punctuation only.

    Numbers/symbols remain visible to the OOV check: no guessed pronunciation.
    """
    text = _reference(text).replace("İ", "i").replace("I", "ı").lower()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def ctc_viterbi(log_probs: np.ndarray, tokens: list[int], blank: int = 0):
    """Full CTC state path with repeated-token/blank transitions.

    Returns target-token-index per frame (-1 for blank), or None if impossible.
    CPU DP is bounded by <=30 s emissions; neural inference remains on MLX.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2 or not np.isfinite(log_probs).all():
        raise ValueError("CTC log probabilities must be a finite [frames, vocabulary] array")
    frames, vocab_size = log_probs.shape
    if not 0 <= blank < vocab_size or any(t == blank or not 0 <= t < vocab_size for t in tokens):
        raise ValueError("CTC target tokens must be in vocabulary and different from blank")
    repeats = sum(a == b for a, b in zip(tokens, tokens[1:], strict=False))
    if not tokens or frames < len(tokens) + repeats:
        return None
    states = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    states[1::2] = tokens
    can_skip = np.zeros(len(states), dtype=bool)
    can_skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    previous = np.full(len(states), -np.inf)
    previous[:2] = log_probs[0, states[:2]]
    back = np.zeros((frames, len(states)), dtype=np.uint8)
    for t in range(1, frames):
        one = np.r_[-np.inf, previous[:-1]]
        two = np.r_[-np.inf, -np.inf, previous[:-2]]
        two[~can_skip] = -np.inf
        candidates = np.stack([previous, one, two])
        back[t] = np.argmax(candidates, axis=0)
        previous = np.max(candidates, axis=0) + log_probs[t, states]
    state = len(states) - (1 if previous[-1] >= previous[-2] else 2)
    if not np.isfinite(previous[state]):
        return None
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state // 2 if state % 2 else -1
        if t:
            state -= int(back[t, state])
    return path


def _converted_weights(source: Path, cache_dir: Path) -> Path:
    """One-time weights-only PyTorch deserialization; never PyTorch inference."""
    from safetensors.numpy import save_file

    key = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    target_dir = cache_dir / "podcurate-converted" / f"{_CONVERSION_VERSION}-{key}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "model.safetensors"
    with (target_dir / "conversion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return target
        torch = _optional_module("torch", "alignment")
        state = torch.load(source / "pytorch_model.bin", map_location="cpu", weights_only=True)
        weights = {}
        for name, tensor in state.items():
            if name == "wav2vec2.masked_spec_embed":
                continue  # Training-only SpecAugment parameter, absent from inference encoder.
            value = tensor.detach().float().numpy()
            if name.endswith(".parametrizations.weight.original0"):
                name = name.replace(".parametrizations.weight.original0", ".weight_g")
            elif name.endswith(".parametrizations.weight.original1"):
                name = name.replace(".parametrizations.weight.original1", ".weight_v")
            if name.endswith((".conv.weight", ".conv.weight_g", ".conv.weight_v")):
                value = value.swapaxes(1, 2)
            weights[name] = np.ascontiguousarray(value)
        fd, tmp = tempfile.mkstemp(prefix="conversion-", suffix=".safetensors", dir=target_dir)
        os.close(fd)
        try:
            save_file(weights, tmp, metadata={
                "source_revision": source.name, "conversion": _CONVERSION_VERSION,
            })
            os.replace(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return target


class TurkishAligner:
    """Turkish Wav2Vec2 FP32 on MLX, then exact CTC dynamic programming."""

    def __init__(self, model_id=TURKISH_MODEL, revision=TURKISH_REVISION,
                 local_files_only=False, *, cache_dir=None):
        self._mx = _optional_module("mlx.core", "alignment")
        module = _optional_module("mlx_audio.stt.models.mms.mms", "alignment")
        self._source = _snapshot(model_id, revision, local_files_only, cache_dir, [
            "config.json", "vocab.json", "preprocessor_config.json", "pytorch_model.bin",
        ])
        config = json.loads((self._source / "config.json").read_text())
        self._vocab = json.loads((self._source / "vocab.json").read_text())
        preprocessing = json.loads((self._source / "preprocessor_config.json").read_text())
        if config.get("architectures") != ["Wav2Vec2ForCTC"]:
            raise ValueError("Expected a Wav2Vec2ForCTC checkpoint")
        if (config.get("hidden_act") != "gelu" or config.get("add_adapter", False)
                or config.get("adapter_attn_dim") is not None):
            raise ValueError("Unsupported Wav2Vec2 architecture variant")
        if preprocessing.get("sampling_rate") != 16000 or not preprocessing.get("do_normalize"):
            raise ValueError("Expected 16 kHz normalized Wav2Vec2 preprocessing")
        self._blank = config["pad_token_id"]
        if self._vocab.get("<pad>") != self._blank or len(self._vocab) != config["vocab_size"]:
            raise ValueError("CTC blank/vocabulary does not match model config")
        if any(c not in self._vocab for c in "çğıöşü|"):
            raise ValueError("Checkpoint lacks the expected Turkish character vocabulary")
        cache_root = Path(cache_dir) if cache_dir else self._source.parents[2]
        weights_path = _converted_weights(self._source, cache_root)
        with weights_path.open("rb") as stream:
            weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        self._model = module.Model(config)
        self._model.load_weights(str(weights_path), strict=True)
        self._model.eval()
        self._mx.eval(self._model.parameters())
        self._stride = int(np.prod(config["conv_stride"]))
        receptive, jump = 1, 1
        for kernel, stride in zip(config["conv_kernel"], config["conv_stride"], strict=True):
            receptive += (kernel - 1) * jump
            jump *= stride
        self._receptive = receptive
        self.identity = {
            "backend": "mlx-wav2vec2-turkish-ctc", "model_id": model_id,
            "revision": revision, "conversion": _CONVERSION_VERSION,
            "converted_weights_sha256": weights_hash,
            "package_version": _package_version("mlx-audio"),
            "mlx_version": _package_version("mlx"), "dtype": "float32",
            "sample_rate": 16000, "max_audio_seconds": 30, "languages": ["tr"],
            "score_kind": "mean_viterbi_log_probability_including_blank",
            "coverage_kind": "aligned_normalized_reference_characters",
            "normalization": "nfc_turkish_case_remove_punctuation_v1",
            "frame_stride_samples": self._stride, "receptive_field_samples": self._receptive,
            "blank_token_id": self._blank,
        }

    def emissions(self, audio_16k):
        """Unnormalized FP32 CTC logits [frames, vocab]; public for parity validation."""
        if self._model is None:
            raise RuntimeError("Aligner is closed")
        audio = _audio(audio_16k)
        if len(audio) < self._receptive:
            raise ValueError("Audio shorter than the model receptive field")
        # Matches Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm, single unpadded utterance.
        normalized = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        hidden = self._model.wav2vec2(
            self._mx.array(normalized[None, :]), output_hidden_states=False,
        ).last_hidden_state
        logits = self._model.lm_head(hidden)
        self._mx.eval(logits)
        result = np.asarray(logits[0], dtype=np.float32).copy()
        del logits, hidden
        self._mx.clear_cache()
        if not np.isfinite(result).all():
            raise RuntimeError("MLX encoder emitted nonfinite CTC logits")
        return result

    def __call__(self, audio_16k, reference_text, language):
        if language != "tr":
            raise ValueError("TurkishAligner supports tr only")
        if self._model is None:
            raise RuntimeError("Aligner is closed")
        audio, text = _audio(audio_16k), normalize_turkish_reference(reference_text)
        if not text:
            return _result(["empty_normalized_reference"], coverage=0.0)
        if len(audio) < self._receptive:
            return _result(["audio_too_short"], coverage=0.0)
        if not np.any(audio):
            return _result(["silent_audio"], coverage=0.0)
        chars = text.replace(" ", "|")
        unknown = sorted(set(chars) - set(self._vocab))
        if unknown:
            return _result(["unsupported_reference_characters"], coverage=0.0,
                           alignment_unsupported_characters=unknown)
        expected_frames = (len(audio) - self._receptive) // self._stride + 1
        if len(chars) > expected_frames:
            return _result(["reference_too_long"], coverage=0.0)
        tokens = [self._vocab[c] for c in chars]
        logits = self.emissions(audio).astype(np.float64)
        maximum = logits.max(axis=1, keepdims=True)
        log_probs = logits - maximum - np.log(np.exp(logits - maximum).sum(axis=1, keepdims=True))
        path = ctc_viterbi(log_probs, tokens, self._blank)
        if path is None:
            return _result(["no_ctc_alignment_path"], coverage=0.0)
        items, offset = [], 0
        for word in text.split():
            frames = np.flatnonzero((path >= offset) & (path < offset + len(word)))
            items.append({
                "text": word,
                "start": float(frames[0] * self._stride / 16000),
                "end": float((frames[-1] + 1) * self._stride / 16000),
            })
            offset += len(word) + 1
        nonblank = np.flatnonzero(path >= 0)
        emitted = np.full(len(path), self._blank, dtype=np.int64)
        emitted[nonblank] = np.asarray(tokens)[path[nonblank]]
        frame_scores = log_probs[np.arange(len(path)), emitted]
        # Include blank penalties: otherwise omitted words forced into blank
        # frames would disappear from the reported path score.
        score = float(frame_scores.mean())
        return _result([], items, 1.0, score,
                       alignment_token_score=float(frame_scores[nonblank].mean()))

    def close(self):
        _release(self)
