"""Optional MLX model adapters. Imports never download models.

Constructors resolve a requested Hugging Face revision once. All inference then
uses that immutable local snapshot. These adapters do not provide MOS scores or
certify hallucination detection.
"""

from __future__ import annotations

import importlib
import math
import re
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
MAX_SAMPLES = 30 * SAMPLE_RATE
LANGUAGES = frozenset({"en", "zh", "ja", "tr"})
_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")


def _optional_module(name: str, extra: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"{name} is required for this backend. On Apple Silicon, install "
            f"podcurate-mlx[{extra}] in a native Python environment. "
            f"Original import error: {exc}"
        ) from exc


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _snapshot(model_id: str, revision: str, local_files_only: bool) -> tuple[str, str]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty Hugging Face repository ID")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be explicit; a full commit SHA is preferred")
    hub = _optional_module("huggingface_hub", "mlx")
    path = Path(
        hub.snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()
    # snapshot_download without local_dir returns .../snapshots/<commit-sha>.
    # Refuse an unidentifiable snapshot rather than record a mutable branch.
    if not _COMMIT.fullmatch(path.name) or not path.is_dir():
        raise RuntimeError(f"Cannot establish an immutable model snapshot: {path}")
    return str(path), path.name.lower()


def _audio(audio_16k: np.ndarray) -> np.ndarray:
    """Check bounded, already resampled mono audio; never infer its sample rate."""
    if not isinstance(audio_16k, np.ndarray):
        raise TypeError("audio_16k must be a mono NumPy array sampled at 16000 Hz")
    if audio_16k.ndim != 1 or audio_16k.size == 0:
        raise ValueError("audio_16k must be a nonempty one-dimensional mono array")
    if audio_16k.size > MAX_SAMPLES:
        raise ValueError("Model input exceeds 30 seconds; prepare shorter segments first")
    if not np.issubdtype(audio_16k.dtype, np.floating):
        raise ValueError("audio_16k must contain floating-point PCM samples")
    audio = np.ascontiguousarray(audio_16k, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("audio_16k contains nonfinite samples")
    return audio


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, (str, bytes, bool)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _segment_mean(segments: list[dict], key: str) -> float | None:
    weighted: list[float] = []
    weights: list[float] = []
    for segment in segments:
        value = _finite(segment.get(key))
        start, end = _finite(segment.get("start")), _finite(segment.get("end"))
        if value is None or start is None or end is None or end <= start:
            continue
        duration = end - start
        if not math.isfinite(duration) or not math.isfinite(value * duration):
            continue
        weighted.append(value * duration)
        weights.append(duration)
    if not weights:
        return None
    try:
        result = math.fsum(weighted) / math.fsum(weights)
    except (OverflowError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


class WhisperASR:
    """MLX Whisper inference for one <=30 s, 16 kHz segment at a time.

    Metrics are duration-weighted means over returned segments with positive
    duration and finite values. They describe decoder output, not calibrated
    speech quality. Language is supplied by the caller, not independently tested.
    """

    def __init__(self, model_id: str, revision: str, local_files_only: bool = False):
        whisper = _optional_module("mlx_whisper", "mlx")
        self._transcribe = whisper.transcribe
        self._path, resolved = _snapshot(model_id, revision, local_files_only)
        self.identity = {
            "backend": "mlx-whisper",
            "model_id": model_id,
            "revision": resolved,
            "requested_revision": revision,
            "package_version": _package_version("mlx-whisper"),
            "mlx_version": _package_version("mlx"),
            "metric_aggregation": "duration_weighted_finite_segments",
            "max_audio_seconds": 30,
            "sample_rate": SAMPLE_RATE,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "task": "transcribe",
            "language_mode": "forced",
        }

    def __call__(self, audio_16k: np.ndarray, language: str) -> dict:
        if language not in LANGUAGES:
            raise ValueError("language must be one of: en, zh, ja, tr")
        audio = _audio(audio_16k)
        result = self._transcribe(
            audio,
            path_or_hf_repo=self._path,
            language=language,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            verbose=None,
        )
        segments = result.get("segments") or []
        return {
            "text": result.get("text", ""),
            # Whisper returns the requested language when one is supplied.
            "detected_language": None,
            "language_probability": None,
            "avg_logprob": _segment_mean(segments, "avg_logprob"),
            "no_speech_prob": _segment_mean(segments, "no_speech_prob"),
            "compression_ratio": _segment_mean(segments, "compression_ratio"),
        }

    def close(self):
        import gc

        module = importlib.import_module("mlx_whisper.transcribe")
        module.ModelHolder.model = None
        module.ModelHolder.model_path = None
        gc.collect()
        _optional_module("mlx.core", "mlx").clear_cache()


class SileroVAD:
    """Optional community MLX Silero port for bounded audio segments.

    stream() retains recurrent state and unfinished speech across decoder blocks.
    Thirty-second forced cuts are explicitly marked for review.
    """

    def __init__(self, model_id: str, revision: str, local_files_only: bool = False):
        vad = _optional_module("mlx_audio.vad", "vad")
        path, resolved = _snapshot(model_id, revision, local_files_only)
        self._model = vad.load(path, strict=True)
        self.identity = {
            "backend": "mlx-audio-silero-vad",
            "model_id": model_id,
            "revision": resolved,
            "requested_revision": revision,
            "package_version": _package_version("mlx-audio"),
            "mlx_version": _package_version("mlx"),
            "sample_rate": SAMPLE_RATE,
            "max_audio_seconds": 30,
            "stream_algorithm": "stateful-512-v1",
        }

    def __call__(self, audio_16k: np.ndarray) -> list[tuple[float, float]]:
        audio = _audio(audio_16k)
        duration = audio.size / SAMPLE_RATE
        timestamps = self._model.get_speech_timestamps(
            audio, sample_rate=SAMPLE_RATE, return_seconds=True
        )
        output = []
        for item in timestamps:
            start, end = _finite(item.get("start")), _finite(item.get("end"))
            if start is None or end is None or start < 0 or end <= start:
                raise RuntimeError("VAD returned an invalid timestamp")
            # Some ports round output seconds. Clamp the final boundary only.
            if start >= duration or end > duration + 0.1:
                raise RuntimeError("VAD returned a timestamp outside the input segment")
            output.append((start, min(end, duration)))
        return output

    def stream(self, blocks):
        """Yield (start_sample, end_sample, forced_cut); offsets relative to stream start.

        Threshold/padding are segmentation settings, not calibrated quality limits.
        The state machine consumes exactly 512 samples, independent of IO block size.
        """
        state = None
        pending = np.empty(0, np.float32)
        position = total = 0
        speech_start = silence_start = None
        continuation = False
        threshold = float(self._model.config.threshold)
        min_speech = round(self._model.config.min_speech_duration_ms * 16)
        min_silence = round(self._model.config.min_silence_duration_ms * 16)
        pad = round(self._model.config.speech_pad_ms * 16)

        def windows():
            nonlocal pending, total
            for block in blocks:
                total += len(block)
                pending = np.concatenate((pending, block))
                while len(pending) >= 512:
                    yield pending[:512]
                    pending = pending[512:]
            if len(pending):
                yield np.pad(pending, (0, 512 - len(pending)))

        for window in windows():
            prob, state = self._model.feed(window, state=state, sample_rate=16000)
            probability = float(np.asarray(prob).reshape(-1)[0])
            if probability >= threshold:
                silence_start = None
                if speech_start is None:
                    speech_start = max(0, position - pad)
            elif probability < max(0.01, threshold - 0.15) and speech_start is not None:
                if silence_start is None:
                    silence_start = position
                if position - silence_start >= min_silence:
                    stop = min(total, silence_start + pad)
                    if stop - speech_start >= min_speech or continuation:
                        yield speech_start, stop, continuation
                    speech_start = silence_start = None
                    continuation = False
            if speech_start is not None and position + 512 - speech_start >= MAX_SAMPLES:
                stop = speech_start + MAX_SAMPLES
                yield speech_start, stop, True
                speech_start, continuation = stop, True
            position += 512
        if speech_start is not None and total > speech_start:
            if total - speech_start >= min_speech or continuation:
                yield speech_start, total, continuation

    def close(self):
        self._model = None
        _optional_module("mlx.core", "vad").clear_cache()
