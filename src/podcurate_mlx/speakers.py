"""Bounded native MLX diarization and speaker embeddings.

Scores describe model outputs. Speaker IDs are recording-local; cosine scores
are not probabilities or identity decisions. Optional imports happen on init.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .backends import SAMPLE_RATE, _audio, _optional_module, _package_version, _snapshot

NEMOTRON_MODEL = "mlx-community/Nemotron-3-Diarization"
NEMOTRON_REVISION = "59ed2dbfc1346dcea9d423c71306a3a2499c568f"
ECAPA_MODEL = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
ECAPA_REVISION = "e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77"


def _release(owner):
    owner._model = None
    gc.collect()
    owner._mx.clear_cache()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    value = np.dot(left.astype(np.float64), right.astype(np.float64))
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    if not np.isfinite(value) or not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Speaker model returned an invalid embedding")
    return float(np.clip(value / norm, -1.0, 1.0))


class NemotronDiarizer:
    """Up to eight recording-local speaker channels, at 10 ms resolution.

    A new stream resets identity; all chunks within a stream share model state.
    Resume must replay the recording or restore the complete upstream state.
    Input blocks are bounded to <=30 s; the iterable is never materialized.
    """

    def __init__(
        self,
        model_id: str = NEMOTRON_MODEL,
        revision: str = NEMOTRON_REVISION,
        local_files_only: bool = False,
        *,
        activity_threshold: float = 0.5,
        preset: str = "offline",
    ):
        if not np.isfinite(activity_threshold) or not 0 < activity_threshold < 1:
            raise ValueError("activity_threshold must be finite and between zero and one")
        self._mx = _optional_module("mlx.core", "speakers")
        vad = _optional_module("mlx_audio.vad", "speakers")
        path, resolved = _snapshot(model_id, revision, local_files_only)
        self._model = vad.load(path, strict=True)
        self._model.eval()
        self._model.set_streaming_config(preset)
        config = self._model.config
        if config.num_speakers != 8 or config.output_subsampling_factor != 1:
            raise ValueError("This adapter requires Nemotron with eight speakers and native frames")
        self.frame_samples = int(config.processor_config.hop_length)
        if self._model.sample_rate != SAMPLE_RATE or self.frame_samples != 160:
            raise ValueError("This adapter requires a 16 kHz model with 160-sample frames")
        self.activity_threshold = float(activity_threshold)
        self.identity = {
            "backend": "mlx-audio-nemotron-diarization",
            "model_id": model_id,
            "revision": resolved,
            "requested_revision": revision,
            "package_version": _package_version("mlx-audio"),
            "mlx_version": _package_version("mlx"),
            "sample_rate": SAMPLE_RATE,
            "frame_samples": self.frame_samples,
            "max_speakers": 8,
            "streaming_preset": preset,
            "activity_threshold": self.activity_threshold,
            "activity_threshold_calibrated": False,
            "overlap_ratio_denominator": "all_input_samples",
            "speaker_identity_scope": "recording",
            "resume": "replay_recording",
        }

    def stream(self, blocks: Iterable[np.ndarray]) -> Iterator[dict]:
        if self._model is None:
            raise RuntimeError("Diarizer is closed")
        state = self._model.init_streaming_state()

        def emit(block, final=False):
            nonlocal state
            start = int(state.frames_processed) * self.frame_samples
            output, state = self._model.feed(
                block,
                state,
                sample_rate=SAMPLE_RATE,
                final=final,
                threshold=self.activity_threshold,
                min_duration=0.0,
                merge_gap=0.0,
            )
            self._mx.eval(output.speaker_probs)
            probs = np.asarray(output.speaker_probs, dtype=np.float32).copy()
            if probs.ndim != 2 or probs.shape[1] != 8 or not np.isfinite(probs).all():
                raise RuntimeError("Diarizer returned invalid speaker probabilities")
            if np.any(probs < 0) or np.any(probs > 1):
                raise RuntimeError("Diarizer returned probabilities outside [0, 1]")
            return {
                "start_sample": start,
                "probabilities": probs,
                "frame_samples": self.frame_samples,
            }

        for block in blocks:
            result = emit(_audio(block))
            if result["probabilities"].shape[0]:
                yield result
        result = emit(np.empty(0, dtype=np.float32), final=True)
        if result["probabilities"].shape[0]:
            yield result

    def __call__(self, audio_16k: np.ndarray) -> dict:
        audio = _audio(audio_16k)
        results = list(self.stream([audio]))
        probs = (
            np.concatenate([r["probabilities"] for r in results])
            if results
            else np.empty((0, 8), dtype=np.float32)
        )
        active = probs > self.activity_threshold
        counts = active.sum(axis=1)
        turns = []
        for speaker in range(8):
            changes = np.diff(np.pad(active[:, speaker].astype(np.int8), (1, 1)))
            for start, end in zip(
                np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
            ):
                turns.append(
                    {
                        "start": int(start) * self.frame_samples / SAMPLE_RATE,
                        "end": min(int(end) * self.frame_samples, audio.size) / SAMPLE_RATE,
                        "speaker": speaker,
                    }
                )
        used = int(np.count_nonzero(active.any(axis=0)))
        return {
            "diar_speakers": used,
            "overlap_ratio": float(np.count_nonzero(counts >= 2) * self.frame_samples / audio.size),
            "speaker_turns": sorted(turns, key=lambda row: (row["start"], row["speaker"])),
            # A capacity warning, not proof of an additional speaker.
            "speaker_capacity_reached": used == 8,
            "diarization_covered_samples": len(probs) * self.frame_samples,
        }

    def close(self):
        _release(self)


class ECAPASpeaker:
    """VoxCeleb ECAPA embeddings using reviewed, vendored MLX source.

    Consistency is mean adjacent-window cosine, using disjoint 3 s windows and
    a final remainder only when >=1 s. It does not prove a single speaker.
    Status reports metric availability; selection policy decides score cutoffs.
    """

    def __init__(
        self,
        model_id: str = ECAPA_MODEL,
        revision: str = ECAPA_REVISION,
        local_files_only: bool = False,
    ):
        self._mx = _optional_module("mlx.core", "speakers")
        from .ecapa_frontend import compute_fbank
        from .ecapa_model import SpeakerModel

        path, resolved = _snapshot(model_id, revision, local_files_only)
        root = Path(path)
        config = json.loads((root / "config.json").read_text())
        expected = {
            "model_type": "speechbrain-ecapa-voxceleb-speaker",
            "n_mels": 80,
            "embedding_dimension": 192,
            "sample_rate": SAMPLE_RATE,
            "n_fft": 400,
            "hop_length": 160,
            "win_length": 400,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("ECAPA checkpoint is incompatible with the vendored frontend/model")
        weights = root / "model.safetensors"
        with weights.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != config.get("artifact_sha256"):
            raise RuntimeError("ECAPA weight checksum does not match the checkpoint manifest")
        self._model = SpeakerModel()
        self._model.load_weights(list(self._mx.load(str(weights)).items()), strict=True)
        self._model.eval()
        self._fbank = compute_fbank
        self.identity = {
            "backend": "mlx-ecapa-voxceleb",
            "model_id": model_id,
            "revision": resolved,
            "requested_revision": revision,
            "weights_sha256": digest,
            "mlx_version": _package_version("mlx"),
            "numpy_version": _package_version("numpy"),
            "source_revision": config.get("source_revision"),
            "vendored_source_revision": ECAPA_REVISION,
            "vendored_source_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("ecapa_frontend.py", "ecapa_model.py")
            },
            "sample_rate": SAMPLE_RATE,
            "max_audio_seconds": 30,
            "embedding_dimension": 192,
            "frontend": "speechbrain_fbank_numpy_80mel_400fft_160hop",
            "consistency": "mean_adjacent_3s_windows_min_1s_remainder_or_halves_if_2_to_4s",
            "cosine_is_probability": False,
            "status_semantics": "metric_availability",
        }

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Speaker model is closed")
        audio = _audio(audio_16k)
        features = self._fbank(audio, 80)
        if not 10 <= features.shape[0] <= 3001:
            raise ValueError(
                "ECAPA input requires 10..3001 mel frames (at least 90 ms; at most 30 s)"
            )
        result = self._model(self._mx.array(features[None, :, :]))
        self._mx.eval(result)
        embedding = np.asarray(result, dtype=np.float32).reshape(-1).copy()
        if embedding.shape != (192,) or not np.isfinite(embedding).all():
            raise RuntimeError("Speaker model returned an invalid embedding")
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            raise RuntimeError("Speaker model returned a zero embedding")
        return embedding / norm

    def __call__(
        self,
        audio_16k: np.ndarray,
        reference_audio: np.ndarray | None = None,
        *,
        reference_embedding: np.ndarray | None = None,
    ) -> dict:
        audio = _audio(audio_16k)
        if reference_audio is not None and reference_embedding is not None:
            raise ValueError("Supply reference_audio or reference_embedding, not both")
        similarity = None
        if reference_audio is not None:
            reference_embedding = self.embed(reference_audio)
        if reference_embedding is not None:
            reference_embedding = np.asarray(reference_embedding, dtype=np.float32)
            if reference_embedding.shape != (192,):
                raise ValueError("reference_embedding must have 192 elements from this model")
            similarity = _cosine(self.embed(audio), reference_embedding)
        step = 3 * SAMPLE_RATE
        windows = [
            audio[i : i + step]
            for i in range(0, audio.size, step)
            if audio[i : i + step].size >= SAMPLE_RATE
        ]
        # Two equal windows make 2..<4 s segments measurable without duplicating PCM.
        if len(windows) == 1 and audio.size >= 2 * SAMPLE_RATE:
            split = audio.size // 2
            windows = [audio[:split], audio[split:]]
        embeddings = [self.embed(window) for window in windows]
        values = [_cosine(a, b) for a, b in zip(embeddings[:-1], embeddings[1:], strict=True)]
        return {
            "speaker_similarity": similarity,
            "speaker_consistency": float(np.mean(values)) if values else None,
            "speaker_reference_available": reference_embedding is not None,
            "speaker_consistency_windows": len(windows),
            "speaker_status": "ok" if values else "review",
            "speaker_review_reason": None if values else "insufficient_duration_for_consistency",
        }

    def close(self):
        _release(self)
