"""Offline contract tests; these do not validate model accuracy or performance."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from podcurate_mlx import backends

SHA = "a" * 40


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / SHA
    snapshot.mkdir(parents=True)
    transcribe = Mock(return_value={"text": "ok", "language": "en", "segments": []})
    download = Mock(return_value=str(snapshot))
    vad_model = SimpleNamespace(get_speech_timestamps=Mock(return_value=[]))
    vad_load = Mock(return_value=vad_model)
    modules = {
        "mlx_whisper": SimpleNamespace(transcribe=transcribe),
        "huggingface_hub": SimpleNamespace(snapshot_download=download),
        "mlx_audio.vad": SimpleNamespace(load=vad_load),
    }
    monkeypatch.setattr(backends.importlib, "import_module", modules.__getitem__)
    return SimpleNamespace(
        transcribe=transcribe, download=download, snapshot=snapshot,
        vad_model=vad_model, vad_load=vad_load,
    )


def test_snapshot_resolved_once_and_language_is_not_claimed_detected(fake_models):
    asr = backends.WhisperASR("org/model", "release-tag", local_files_only=True)
    for language in ("en", "zh", "ja", "tr"):
        output = asr(np.zeros(16000, dtype=np.float32), language)
        assert output["detected_language"] is None
        assert output["language_probability"] is None
    fake_models.download.assert_called_once_with(
        repo_id="org/model", revision="release-tag", local_files_only=True
    )
    assert asr.identity["revision"] == SHA
    kwargs = fake_models.transcribe.call_args.kwargs
    assert kwargs["path_or_hf_repo"] == str(fake_models.snapshot)
    assert kwargs["language"] == "tr"
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["temperature"] == 0.0
    assert kwargs["task"] == "transcribe"
    assert "beam_size" not in kwargs


def test_duration_weighted_metrics_skip_nonfinite_and_zero_duration(fake_models):
    fake_models.transcribe.return_value = {
        "text": "sample", "segments": [
            {"start": 0, "end": 1, "avg_logprob": -1.0,
             "no_speech_prob": float("nan"), "compression_ratio": 2.0},
            {"start": 1, "end": 4, "avg_logprob": -3.0,
             "no_speech_prob": 0.4, "compression_ratio": float("inf")},
            {"start": 4, "end": 4, "avg_logprob": 1234.0},
        ],
    }
    asr = backends.WhisperASR("org/model", SHA)
    output = asr(np.zeros(64000, dtype=np.float32), "en")
    assert output["avg_logprob"] == -2.5
    assert output["no_speech_prob"] == pytest.approx(0.4)
    assert output["compression_ratio"] == 2.0


def test_empty_decoder_segments_remain_missing_scores(fake_models):
    output = backends.WhisperASR("org/model", SHA)(np.zeros(100, np.float32), "en")
    assert output["avg_logprob"] is None
    assert output["no_speech_prob"] is None
    assert output["compression_ratio"] is None


@pytest.mark.parametrize("audio", [
    np.array([], dtype=np.float32),
    np.zeros((1, 16000), dtype=np.float32),
    np.zeros(480001, dtype=np.float32),
    np.array([float("nan")], dtype=np.float32),
    np.array([float("inf")], dtype=np.float32),
    np.array([100], dtype=np.int16),
])
def test_invalid_audio_rejected_before_inference(fake_models, audio):
    asr = backends.WhisperASR("org/model", SHA)
    with pytest.raises(ValueError):
        asr(audio, "en")
    fake_models.transcribe.assert_not_called()


def test_unsupported_language_rejected(fake_models):
    asr = backends.WhisperASR("org/model", SHA)
    with pytest.raises(ValueError, match="language"):
        asr(np.zeros(100, dtype=np.float32), "auto")
    fake_models.transcribe.assert_not_called()


def test_unresolved_revision_is_not_recorded_as_pinned(fake_models, tmp_path):
    fake_models.download.return_value = str(tmp_path)
    with pytest.raises(RuntimeError, match="immutable"):
        backends.WhisperASR("org/model", "main")


def test_missing_dependency_has_actionable_message(monkeypatch):
    def missing(name):
        raise ImportError("No module named mlx_whisper")
    monkeypatch.setattr(backends.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"podcurate-mlx\[mlx\]"):
        backends.WhisperASR("org/model", SHA)


def test_missing_vad_dependency_names_vad_extra(monkeypatch):
    def missing(name):
        raise ImportError("No module named mlx_audio")
    monkeypatch.setattr(backends.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"podcurate-mlx\[vad\]"):
        backends.SileroVAD("org/vad", SHA)


def test_vad_uses_local_snapshot_and_seconds(fake_models):
    fake_models.vad_model.get_speech_timestamps.return_value = [{"start": 0.1, "end": 0.7}]
    vad = backends.SileroVAD("org/vad", SHA)
    assert vad(np.zeros(16000, np.float32)) == [(0.1, 0.7)]
    fake_models.vad_load.assert_called_once_with(str(fake_models.snapshot), strict=True)
    kwargs = fake_models.vad_model.get_speech_timestamps.call_args.kwargs
    assert kwargs == {"sample_rate": 16000, "return_seconds": True}


def test_vad_invalid_output_fails_instead_of_silent_accept(fake_models):
    fake_models.vad_model.get_speech_timestamps.return_value = [{"start": 0.1, "end": 8.0}]
    vad = backends.SileroVAD("org/vad", SHA)
    with pytest.raises(RuntimeError, match="outside"):
        vad(np.zeros(16000, np.float32))
