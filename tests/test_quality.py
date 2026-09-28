"""Offline contracts plus opt-in, real MLX/ONNX numerical parity checks."""

import builtins
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from podcurate_mlx.audio import sha256
from podcurate_mlx.dnsmos_mlx import resolve_model
from podcurate_mlx.quality import DNSMOS, ONNXDNSMOS, windows


class RecordingSession:
    def __init__(self):
        self.windows = []

    def get_inputs(self):
        return [SimpleNamespace(name="audio_input")]

    def run(self, outputs, feed):
        assert outputs is None
        self.windows.append(feed["audio_input"].copy())
        return [np.array([[3.0, 4.0, 5.0]], dtype=np.float32)]


@pytest.fixture
def scorer():
    result = ONNXDNSMOS.__new__(ONNXDNSMOS)
    result.session = RecordingSession()
    return result


def test_short_audio_is_repeated_and_cpu_output_calibration_is_applied(scorer):
    signal = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    result = scorer(signal)
    assert result == pytest.approx({
        "dnsmos_sig": 2.91200747, "dnsmos_bak": 3.93387302, "dnsmos_ovrl": 3.931778,
    })
    window = scorer.session.windows[0]
    assert window.shape == (1, 144160)
    assert window.dtype == np.float32
    np.testing.assert_array_equal(window[0, :8], [0.25, -0.5, 0.75, -1] * 2)
    np.testing.assert_array_equal(signal, [0.25, -0.5, 0.75, -1])


def test_long_audio_is_scored_in_bounded_overlapping_windows(scorer):
    signal = np.arange(16000 * 12, dtype=np.float32) / 200000
    scorer(signal)
    assert len(scorer.session.windows) >= 2
    assert all(window.shape == (1, 144160) for window in scorer.session.windows)
    np.testing.assert_array_equal(scorer.session.windows[1][0], signal[16000:16000 + 144160])


def test_empty_audio_fails_before_onnx_inference(scorer):
    with pytest.raises(ValueError, match="empty"):
        scorer(np.array([], dtype=np.float32))
    assert scorer.session.windows == []


@pytest.mark.parametrize("audio", [np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2))])
def test_invalid_audio_never_reaches_reference_model(scorer, audio):
    with pytest.raises(ValueError):
        scorer(audio)
    assert scorer.session.windows == []


def test_reference_hop_count_at_ten_seconds(scorer):
    # The upstream expression emits one, not two, windows for exactly 10 s.
    scorer(np.zeros(160000, dtype=np.float32))
    assert len(scorer.session.windows) == 1


def test_native_adapter_uses_mlx_model_and_releases_it(monkeypatch):
    class RecordingMLX:
        identity = {"provider": "MLX-Metal"}
        closed = False

        def __init__(self, path, *, local_files_only=False):
            assert path is None
            assert not local_files_only

        def __call__(self, window):
            assert window.shape == (144160,)
            return np.array([3, 4, 5], dtype=np.float32)

        def close(self):
            self.closed = True

    monkeypatch.setattr("podcurate_mlx.dnsmos_mlx.P835", RecordingMLX)
    scorer = DNSMOS()
    model = scorer.model
    assert scorer.identity["provider"] == "MLX-Metal"
    assert scorer(np.zeros(144160, np.float32))["dnsmos_sig"] == pytest.approx(2.91200747)
    scorer.close()
    scorer.close()
    assert model.closed
    with pytest.raises(RuntimeError, match="closed"):
        scorer(np.zeros(144160, np.float32))


def test_unrecognized_model_rejected_before_conversion(tmp_path):
    path = tmp_path / "different.onnx"
    path.write_bytes(b"not the pinned DNSMOS model")
    with pytest.raises(ValueError, match="checksum"):
        resolve_model(str(path))


def test_failed_download_checksum_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("PODCURATE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="checksum"):
        resolve_model(None)
    assert list((tmp_path / "dnsmos").iterdir()) == []


def test_offline_missing_model_does_not_attempt_network(tmp_path, monkeypatch):
    monkeypatch.setenv("PODCURATE_CACHE_DIR", str(tmp_path))

    def forbidden_network(*args, **kwargs):
        raise AssertionError("Offline resolution must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_network)
    with pytest.raises(FileNotFoundError, match="offline"):
        resolve_model(None, local_files_only=True)


@pytest.fixture
def real_model_path():
    value = os.environ.get("PODCURATE_DNSMOS_TEST_MODEL")
    if not value:
        pytest.skip("Set PODCURATE_DNSMOS_TEST_MODEL for real local-weight parity; no downloads")
    pytest.importorskip("mlx.core")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    return str(Path(value).resolve(strict=True))


def test_real_native_calibrated_parity_without_onnx_runtime_import(
        real_model_path, tmp_path, monkeypatch):
    monkeypatch.setenv("PODCURATE_CACHE_DIR", str(tmp_path))
    native = DNSMOS(real_model_path)
    reference = ONNXDNSMOS(real_model_path)
    samples = [np.zeros(144160, dtype=np.float32),
               np.random.default_rng(314159).normal(0, 0.05, 16000 * 12).astype(np.float32)]
    expected = [reference(audio) for audio in samples]
    original_import = builtins.__import__

    def disallow_onnx(name, *args, **kwargs):
        if name.split(".")[0] in ("onnx", "onnxruntime"):
            raise AssertionError("Cached native inference must not import ONNX/ONNX Runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", disallow_onnx)
    native.close()
    native = DNSMOS(real_model_path)
    try:
        for audio, target in zip(samples, expected, strict=True):
            assert native(audio) == pytest.approx(target, abs=2e-5)
    finally:
        native.close()
        reference.close()


def intermediate_reference(real_model_path):
    import onnx
    import onnxruntime as ort

    mapping = {
        "frames": "mos_estimator_logpow/concat:0", "real": "transpose_output7",
        "imag": "transpose_output8", "logpower": "mos_estimator_logpow/truediv:0",
        "global_max": "mos_estimator_logpow/global_max_pooling2d/Max:0",
        "dense0": "mos_estimator_logpow/dense/Relu:0",
        "dense1": "mos_estimator_logpow/dense_1/Relu:0", "dense2": "Identity:0",
        "pool0": "mos_estimator_logpow/conv2d_3/Relu:0_pooling0",
        "pool1": "mos_estimator_logpow/max_pooling2d/MaxPool_1_conv0",
        "pool2": "mos_estimator_logpow/max_pooling2d/MaxPool_2_conv0",
    }
    for index in range(7):
        name = "conv2d" + (f"_{index}" if index else "")
        mapping[f"conv{index}"] = f"mos_estimator_logpow/{name}/Relu:0"
    graph = onnx.load(real_model_path)
    existing = {item.name for item in graph.graph.output}
    for name in mapping.values():
        if name not in existing:
            graph.graph.output.append(onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, None))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(graph.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    return mapping, session


def assert_tensor_parity(native, audio, mapping, session):
    trace = {}
    native.model(audio, trace=trace)
    reference = session.run(list(mapping.values()), {"input_1": audio[None]})
    for name, expected in zip(mapping, reference, strict=True):
        if name.startswith(("conv", "pool")):
            expected = expected.transpose(0, 2, 3, 1)
        np.testing.assert_allclose(trace[name], expected, rtol=2e-4, atol=2e-5, err_msg=name)


def test_real_intermediate_tensor_parity(real_model_path, tmp_path, monkeypatch):
    monkeypatch.setenv("PODCURATE_CACHE_DIR", str(tmp_path))
    mapping, session = intermediate_reference(real_model_path)
    audio = np.random.default_rng(314159).normal(0, 0.05, 144160).astype(np.float32)
    native = DNSMOS(real_model_path)
    try:
        assert_tensor_parity(native, audio, mapping, session)
    finally:
        native.close()


def test_real_four_language_speech_parity(real_model_path, tmp_path, monkeypatch):
    fixture_dir = os.environ.get("PODCURATE_DNSMOS_TEST_FIXTURES")
    if not fixture_dir:
        pytest.skip("Set PODCURATE_DNSMOS_TEST_FIXTURES to the local FLEURS fixture directory")
    directory = Path(fixture_dir).resolve(strict=True)
    rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    assert {row["language"] for row in rows} == {"en", "zh", "ja", "tr"}
    monkeypatch.setenv("PODCURATE_CACHE_DIR", str(tmp_path))
    mapping, session = intermediate_reference(real_model_path)
    native, reference = DNSMOS(real_model_path), ONNXDNSMOS(real_model_path)
    try:
        for row in rows:
            path = directory / row["audio"]
            assert sha256(path) == row["sha256"]
            decoded = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                 "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                capture_output=True, check=True,
            )
            audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
            assert native(audio) == pytest.approx(reference(audio), abs=2e-5), row["id"]
            assert_tensor_parity(native, next(windows(audio)), mapping, session)
    finally:
        native.close()
        reference.close()
