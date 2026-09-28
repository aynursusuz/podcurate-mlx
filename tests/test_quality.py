"""Offline DNSMOS adapter tests. Fake ONNX outputs do not test MOS accuracy."""

from types import SimpleNamespace

import numpy as np
import pytest

from podcurate_mlx.quality import DNSMOS


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
    result = DNSMOS.__new__(DNSMOS)
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
