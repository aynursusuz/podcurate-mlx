"""DNSMOS P.835: native MLX inference, with an explicit ONNX reference adapter.

Windowing and nonpersonalized calibration follow Microsoft's DNS-Challenge
DNSMOS/dnsmos_local.py (commit 591184a9fcb2cbdec02520fed81a32bbbf9d73ff).
DNSMOS predicts SIG/BAK/OVRL; it does not establish transcript or speaker fidelity.
"""

from pathlib import Path

import numpy as np

from .audio import sha256

WINDOW_SAMPLES = 144160
POLYNOMIALS = ((-0.08397278, 1.22083953, 0.0052439),
               (-0.13166888, 1.60915514, -0.39604546),
               (-0.06766283, 1.11546468, 0.04602535))


def windows(audio: np.ndarray):
    """Yield the reference 9.01 s windows and 1 s hops, including repeat padding."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("DNSMOS expects one-dimensional 16 kHz mono audio")
    if audio.size == 0:
        raise ValueError("DNSMOS cannot score empty audio")
    if not np.isfinite(audio).all():
        raise ValueError("DNSMOS requires finite audio")
    while audio.size < WINDOW_SAMPLES:
        audio = np.concatenate([audio, audio])
    # Preserve the reference's truncation order; replacing this with ordinary
    # sliding-window count changes its behavior for short repeated clips.
    count = int(np.floor(len(audio) / 16000) - 9.01) + 1
    for index in range(count):
        window = audio[index * 16000:index * 16000 + WINDOW_SAMPLES]
        if window.size == WINDOW_SAMPLES:
            yield window


def _score(audio, infer):
    sums = np.zeros(3, dtype=np.float64)
    count = 0
    for window in windows(audio):
        raw = np.asarray(infer(window), dtype=np.float32)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError("DNSMOS returned invalid scores")
        sums += [np.polyval(poly, raw[j]) for j, poly in enumerate(POLYNOMIALS)]
        count += 1
    return {f"dnsmos_{key}": float(sums[j] / count)
            for j, key in enumerate(("sig", "bak", "ovrl"))}


class DNSMOS:
    """Pinned nonpersonalized P.835 model executed in float32 on MLX Metal.

    The first run downloads the verified official ONNX weights and converts
    their tensors automatically. ONNX is only a weight-container parser here;
    ONNX Runtime is neither imported nor used by this production adapter.
    """

    def __init__(self, model_path: str | None = None, *, local_files_only: bool = False):
        from .dnsmos_mlx import P835

        self.model = P835(model_path, local_files_only=local_files_only)
        self.identity = self.model.identity

    def __call__(self, audio: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("DNSMOS is closed")
        return _score(audio, self.model)

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None


class ONNXDNSMOS:
    """Explicit CPU reference for numerical validation; never a default fallback."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        path = Path(model_path).resolve(strict=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.identity = {"file_sha256": sha256(path), "provider": "CPUExecutionProvider",
                         "variant": "DNSMOS-P835-nonpersonalized-reference"}

    def __call__(self, audio: np.ndarray) -> dict:
        if self.session is None:
            raise RuntimeError("ONNXDNSMOS is closed")
        name = self.session.get_inputs()[0].name
        return _score(audio, lambda window: self.session.run(None, {name: window[None]})[0][0])

    def close(self):
        self.session = None
