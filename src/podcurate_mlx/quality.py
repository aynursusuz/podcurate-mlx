"""Optional DNSMOS P.835 reference inference on CPU; explicitly not an MLX port.

Window/calibration source: microsoft/DNS-Challenge, DNSMOS/dnsmos_local.py.
No network download: users provide the non-personalized sig_bak_ovr.onnx file.
"""

from pathlib import Path

import numpy as np

from .audio import sha256


class DNSMOS:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        path = Path(model_path).resolve(strict=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.identity = {"file_sha256": sha256(path), "provider": "CPUExecutionProvider",
                         "variant": "DNSMOS-P835-nonpersonalized"}

    def __call__(self, audio: np.ndarray) -> dict:
        width = 144160  # 9.01 seconds at 16kHz
        if audio.size == 0:
            raise ValueError("DNSMOS cannot score empty audio")
        while audio.size < width:
            audio = np.concatenate([audio, audio])
        count = int(np.floor(len(audio) / 16000) - 9.01) + 1
        name = self.session.get_inputs()[0].name
        raw = np.array([
            self.session.run(None, {name: audio[i * 16000:i * 16000 + width][None]})[0][0]
            for i in range(count)
        ])
        polys = [(-0.08397278, 1.22083953, 0.0052439),
                 (-0.13166888, 1.60915514, -0.39604546),
                 (-0.06766283, 1.11546468, 0.04602535)]
        return {f"dnsmos_{key}": float(np.polyval(poly, raw[:, j]).mean())
                for j, (key, poly) in enumerate(zip(("sig", "bak", "ovrl"), polys, strict=True))}
