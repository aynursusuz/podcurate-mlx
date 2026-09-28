"""Graph-specific MLX port of Microsoft's nonpersonalized DNSMOS P.835 model.

Model and reference implementation: microsoft/DNS-Challenge, commit
591184a9fcb2cbdec02520fed81a32bbbf9d73ff. This is not a general ONNX converter.
Only the exact model checksum below is supported, in full float32 precision.
"""

import importlib.metadata
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from .audio import sha256

SOURCE_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
SOURCE_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/"
    "591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
GRAPH_VERSION = "p835-mlx-f32-v1"


def cache_dir() -> Path:
    custom = os.environ.get("PODCURATE_CACHE_DIR")
    base = Path(custom) if custom else Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ) / "podcurate-mlx"
    return base / "dnsmos"


def resolve_model(model_path: str | None, *, local_files_only: bool = False) -> Path:
    """Download only the commit-pinned model; validate local overrides equally."""
    if model_path is not None:
        path = Path(model_path).expanduser().resolve(strict=True)
    else:
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{SOURCE_SHA256}.onnx"
        if not path.exists():
            if local_files_only:
                message = f"DNSMOS model is not cached and offline mode is enabled: {path}"
                raise FileNotFoundError(message)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                    temporary = Path(output.name)
                    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                        while block := response.read(1 << 20):
                            output.write(block)
                if sha256(temporary) != SOURCE_SHA256:
                    raise ValueError("Downloaded DNSMOS model checksum mismatch")
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    if sha256(path) != SOURCE_SHA256:
        raise ValueError("Unsupported DNSMOS model: expected pinned nonpersonalized P.835 checksum")
    return path


def _convert(path: Path, destination: Path):
    """Extract constant tensors. ONNX is imported only when the cache is absent."""
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        message = "First DNSMOS conversion requires the 'converter' extra (onnx)"
        raise ImportError(message) from error

    tensors = {item.name: numpy_helper.to_array(item) for item in onnx.load(path).graph.initializer}
    weights = {
        "real": tensors["time2freq/stft-real/kernel:0"][:, :, 0].T.copy(),
        "imag": tensors["time2freq/stft-imag/kernel:0"][:, :, 0].T.copy(),
    }
    for index in range(7):
        name = "conv2d" + (f"_{index}" if index else "")
        weights[f"conv{index}_weight"] = tensors[f"{name}/kernel:0"].transpose(0, 2, 3, 1)
        weights[f"conv{index}_bias"] = tensors[f"{name}/bias:0"]
    for index, name in enumerate(("dense", "dense_1", "dense_3")):
        prefix = f"mos_estimator_logpow/{name}"
        weights[f"dense{index}_weight"] = tensors[f"{prefix}/MatMul/ReadVariableOp/resource:0"]
        weights[f"dense{index}_bias"] = tensors[f"{prefix}/BiasAdd/ReadVariableOp/resource:0"]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, **weights)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class P835:
    def __init__(self, model_path: str | None = None, *, local_files_only: bool = False):
        import mlx.core as mx

        if not mx.metal.is_available():
            raise RuntimeError("Native DNSMOS requires Apple Silicon MLX Metal")
        source = resolve_model(model_path, local_files_only=local_files_only)
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        converted = directory / f"{SOURCE_SHA256}-{GRAPH_VERSION}.npz"
        if not converted.exists():
            _convert(source, converted)
        self.mx = mx
        with np.load(converted, allow_pickle=False) as weights, mx.stream(mx.gpu):
            self.weights = {name: mx.array(weights[name], dtype=mx.float32) for name in weights}
            mx.eval(self.weights)
        self.identity = {
            "file_sha256": SOURCE_SHA256, "weights_sha256": sha256(converted),
            "provider": "MLX-Metal", "variant": "DNSMOS-P835-nonpersonalized",
            "graph_version": GRAPH_VERSION, "mlx_version": importlib.metadata.version("mlx"),
            "graph_source_sha256": sha256(Path(__file__)),
            "dtype": "float32", "source_url": SOURCE_URL,
        }

    def __call__(self, window: np.ndarray, trace: dict | None = None) -> np.ndarray:
        """One window; optional trace exposes evaluated tensors for parity tests."""
        if not self.weights:
            raise RuntimeError("DNSMOS model is closed")
        if window.shape != (144160,):
            raise ValueError("P835 expects exactly 144160 samples")
        mx = self.mx
        weights = self.weights

        def record(name, value):
            mx.eval(value)
            if trace is not None:
                trace[name] = np.array(value)
            return value

        with mx.stream(mx.gpu):
            audio = mx.array(window, dtype=mx.float32)[None]
            frames = record("frames", mx.concatenate(
                [audio[:, :144000].reshape(1, 900, 160),
                 audio[:, 160:].reshape(1, 900, 160)], axis=2))
            real = record("real", frames @ weights["real"])
            imag = record("imag", frames @ weights["imag"])
            # Retain the source graph's sqrt/pow and natural-log/division order.
            power = mx.power(mx.sqrt(real * real + imag * imag), 2.0)
            value = record("logpower", mx.log(mx.maximum(power, 1e-12)) / mx.array(2.3025851))
            value = value[..., None]
            for index in range(7):
                value = record(f"conv{index}", mx.maximum(
                    mx.conv2d(value, weights[f"conv{index}_weight"], padding=1)
                    + weights[f"conv{index}_bias"], 0))
                if index in (3, 4, 5):
                    batch, height, width, channels = value.shape
                    value = value[:, :height // 2 * 2, :width // 2 * 2, :]
                    value = value.reshape(batch, height // 2, 2, width // 2, 2, channels)
                    value = record(f"pool{index - 3}", mx.max(value, axis=(2, 4)))
            value = record("global_max", mx.max(value, axis=(1, 2)))
            for index in range(3):
                value = value @ weights[f"dense{index}_weight"] + weights[f"dense{index}_bias"]
                if index < 2:
                    value = mx.maximum(value, 0)
                value = record(f"dense{index}", value)
            return np.array(value[0])

    def close(self):
        self.mx.synchronize()
        self.weights = {}
        self.mx.clear_cache()
