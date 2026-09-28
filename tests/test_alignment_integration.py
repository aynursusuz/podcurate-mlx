"""Opt-in real-model tests: use pinned local models and a real audio manifest.

PODCURATE_ALIGNMENT_FIXTURES=/path/manifest.jsonl
PODCURATE_ALIGNMENT_CACHE=/path/huggingface-cache
pytest tests/test_alignment_integration.py
"""

import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pytest

from podcurate_mlx.alignment import QwenAligner, TurkishAligner
from podcurate_mlx.audio import decode, duration

pytestmark = pytest.mark.skipif(
    not os.environ.get("PODCURATE_ALIGNMENT_FIXTURES"),
    reason="Set PODCURATE_ALIGNMENT_FIXTURES to opt into real-model validation",
)


@pytest.fixture(scope="module")
def real_inputs():
    manifest = Path(os.environ["PODCURATE_ALIGNMENT_FIXTURES"]).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    data = []
    for row in rows:
        path = manifest.parent / row["audio"]
        audio = decode(path, 0.0, duration(path))
        assert audio.ndim == 1 and len(audio) <= 480000
        data.append((row, audio))
    assert {row["language"] for row, _ in data} == {"en", "zh", "ja", "tr"}
    return data


@pytest.fixture(scope="module")
def locked_cache():
    cache = Path(os.environ["PODCURATE_ALIGNMENT_CACHE"]).resolve()
    lock_path = Path(os.environ.get("PODCURATE_ALIGNMENT_GPU_LOCK", cache.parent / "gpu.lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield cache


def test_turkish_mlx_logits_match_original_pytorch(real_inputs, locked_cache):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    samples = [(row, audio) for row, audio in real_inputs if row["language"] == "tr"]
    model = TurkishAligner(cache_dir=locked_cache, local_files_only=True)
    source = str(model._source)
    try:
        logits = [model.emissions(audio) for _, audio in samples]
        for row, audio in samples:
            result = model(audio, row["reference_text"], "tr")
            assert result["alignment_status"] == "ok"
            assert result["alignment_coverage"] == 1.0
            assert result["alignment_score"] <= 0
    finally:
        model.close()
    reference = Wav2Vec2ForCTC.from_pretrained(source, local_files_only=True).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(source, local_files_only=True)
    for (_, audio), actual in zip(samples, logits, strict=True):
        features = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            expected = reference(**features).logits[0].numpy()
        assert actual.shape == expected.shape
        error = np.abs(actual - expected)
        # Numerical conversion checks, not corpus-quality acceptance thresholds.
        assert error.max() < 0.01
        assert error.mean() < 0.001
        assert (actual.argmax(1) == expected.argmax(1)).mean() >= 0.995


def test_qwen_real_en_zh_ja_timestamp_contract(real_inputs, locked_cache):
    model = QwenAligner(cache_dir=locked_cache, local_files_only=True)
    try:
        for row, audio in real_inputs:
            if row["language"] == "tr":
                continue
            result = model(audio, row["reference_text"], row["language"])
            assert result["alignment_score"] is None
            assert result["alignment"]
            assert result["alignment_status"] in {"ok", "review"}
            assert 0 <= result["alignment_coverage"] <= 1
            for item in result["alignment"]:
                assert np.isfinite([item["start"], item["end"]]).all()
                assert item["end"] > item["start"] >= 0
    finally:
        model.close()
