"""Audit regressions: real file/SQLite/FFmpeg behavior, no model or network access."""

import json
import shutil
import sqlite3
import wave
from pathlib import Path

import numpy as np
import pytest

from podcurate_mlx import pipeline, stages
from podcurate_mlx.audio import decode, sha256
from podcurate_mlx.curation import calibrate
from podcurate_mlx.preparing import prepare_stream


class ASR:
    identity = {"backend": "audit-fake-asr", "revision": "1"}

    def __init__(self):
        self.calls = 0

    def __call__(self, audio, language):
        self.calls += 1
        return {"text": "hello", "avg_logprob": -0.1, "compression_ratio": 1.0}


class VAD:
    def __init__(self, revision="1"):
        self.identity = {"backend": "audit-fake-vad", "revision": revision}

    def stream(self, blocks):
        length = sum(len(block) for block in blocks)
        yield 0, length, False


class Diarizer:
    def __init__(self, revision="1"):
        self.identity = {"backend": "audit-fake-diarizer", "revision": revision,
                         "activity_threshold": 0.5}

    def stream(self, blocks):
        position = 0
        for block in blocks:
            values = np.zeros(((len(block) + 159) // 160, 8), dtype=np.float32)
            values[:, 0] = 0.9
            yield {"start_sample": position, "frame_samples": 160, "probabilities": values}
            position += len(block)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    return path


def records(database):
    with sqlite3.connect(database) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT json FROM records ORDER BY seq")]


@pytest.fixture
def audio(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe are required")

    def write(name="sample.wav", amplitude=0.2):
        target = tmp_path / name
        pcm = (amplitude * 32767 * np.sin(2 * np.pi * 330 * np.arange(4000) / 16000)).astype(
            "<i2"
        )
        with wave.open(str(target), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(pcm.tobytes())
        return target

    return write


def synthetic_run(tmp_path, path, **fields):
    manifest = write_jsonl(tmp_path / "manifest.jsonl", [{
        "id": "clip", "audio": str(path), "language": "en", "kind": "synthetic",
        "reference_text": "hello", **fields,
    }])
    return manifest, tmp_path / "scores.sqlite"


@pytest.mark.parametrize("changed", ["vad", "diarization"])
def test_completed_prepare_refuses_different_model(tmp_path, audio, changed):
    manifest = write_jsonl(tmp_path / "episode.jsonl", [{
        "id": "episode", "audio": str(audio()), "language": "en", "kind": "podcast",
    }])
    output = tmp_path / "prepared.jsonl"
    assert prepare_stream(manifest, output, VAD, Diarizer) == 1
    published = output.read_bytes()
    def vad():
        return VAD("2" if changed == "vad" else "1")

    def diarizer():
        return Diarizer("2" if changed == "diarization" else "1")

    with pytest.raises(ValueError, match="changed|identity|model"):
        prepare_stream(manifest, output, vad, diarizer)
    assert output.read_bytes() == published


def test_retry_errors_retries_base_decode_and_removes_persisted_error(tmp_path, audio, monkeypatch):
    manifest, database = synthetic_run(tmp_path, audio())
    failures = [True]

    def transient_decode(path, start, end):
        if failures:
            failures.pop()
            raise OSError("temporary decoder failure")
        return decode(path, start, end)

    # Keep the same callable throughout so adapter provenance cannot change between runs.
    monkeypatch.setattr(stages, "decode", transient_decode)
    asr = ASR()
    first = pipeline.score(manifest, database, asr)
    assert first["errors"] == 1
    assert asr.calls == 0
    assert "temporary decoder failure" in records(database)[0]["error"]
    unchanged = pipeline.score(manifest, database, asr)
    assert unchanged["errors"] == 1
    assert asr.calls == 0
    retried = pipeline.score(manifest, database, asr, retry_errors=True)
    assert retried == {"scored": 1, "resumed": 0, "errors": 0}
    assert asr.calls == 1
    row = records(database)[0]
    assert "error" not in row
    assert row["reference_cer"] == 0
    assert row["stages"]["asr"]["status"] == "ok"
    assert pipeline.score(manifest, database, asr)["resumed"] == 1
    assert asr.calls == 1


def test_select_rejects_reference_changed_after_scoring(tmp_path, audio):
    reference = audio("reference.wav", amplitude=0.1)
    manifest, database = synthetic_run(tmp_path, audio(), reference_audio=str(reference))
    pipeline.score(manifest, database, ASR())
    original = records(database)[0]["reference_sha256"]
    audio("reference.wav", amplitude=0.4)
    assert sha256(reference) != original
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"languages": {"en": {"reference_cer": {"max": 0}}}}))
    output = tmp_path / "selection"
    result = pipeline.select(database, policy, output)
    assert result["error"] == 1
    assert result["accept"] == 0
    assert (output / "accepted.jsonl").read_text() == ""
    decision = json.loads((output / "decisions.jsonl").read_text())
    assert "reference" in " ".join(decision["reasons"]).lower()


@pytest.mark.parametrize("changed", ["pcm_sha256", "source_sha256"])
def test_calibration_refuses_stale_audio_labels(tmp_path, audio, changed):
    manifest, database = synthetic_run(tmp_path, audio())
    pipeline.score(manifest, database, ASR())
    row = records(database)[0]
    label = {"id": row["id"], "human_label": "accept",
             "pcm_sha256": row["pcm_sha256"], "source_sha256": row["source_sha256"]}
    label[changed] = "0" * 64
    labels = write_jsonl(tmp_path / "labels.jsonl", [label])
    policy = tmp_path / "profile.json"
    policy.write_text(json.dumps({"profiles": {"asr": {"languages": {"en": {
        "synthetic": {"reference_cer": {"max": 0}},
    }}}}}))
    with pytest.raises(ValueError, match="changed|stale|hash|match|provenance"):
        calibrate(database, tmp_path / "calibration", labels=labels,
                  policy_path=policy, profile="asr")


def test_interleaved_sources_are_hashed_once_per_run_and_rehashed_after_mutation(
        tmp_path, monkeypatch):
    paths = []
    for index in range(80):
        path = tmp_path / f"source-{index}.bin"
        path.write_bytes(bytes([index]) * 128)
        paths.append(path)
    expected = {path: sha256(path) for path in paths}
    calls = []

    def count_hash(path):
        calls.append(Path(path))
        return sha256(Path(path))

    monkeypatch.setattr(pipeline, "sha256", count_hash)
    with sqlite3.connect(tmp_path / "cache.sqlite") as db:
        cache = pipeline.SourceHashes(db)
        for _ in range(4):
            for path in paths:
                assert cache(path) == expected[path]
        assert len(calls) == len(paths)
        paths[0].write_bytes(b"different length and content")
        assert cache(paths[0]) == sha256(paths[0])
        assert len(calls) == len(paths) + 1
    # A new run revalidates files rather than blindly trusting a persisted content hash.
    with sqlite3.connect(tmp_path / "cache.sqlite") as db:
        cache = pipeline.SourceHashes(db)
        assert cache(paths[1]) == expected[paths[1]]
        assert len(calls) == len(paths) + 2
