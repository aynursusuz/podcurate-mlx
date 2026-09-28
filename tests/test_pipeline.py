"""Real FFmpeg/SQLite integration with fake ASR: no model/network access required."""

import json
import shutil
import sqlite3
import wave

import numpy as np
import pytest

from podcurate_mlx import pipeline
from podcurate_mlx.audio import decode, signals


class FakeASR:
    def __init__(self, results=None, identity="test-asr"):
        self.identity = {"name": identity}
        self.results = iter(results) if results is not None else None
        self.calls = 0

    def __call__(self, audio, language):
        assert audio.ndim == 1 and audio.dtype == np.float32
        assert np.isfinite(audio).all() and 0 < len(audio) <= 480000
        self.calls += 1
        result = next(self.results) if self.results is not None else {"text": "hello"}
        if isinstance(result, Exception):
            raise result
        return result


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_records(path):
    with sqlite3.connect(path) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT json FROM records ORDER BY seq")]


@pytest.fixture
def wav_file(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("real FFmpeg and ffprobe are required for audio integration")

    def write(name="sample.wav", seconds=0.1, amplitude=0.25):
        path = tmp_path / name
        samples = (np.sin(2 * np.pi * 440 * np.arange(round(seconds * 16000)) / 16000)
                   * amplitude * 32767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(samples.tobytes())
        return path
    return write


def setup_run(tmp_path, audio, **fields):
    row = {"id": "sample", "audio": str(audio), "language": "en",
           "kind": "synthetic", "reference_text": "hello", **fields}
    return write_manifest(tmp_path / "manifest.jsonl", [row]), tmp_path / "scores.sqlite"


def policy(tmp_path, metric="reference_cer", maximum=0.0):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"languages": {"en": {metric: {"max": maximum}}}}))
    return path


def test_actual_decode_score_and_resume_avoid_repeat_asr(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    asr = FakeASR()
    assert pipeline.score(manifest, database, asr) == {"scored": 1, "resumed": 0, "errors": 0}
    row = read_records(database)[0]
    assert row["reference_cer"] == 0
    assert row["duration_s"] == pytest.approx(0.1)
    assert len(row["pcm_sha256"]) == len(row["source_sha256"]) == 64
    assert pipeline.score(manifest, database, asr) == {"scored": 0, "resumed": 1, "errors": 0}
    assert asr.calls == 1


def test_each_record_committed_before_next_asr(tmp_path, wav_file):
    audio = wav_file()
    manifest = write_manifest(tmp_path / "manifest.jsonl", [
        {"id": str(i), "audio": str(audio), "language": "en"} for i in range(2)
    ])
    database = tmp_path / "scores.sqlite"

    class ObservingASR(FakeASR):
        def __call__(self, audio, language):
            with sqlite3.connect(database) as reader:
                assert reader.execute("SELECT count(*) FROM records").fetchone()[0] == self.calls
            return super().__call__(audio, language)

    assert pipeline.score(manifest, database, ObservingASR())["scored"] == 2


def test_source_changed_rejects_resume(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    pipeline.score(manifest, database, FakeASR())
    wav_file(seconds=0.2, amplitude=0.5)
    asr = FakeASR()
    with pytest.raises(ValueError, match="source content changed"):
        pipeline.score(manifest, database, asr)
    assert asr.calls == 0


def test_selection_refuses_audio_changed_after_scoring(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    pipeline.score(manifest, database, FakeASR())
    wav_file(amplitude=0.5)
    out = tmp_path / "selected"
    assert pipeline.select(database, policy(tmp_path), out)["error"] == 1
    assert read_jsonl(out / "accepted.jsonl") == []


def test_incomplete_score_cannot_be_selected(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    manifest.write_text(manifest.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate manifest id"):
        pipeline.score(manifest, database, FakeASR())
    out = tmp_path / "selected"
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.select(database, policy(tmp_path), out)
    assert (out / "_FAILED").exists()
    assert not (out / "_SUCCESS").exists()


@pytest.mark.parametrize("change", ["manifest", "asr"])
def test_run_identity_changes_require_new_database(tmp_path, wav_file, change):
    manifest, database = setup_run(tmp_path, wav_file())
    pipeline.score(manifest, database, FakeASR())
    asr = FakeASR(identity="different" if change == "asr" else "test-asr")
    if change == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="run identity changed"):
        pipeline.score(manifest, database, asr)
    assert asr.calls == 0


def test_moved_manifest_refuses_resume_with_stale_absolute_paths(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    pipeline.score(manifest, database, FakeASR())
    moved = tmp_path / "moved.jsonl"
    moved.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="run identity changed"):
        pipeline.score(moved, database, FakeASR())


def test_duplicate_manifest_ids_fail_and_keep_completed_record(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    manifest.write_text(manifest.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate manifest id"):
        pipeline.score(manifest, database, FakeASR())
    assert len(read_records(database)) == 1


def test_pcm_dedup_only_uses_previously_accepted_record(tmp_path, wav_file):
    audio = wav_file()
    manifest = write_manifest(tmp_path / "manifest.jsonl", [
        {"id": name, "audio": str(audio), "language": "en", "kind": "synthetic",
         "reference_text": "hello"} for name in ("bad", "good", "duplicate")
    ])
    database = tmp_path / "scores.sqlite"
    pipeline.score(manifest, database, FakeASR([
        {"text": "wrong"}, {"text": "hello"}, {"text": "hello"},
    ]))
    out = tmp_path / "selected"
    assert pipeline.select(database, policy(tmp_path), out) == {
        "accept": 1, "reject": 2, "review": 0, "error": 0,
    }
    assert [row["id"] for row in read_jsonl(out / "accepted.jsonl")] == ["good"]
    assert read_jsonl(out / "decisions.jsonl")[2]["reasons"] == ["exact_pcm_duplicate:good"]
    assert (out / "_SUCCESS").exists()


@pytest.mark.parametrize("result", [RuntimeError("inference failed"),
                                      {"text": "hello", "avg_logprob": float("nan")}])
def test_failed_or_nonfinite_inference_persists_error_and_never_accepts(
    tmp_path, wav_file, result,
):
    manifest, database = setup_run(tmp_path, wav_file())
    asr = FakeASR([result])
    assert pipeline.score(manifest, database, asr)["errors"] == 1
    assert "error" in read_records(database)[0]
    assert pipeline.score(manifest, database, asr)["resumed"] == 1
    assert asr.calls == 1
    out = tmp_path / "selected"
    counts = pipeline.select(database, policy(tmp_path), out)
    assert counts["error"] == 1 and counts["accept"] == 0
    assert read_jsonl(out / "accepted.jsonl") == []


def test_missing_required_model_score_is_review(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    pipeline.score(manifest, database, FakeASR())
    out = tmp_path / "selected"
    assert pipeline.select(database, policy(tmp_path, "avg_logprob", 0.0), out)["review"] == 1
    assert read_jsonl(out / "accepted.jsonl") == []


def test_prepare_marks_block_boundaries_for_review(tmp_path, wav_file):
    audio = wav_file(seconds=30.2)
    manifest = write_manifest(tmp_path / "episode.jsonl", [
        {"id": "episode", "audio": str(audio), "language": "en", "kind": "podcast"},
    ])

    class FullSpanVAD:
        identity = {"name": "fake-vad"}

        def __call__(self, audio):
            return [(0.0, len(audio) / 16000)]

    prepared = tmp_path / "segments.jsonl"
    assert pipeline.prepare(manifest, prepared, FullSpanVAD()) == 2
    proposals = read_jsonl(prepared)
    assert all(row["boundary_cut"] for row in proposals)
    assert [(row["start"], row["end"]) for row in proposals] == [(0.0, 30.0), (30.0, 30.2)]
    database = tmp_path / "scores.sqlite"
    pipeline.score(prepared, database, FakeASR())
    out = tmp_path / "selected"
    assert pipeline.select(database, policy(tmp_path, "peak", 1.0), out)["review"] == 2
    assert all("boundary_cut" in row["reasons"] for row in read_jsonl(out / "decisions.jsonl"))


def test_silent_pcm_is_not_sent_to_asr(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file(amplitude=0))
    asr = FakeASR()
    pipeline.score(manifest, database, asr)
    assert asr.calls == 0
    assert read_records(database)[0]["silent"] is True
    assert pipeline.select(database, policy(tmp_path), tmp_path / "selected")["reject"] == 1


def test_decode_refuses_unbounded_reads_and_preserves_silence_measure(tmp_path):
    with pytest.raises(ValueError, match="30 seconds"):
        decode(tmp_path / "not-opened.wav", 0, 31)
    with pytest.raises(ValueError, match="finite"):
        decode(tmp_path / "not-opened.wav", 0, float("inf"))
    assert signals(np.zeros(1600, dtype=np.float32))["rms_dbfs"] == -240.0
