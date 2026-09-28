"""Durability, profile separation and portable export contracts without model downloads."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from test_pipeline import (
    FakeASR,
    policy,
    read_jsonl,
    read_records,
    setup_run,
    write_manifest,
)
from test_pipeline import wav_file as wav_file

from podcurate_mlx import pipeline
from podcurate_mlx.audio import sha256, stream
from podcurate_mlx.curation import calibrate, export
from podcurate_mlx.metrics import profile_decision, validate_profiles
from podcurate_mlx.preparing import prepare_stream
from podcurate_mlx.stages import Stage


class Quality:
    identity = {"fake": "quality"}

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def __call__(self, audio):
        self.calls += 1
        if self.fail:
            raise RuntimeError("temporary quality failure")
        return {"dnsmos_ovrl": 3.0}


def test_add_stage_and_retry_without_repeating_asr(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    asr = FakeASR()
    pipeline.score(manifest, database, asr)
    broken = Quality(fail=True)
    specs = [Stage("asr", lambda: asr), Stage("quality", lambda: broken)]
    assert pipeline.score(manifest, database, stages=specs)["errors"] == 1
    assert asr.calls == 1
    assert pipeline.score(manifest, database, stages=specs)["resumed"] == 1
    assert broken.calls == 1
    good = Quality()
    specs[1] = Stage("quality", lambda: good)
    assert pipeline.score(manifest, database, stages=specs, retry_errors=True)["errors"] == 0
    assert asr.calls == 1 and good.calls == 1
    assert read_records(database)[0]["dnsmos_ovrl"] == 3.0
    assert "error" not in read_records(database)[0]


def test_reference_change_refuses_resume(tmp_path, wav_file):
    reference = wav_file("reference.wav", amplitude=0.4)
    manifest, database = setup_run(tmp_path, wav_file(), reference_audio=str(reference))
    pipeline.score(manifest, database, FakeASR())
    wav_file("reference.wav", amplitude=0.6)
    with pytest.raises(ValueError, match="reference content changed"):
        pipeline.score(manifest, database, FakeASR())


def test_sample_timebase_and_disagreement(tmp_path, wav_file):
    manifest, database = setup_run(
        tmp_path, wav_file(), start_sample=160, end_sample=1600, timebase_hz=16000
    )
    pipeline.score(manifest, database, FakeASR())
    assert read_records(database)[0]["duration_s"] == pytest.approx(0.09)
    row = read_jsonl(manifest)[0]
    row["start"] = 0.2
    write_manifest(manifest, [row])
    with pytest.raises(ValueError, match="disagree"):
        list(pipeline.entries(manifest))


class StreamingVAD:
    identity = {"fake": "stream-v1"}
    calls = 0
    interrupt = False

    def stream(self, blocks):
        type(self).calls += 1
        total = 0
        for block in blocks:
            total += len(block)
        yield 0, min(total, 480000), total > 480000
        if self.interrupt:
            raise KeyboardInterrupt("simulated stop")
        if total > 480000:
            yield 480000, total, True


class StreamingDiarizer:
    identity = {"fake": "diar-v1"}

    def stream(self, blocks):
        offset = 0
        for block in blocks:
            probabilities = np.zeros(((len(block) + 159) // 160, 8), np.float32)
            probabilities[:, 0] = 0.9
            probabilities[:10, 1] = 0.8
            yield {"start_sample": offset, "frame_samples": 160, "probabilities": probabilities}
            offset += len(block)


def test_prepare_replay_matches_uninterrupted_and_keeps_state(tmp_path, wav_file):
    path = wav_file(seconds=30.2)
    manifest = write_manifest(
        tmp_path / "episodes.jsonl", [{"id": "episode", "audio": str(path), "language": "tr"}]
    )
    StreamingVAD.interrupt = True
    output = tmp_path / "replayed.jsonl"
    with pytest.raises(KeyboardInterrupt):
        prepare_stream(manifest, output, StreamingVAD, StreamingDiarizer)
    assert not output.exists()
    StreamingVAD.interrupt = False
    assert prepare_stream(manifest, output, StreamingVAD, StreamingDiarizer) == 2
    fresh = tmp_path / "fresh.jsonl"
    prepare_stream(manifest, fresh, StreamingVAD, StreamingDiarizer)
    assert output.read_bytes() == fresh.read_bytes()
    calls = StreamingVAD.calls
    prepare_stream(manifest, output, StreamingVAD, StreamingDiarizer)
    assert StreamingVAD.calls == calls
    rows = read_jsonl(output)
    assert all(row["boundary_cut"] for row in rows)
    assert rows[0]["prepared_diarization"]["diar_speakers"] == 2
    assert rows[0]["prepared_diarization"]["overlap_ratio"] > 0


def profile_policy():
    asr = {"reference_cer": {"max": 0.1}}
    tts = {
        **asr,
        "overlap_ratio": {"max": 0},
        "speaker_consistency": {"min": 0.2},
        "speaker_similarity": {"min": 0.7},
        "dnsmos_ovrl": {"min": 2},
    }
    return {
        "profiles": {
            p: {"languages": {"en": {"synthetic": b}}} for p, b in (("tts", tts), ("asr", asr))
        }
    }


def scored_row():
    return {
        "language": "en",
        "kind": "synthetic",
        "text": "hello",
        "reference_cer": 0,
        "input": {"reference_text": "hello"},
        "alignment_status": "ok",
        "diar_speakers": 1,
        "overlap_ratio": 0,
        "speaker_consistency": 0.8,
        "speaker_similarity": 0.9,
        "speaker_reference_available": True,
        "speaker_status": "ok",
        "dnsmos_ovrl": 3,
        "stages": {
            stage: {"status": "ok"}
            for stage in ("asr", "alignment_qwen", "quality", "diarization", "speaker")
        },
    }


def test_profile_requirements_overlap_noise_wrong_reference_and_cut_words():
    policy = profile_policy()
    validate_profiles(policy)
    row = scored_row()
    assert profile_decision(row, policy, "tts")[0] == "accept"
    row.update(diar_speakers=2, overlap_ratio=0.2, dnsmos_ovrl=1, speaker_similarity=0.1)
    assert profile_decision(row, policy, "tts")[0] == "reject"
    assert profile_decision(row, policy, "asr")[0] == "accept"
    row["boundary_cut"] = True
    assert profile_decision(row, policy, "asr")[0] == "review"
    row = scored_row()
    del row["stages"]["speaker"]
    assert profile_decision(row, policy, "tts")[0] == "review"
    row = scored_row()
    row.update(speaker_reference_available=False, speaker_similarity=None)
    # Reference-only bound should be omitted for data without references.
    del policy["profiles"]["tts"]["languages"]["en"]["synthetic"]["speaker_similarity"]
    assert profile_decision(row, policy, "tts")[0] == "accept"


def test_export_preserves_training_rate_and_transitive_groups(tmp_path):
    rows = []
    for i in range(4):
        path = tmp_path / f"source{i}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.writeframes((np.sin(np.arange(4410) * 0.1) * (1000 + i)).astype("<i2").tobytes())
        rows.append(
            {
                "id": str(i),
                "audio": str(path),
                "language": "en",
                "kind": "synthetic",
                "reference_text": "hello",
            }
        )
    rows[0]["source_id"] = rows[1]["source_id"] = "same-podcast"
    rows[1]["speaker_id"] = rows[2]["speaker_id"] = "known-speaker"
    rows[3]["reference_audio"] = rows[2]["audio"]
    manifest = write_manifest(tmp_path / "manifest.jsonl", rows)
    database = tmp_path / "scores.sqlite"
    pipeline.score(manifest, database, FakeASR())
    selection = tmp_path / "selection"
    assert pipeline.select(database, policy(tmp_path), selection)["accept"] == 4
    out = tmp_path / "export"
    counts = export(selection, out, sample_rate="preserve")
    records = read_jsonl(out / "metadata.jsonl")
    assert sum(counts.values()) == 4
    assert len({r["split"] for r in records}) == len({r["group_id"] for r in records}) == 1
    assert all(r["sample_rate"] == 44100 and r["num_samples"] == 4410 for r in records)
    assert all(not Path(r["audio"]).is_absolute() and (out / r["audio"]).is_file() for r in records)
    digest = sha256(out / "metadata.jsonl")
    assert export(selection, out, sample_rate="preserve") == counts
    assert sha256(out / "metadata.jsonl") == digest
    with pytest.raises(ValueError, match="identity changed"):
        export(selection, out, sample_rate=24000)
    (out / records[0]["audio"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="file changed"):
        export(selection, out, sample_rate="preserve")


def test_calibration_is_stratified_repeatable_and_evaluates_labels(tmp_path, wav_file):
    manifest, database = setup_run(tmp_path, wav_file())
    pipeline.score(manifest, database, FakeASR())
    for name in ("first", "second"):
        calibrate(database, tmp_path / name, seed="test", per_group=1)
    assert (tmp_path / "first/review.jsonl").read_bytes() == (
        tmp_path / "second/review.jsonl"
    ).read_bytes()
    assert json.loads((tmp_path / "first/report.json").read_text())["thresholds_validated"] is False
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({**read_records(database)[0], "human_label": "accept"}) + "\n")
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(profile_policy()))
    report = calibrate(
        database, tmp_path / "evaluated", labels=labels, policy_path=p, profile="asr"
    )
    assert report["evaluation"]["en/synthetic"]["review_or_error"] == 1


def test_audio_stream_blocks_are_bounded(tmp_path, wav_file):
    path = wav_file(seconds=30.2)
    chunks = list(stream(path, block_samples=997))
    assert max(map(len, chunks)) == 997
    assert sum(map(len, chunks)) == 483200
