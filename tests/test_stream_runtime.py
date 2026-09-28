"""Regression cases for clipping diarization on the original frame timebase."""

import sqlite3

import numpy as np
import pytest

from podcurate_mlx.preparing import summarize


def summarize_frames(rows, begin, end):
    probabilities = np.zeros((len(rows), 8), dtype=np.float32)
    for frame, speakers in enumerate(rows):
        probabilities[frame, speakers] = 0.9
    with sqlite3.connect(":memory:") as db:
        db.execute(
            "CREATE TABLE probabilities(episode TEXT,a INTEGER,b INTEGER,step INTEGER,pcm BLOB)"
        )
        db.execute(
            "INSERT INTO probabilities VALUES(?,?,?,?,?)",
            ("episode", 0, len(rows) * 160, 160, probabilities.tobytes()),
        )
        return summarize(db, "episode", begin, end)


def test_unaligned_span_does_not_turn_sequential_speakers_into_overlap():
    result = summarize_frames([[0], [1]], 80, 240)
    assert result["diar_speakers"] == 2
    assert result["overlap_ratio"] == 0.0
    assert result["speaker_turns"] == [
        {"start": 0.0, "end": 0.005, "speaker": "speaker_0"},
        {"start": 0.005, "end": 0.01, "speaker": "speaker_1"},
    ]


def test_unaligned_overlap_counts_exact_clipped_samples():
    result = summarize_frames([[0, 1], [1]], 80, 240)
    assert result["overlap_ratio"] == pytest.approx(0.5)
    assert result["overlap_seconds"] == pytest.approx(0.005)


def test_aligned_nonoverlapping_channels_still_report_capacity():
    result = summarize_frames([[speaker] for speaker in range(8)], 0, 1280)
    assert result["diar_speakers"] == 8
    assert result["speaker_capacity_reached"] is True
    assert result["overlap_ratio"] == 0.0


def test_prepare_reports_source_capacity_in_single_speaker_spans(tmp_path, monkeypatch):
    import json

    from podcurate_mlx import preparing

    class VAD:
        identity = {"backend": "capacity-test-vad"}

        def stream(self, blocks):
            yield 0, 160, False
            yield 640, 800, False

    class Diarizer:
        identity = {"backend": "capacity-test-diar", "activity_threshold": 0.5}

        def stream(self, blocks):
            yield {
                "start_sample": 0,
                "probabilities": np.eye(8, dtype=np.float32),
                "frame_samples": 160,
            }

    audio = tmp_path / "source.wav"
    audio.write_bytes(b"Source bytes are hashed; decoding is replaced in this test.")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "episode", "audio": "source.wav", "language": "en", "kind": "podcast"})
        + "\n"
    )
    monkeypatch.setattr(preparing, "bounds", lambda *_: (0.0, 0.08))
    monkeypatch.setattr(preparing, "stream", lambda *_: iter([np.zeros(1280, np.float32)]))
    output = tmp_path / "prepared.jsonl"
    assert preparing.prepare_stream(manifest, output, VAD, Diarizer) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row["prepared_diarization"]["diar_speakers"] == 1 for row in rows)
    assert all(row["prepared_diarization"]["speaker_capacity_reached"] for row in rows)
    assert all(row["prepared_diarization"]["overlap_ratio"] == 0 for row in rows)
