"""Exact input reuse and durable cached work, with actual FFmpeg/SQLite."""

import sqlite3

from test_pipeline import FakeASR, read_records, setup_run, write_manifest
from test_pipeline import wav_file as wav_file
from test_workflow import Quality, StreamingDiarizer, StreamingVAD

from podcurate_mlx import pipeline, stages
from podcurate_mlx.audio import decode, signals
from podcurate_mlx.preparing import prepare_stream
from podcurate_mlx.stages import Stage


def test_new_asr_reuses_exact_analysis_audio_and_cached_rows_are_not_rewritten(
    tmp_path, wav_file, monkeypatch,
):
    audio = wav_file()
    manifest, database = setup_run(
        tmp_path, audio, start=0, end=0.1, reference_audio=str(audio),
        reference_start=0, reference_end=0.1,
    )
    decoded, statements = [], []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    def tracked_decode(path, start, end):
        result = decode(path, start, end)
        decoded.append(signals(result)["pcm_sha256"])
        return result

    def unexpected_probe(path):
        raise AssertionError("Explicit source/reference boundaries must not probe duration")

    class ObservingASR(FakeASR):
        def __call__(self, data, language):
            assert signals(data)["pcm_sha256"] == decoded[0]
            return super().__call__(data, language)

    monkeypatch.setattr(stages.sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(stages, "decode", tracked_decode)
    monkeypatch.setattr(stages, "duration", unexpected_probe)
    asr, quality = ObservingASR(), Quality()
    specs = [Stage("asr", lambda: asr), Stage("quality", lambda: quality)]
    assert pipeline.score(manifest, database, stages=specs)["errors"] == 0
    assert len(decoded) == 2  # One signals+ASR decode and one quality decode.
    assert len([s for s in statements if s.startswith("INSERT OR REPLACE INTO records")]) == 2
    saved = read_records(database)
    assert saved[0]["pcm_sha256"] == decoded[0] == decoded[1]
    statements.clear()
    assert pipeline.score(manifest, database, stages=specs) == {
        "scored": 0, "resumed": 1, "errors": 0,
    }
    assert read_records(database) == saved
    assert len(decoded) == 2 and asr.calls == quality.calls == 1
    assert not any(s.startswith("INSERT OR REPLACE INTO records") for s in statements)


def test_prepared_diarization_reuse_does_not_decode_again(tmp_path, wav_file, monkeypatch):
    source = write_manifest(tmp_path / "episode.jsonl", [
        {"id": "episode", "audio": str(wav_file()), "language": "en"},
    ])
    prepared = tmp_path / "prepared.jsonl"
    prepare_stream(source, prepared, StreamingVAD, StreamingDiarizer)
    calls = []

    def tracked_decode(path, start, end):
        calls.append((path, start, end))
        return decode(path, start, end)

    # Decoder identity must stay the real module's identity for prepared provenance.
    monkeypatch.setattr(stages, "decode", tracked_decode)
    getfile = stages.inspect.getfile
    monkeypatch.setattr(
        stages.inspect, "getfile", lambda obj: getfile(decode if obj is tracked_decode else obj),
    )
    database = tmp_path / "scores.sqlite"
    result = pipeline.score(prepared, database, stages=[
        Stage("asr", FakeASR), Stage("diarization", StreamingDiarizer),
    ])
    assert result["errors"] == 0
    assert len(calls) == 1
    row = read_records(database)[0]
    assert row["stages"]["diarization"]["status"] == "ok"
    assert row["diar_speakers"] == 2
