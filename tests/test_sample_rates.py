"""Real resampling/provenance checks; fixture bounds are not quality calibration."""

import json
import subprocess
import sys
import wave

import numpy as np
import pytest
from test_pipeline import FakeASR, policy, read_jsonl, read_records, setup_run, write_manifest
from test_workflow import StreamingDiarizer, StreamingVAD, profile_policy, scored_row

from podcurate_mlx import pipeline
from podcurate_mlx.audio import decode, info, sha256, stream
from podcurate_mlx.curation import export
from podcurate_mlx.metrics import profile_decision, validate_profiles
from podcurate_mlx.preparing import prepare_stream
from podcurate_mlx.stages import Stage


def write_audio(path, rate, channels=1, frequency=440):
    samples = (np.sin(2 * np.pi * frequency * np.arange(rate // 10) / rate) * 8000)
    pcm = np.repeat(samples[:, None], channels, axis=1).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())
    return path


@pytest.mark.parametrize("source_rate,target_rate", [
    (8000, 24000), (48000, 24000), (16000, 16000), (44100, "preserve"),
])
def test_rate_provenance_conversion_warnings_and_resume(
    tmp_path, capsys, source_rate, target_rate,
):
    audio = write_audio(tmp_path / "audio.wav", source_rate)
    source_digest = sha256(audio)
    manifest, database = setup_run(tmp_path, audio)
    asr = FakeASR()
    pipeline.score(manifest, database, asr)
    scored = read_records(database)[0]
    assert scored["source_sample_rate"] == source_rate
    assert scored["analysis_sample_rate"] == 16000
    assert bool(capsys.readouterr().err) == (source_rate != 16000)
    assert pipeline.score(manifest, database, asr)["resumed"] == 1
    assert asr.calls == 1
    capsys.readouterr()
    selection = tmp_path / "selected"
    pipeline.select(database, policy(tmp_path), selection)
    out = tmp_path / "export"
    counts = export(selection, out, sample_rate=target_rate)
    rate = source_rate if target_rate == "preserve" else target_rate
    first_stderr = capsys.readouterr().err
    assert bool(first_stderr) == (source_rate != rate)
    metadata = read_jsonl(out / "metadata.jsonl")[0]
    assert metadata["source_sample_rate"] == source_rate
    assert metadata["analysis_sample_rate"] == metadata["source_timebase_hz"] == 16000
    assert metadata["sample_rate"] == rate
    assert metadata["num_samples"] == rate // 10
    assert int(info(out / metadata["audio"])["sample_rate"]) == rate
    assert bool(metadata["sample_rate_warnings"]) == (source_rate != rate)
    if source_rate < rate:
        assert "does not restore" in first_stderr
    if source_rate > rate:
        assert f"below {rate / 2:g} Hz" in first_stderr
    saved = [(out / name).read_bytes() for name in ("metadata.jsonl", "run.json")]
    assert export(selection, out, sample_rate=target_rate) == counts
    assert capsys.readouterr().err == first_stderr
    assert saved == [(out / name).read_bytes() for name in ("metadata.jsonl", "run.json")]
    assert sha256(audio) == source_digest


def test_source_rate_filter_uses_file_not_manifest_or_analysis_rate(tmp_path):
    audio = write_audio(tmp_path / "telephone.wav", 8000)
    manifest, database = setup_run(tmp_path, audio, source_sample_rate=48000)
    pipeline.score(manifest, database, FakeASR())
    assert read_records(database)[0]["source_sample_rate"] == 8000
    limits = tmp_path / "policy.json"
    limits.write_text(json.dumps({"languages": {"en": {"source_sample_rate": {"min": 16000}}}}))
    out = tmp_path / "selected"
    assert pipeline.select(database, limits, out)["reject"] == 1
    assert read_jsonl(out / "decisions.jsonl")[0]["reasons"] == [
        "source_sample_rate:outside_bounds",
    ]
    assert not read_jsonl(out / "accepted.jsonl")


@pytest.mark.parametrize("profile", ["tts", "asr"])
def test_source_rate_is_an_optional_explicit_profile_bound(profile):
    row, limits = scored_row(), profile_policy()
    assert profile_decision(row, limits, profile)[0] == "accept"
    limits["profiles"][profile]["languages"]["en"]["synthetic"]["source_sample_rate"] = {
        "min": 24000,
    }
    validate_profiles(limits)
    for value, expected in [
        (None, "review"), (True, "review"), (8000, "reject"), (48000, "accept"),
    ]:
        row["source_sample_rate"] = value
        assert profile_decision(row, limits, profile)[0] == expected


def test_mixed_preserve_cli_warns_without_corrupting_json_and_keeps_resume_audit(tmp_path):
    rows = [
        {"id": str(rate), "audio": str(write_audio(tmp_path / f"{rate}.wav", rate, frequency=hz)),
         "language": "en", "reference_text": "hello"}
        for rate, hz in [(8000, 440), (48000, 660)]
    ]
    manifest = write_manifest(tmp_path / "manifest.jsonl", rows)
    database, selection, out = tmp_path / "scores.sqlite", tmp_path / "selected", tmp_path / "out"
    pipeline.score(manifest, database, FakeASR())
    assert pipeline.select(database, policy(tmp_path), selection)["accept"] == 2
    command = [sys.executable, "-m", "podcurate_mlx.cli", "export", str(selection),
               "--out", str(out), "--sample-rate", "preserve"]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    assert sum(json.loads(first.stdout).values()) == 2
    assert "Mixed output sample rates: 8000 Hz, 48000 Hz" in first.stderr
    report = json.loads((out / "run.json").read_text())
    assert len(report["sample_rate_conversions"]) == 2
    assert all(item["records"] == 1 for item in report["sample_rate_conversions"])
    assert report["sample_rate_warnings"]
    saved = (out / "run.json").read_bytes()
    resumed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert (resumed.stdout, resumed.stderr) == (first.stdout, first.stderr)
    assert saved == (out / "run.json").read_bytes()


def test_probe_analysis_and_export_use_the_same_first_audio_stream(tmp_path):
    first = write_audio(tmp_path / "first.wav", 48000)
    second = write_audio(tmp_path / "second.wav", 8000, channels=2, frequency=660)
    source = tmp_path / "multiple.mka"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(first), "-i", str(second),
         "-map", "0:a:0", "-map", "1:a:0", "-c:a", "pcm_s16le", str(source)],
        capture_output=True, check=True,
    )
    expected = decode(first, 0, 0.1)
    np.testing.assert_array_equal(decode(source, 0, 0.1), expected)
    np.testing.assert_array_equal(np.concatenate(list(stream(source, end=0.1))), expected)
    manifest, database = setup_run(tmp_path, source)
    pipeline.score(manifest, database, FakeASR())
    assert read_records(database)[0]["source_sample_rate"] == 48000
    selection, out = tmp_path / "selected", tmp_path / "out"
    pipeline.select(database, policy(tmp_path), selection)
    export(selection, out, sample_rate="preserve")
    metadata = read_jsonl(out / "metadata.jsonl")[0]
    assert metadata["channels"] == 1
    assert metadata["sample_rate"] == metadata["source_sample_rate"] == 48000
    np.testing.assert_allclose(decode(out / metadata["audio"], 0, 0.1), expected, atol=1e-6)

    # Legacy scoring used automatic stream selection, which can pick the second stream.
    # Without provenance, do not export the newly selected first stream against old scores.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    row = read_jsonl(selection / "accepted.jsonl")[0]
    del row["source_sample_rate"]
    write_manifest(legacy / "accepted.jsonl", [row])
    (legacy / "_SUCCESS").touch()
    with pytest.raises(ValueError, match="rescore and select"):
        export(legacy, tmp_path / "legacy-export", sample_rate="preserve")
    assert not list((tmp_path / "legacy-export").rglob("*.flac"))


@pytest.mark.parametrize("with_diarization", [False, True])
def test_prepared_audio_identity_prevents_reusing_old_stream_results(tmp_path, with_diarization):
    audio = write_audio(tmp_path / "audio.wav", 48000)
    manifest = write_manifest(tmp_path / "episodes.jsonl", [
        {"id": "episode", "audio": str(audio), "language": "en"},
    ])
    prepared = tmp_path / "prepared.jsonl"
    prepare_stream(
        manifest, prepared, StreamingVAD, StreamingDiarizer if with_diarization else None,
    )
    asr = FakeASR()
    stages = [Stage("asr", lambda: asr)]
    if with_diarization:
        stages.append(Stage("diarization", StreamingDiarizer))
    assert pipeline.score(prepared, tmp_path / "fresh.sqlite", stages=stages)["errors"] == 0
    assert asr.calls == 1
    rows = read_jsonl(prepared)
    del rows[0]["analysis_audio_adapter_sha256"]
    legacy = write_manifest(tmp_path / "legacy.jsonl", rows)
    assert pipeline.score(legacy, tmp_path / "legacy.sqlite", stages=stages)["errors"] == 1
    assert "run prepare again" in read_records(tmp_path / "legacy.sqlite")[0]["error"]
    assert asr.calls == 1
