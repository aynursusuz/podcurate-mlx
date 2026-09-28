"""Real FLAC validation is durable; unchanged resume avoids regrouping and decode."""

import json
import sqlite3

import pytest
from test_pipeline import FakeASR, policy, read_jsonl, write_manifest
from test_pipeline import wav_file as wav_file

from podcurate_mlx import curation, pipeline


@pytest.fixture
def selected(tmp_path, wav_file):
    shared = wav_file("shared.wav", seconds=0.3, amplitude=0.2)
    other = wav_file("other.wav", amplitude=0.4)
    manifest = write_manifest(tmp_path / "manifest.jsonl", [
        {"id": "first", "audio": str(shared), "language": "en", "reference_text": "hello",
         "start": 0, "end": 0.1, "source_id": "episode"},
        {"id": "second", "audio": str(other), "language": "en", "reference_text": "hello",
         "source_id": "episode"},
        {"id": "third", "audio": str(shared), "language": "en", "reference_text": "hello",
         "start": 0.013, "end": 0.113},
    ])
    database = tmp_path / "score.sqlite"
    pipeline.score(manifest, database, FakeASR())
    selection = tmp_path / "selection"
    assert pipeline.select(database, policy(tmp_path), selection)["accept"] == 3
    return selection, shared


def forbidden(*args, **kwargs):
    pytest.fail("unchanged completed export repeated grouping or full FLAC verification")


def test_completed_resume_keeps_bytes_without_grouping_or_flac_decode(
    tmp_path, selected, monkeypatch,
):
    selection, _ = selected
    out = tmp_path / "export"
    original_verify = curation._verify_flac
    verified = []

    def verify(*args):
        verified.append(args[0])
        return original_verify(*args)

    monkeypatch.setattr(curation, "_verify_flac", verify)
    counts = curation.export(selection, out, sample_rate=24000)
    assert len(verified) == 3
    rows = read_jsonl(out / "metadata.jsonl")
    assert len({row["group_id"] for row in rows}) == 1
    paths = [out / name for name in ("metadata.jsonl", "run.json")]
    paths += [out / row["audio"] for row in rows]
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(curation, "_root", forbidden)
    monkeypatch.setattr(curation, "_verify_flac", forbidden)
    monkeypatch.setattr(curation, "info", forbidden)
    assert curation.export(selection, out, sample_rate=24000) == counts
    assert before == {path: path.read_bytes() for path in paths}


def test_interrupted_export_resumes_only_unfinished_outputs(tmp_path, selected, monkeypatch):
    selection, _ = selected
    out = tmp_path / "export"
    original_verify = curation._verify_flac
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_verify(*args)

    monkeypatch.setattr(curation, "_verify_flac", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        curation.export(selection, out, sample_rate=24000)
    with sqlite3.connect(out / "export.sqlite") as db:
        assert db.execute("SELECT count(*) FROM items").fetchone()[0] == 3
        assert db.execute("SELECT count(*) FROM outputs").fetchone()[0] == 1
    original_root = curation._root
    roots, verified = [], []

    def root(*args):
        roots.append(args[1])
        return original_root(*args)

    def verify(*args):
        verified.append(args[0])
        return original_verify(*args)

    monkeypatch.setattr(curation, "_root", root)
    monkeypatch.setattr(curation, "_verify_flac", verify)
    assert sum(curation.export(selection, out, sample_rate=24000).values()) == 3
    assert len(roots) == len(verified) == 2
    reference = tmp_path / "uninterrupted"
    curation.export(selection, reference, sample_rate=24000)
    assert (out / "metadata.jsonl").read_bytes() == (reference / "metadata.jsonl").read_bytes()


@pytest.mark.parametrize("change", ["flac", "source"])
def test_completed_resume_refuses_changed_files(tmp_path, selected, monkeypatch, change):
    selection, source = selected
    out = tmp_path / "export"
    curation.export(selection, out, sample_rate=24000)
    if change == "flac":
        target = out / read_jsonl(out / "metadata.jsonl")[0]["audio"]
        target.write_bytes(b"corrupt")
        message = "completed export file changed"
    else:
        source.write_bytes(source.read_bytes() + b"changed")
        message = "source changed before export"
    monkeypatch.setattr(curation, "_verify_flac", forbidden)
    with pytest.raises(ValueError, match=message):
        curation.export(selection, out, sample_rate=24000)


@pytest.mark.parametrize("field,value", [
    ("sample_rate", 16000), ("num_samples", 50000), ("channels", 0),
    ("source_sample_rate", None), ("source_sample_rate", "24000"),
    ("source_sample_rate", 8000),
])
def test_completed_resume_refuses_invalid_cached_metadata(
    tmp_path, selected, monkeypatch, field, value,
):
    selection, _ = selected
    out = tmp_path / "export"
    curation.export(selection, out, sample_rate=24000)
    with sqlite3.connect(out / "export.sqlite") as db:
        metadata = json.loads(db.execute("SELECT json FROM outputs WHERE id='first'").fetchone()[0])
        metadata[field] = value
        db.execute("UPDATE outputs SET json=? WHERE id='first'", (json.dumps(metadata),))
    monkeypatch.setattr(curation, "_verify_flac", forbidden)
    with pytest.raises(ValueError, match="completed export metadata changed|source sample rate"):
        curation.export(selection, out, sample_rate=24000)


def test_verifier_change_refuses_cached_outputs(tmp_path, selected, monkeypatch):
    selection, _ = selected
    out = tmp_path / "export"
    curation.export(selection, out, sample_rate=24000)
    identity = {**curation._verifier_identity(), "ffmpeg": "changed verifier"}
    monkeypatch.setattr(curation, "_verifier_identity", lambda: identity)
    monkeypatch.setattr(curation, "_verify_flac", forbidden)
    with pytest.raises(ValueError, match="export identity changed"):
        curation.export(selection, out, sample_rate=24000)


def test_source_cache_rechecks_shared_file_during_export(tmp_path, selected, monkeypatch):
    selection, source = selected
    out = tmp_path / "export"
    original_verify = curation._verify_flac
    changed = False

    def verify(*args):
        nonlocal changed
        result = original_verify(*args)
        if not changed:
            source.write_bytes(source.read_bytes() + b"changed after first interval")
            changed = True
        return result

    monkeypatch.setattr(curation, "_verify_flac", verify)
    with pytest.raises(ValueError, match="source changed before export: third"):
        curation.export(selection, out, sample_rate=24000)
    assert not (out / "_SUCCESS").exists()
