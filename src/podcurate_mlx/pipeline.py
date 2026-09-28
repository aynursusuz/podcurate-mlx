"""Streaming manifest processing; SQLite commits one completed record at a time."""

import inspect
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path

from .audio import MAX_SECONDS, decode, duration, sha256
from .metrics import LANGUAGES, decision, validate_policy


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def entries(path: Path):
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {number}: expected an object")
            for key in ("id", "audio", "language"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"line {number}: {key} must be a nonempty string")
            if row["language"] not in LANGUAGES:
                raise ValueError(f"line {number}: language must be en, zh, ja or tr")
            kind = row.get("kind", "podcast")
            if kind not in {"podcast", "synthetic"}:
                raise ValueError(f"line {number}: kind must be podcast or synthetic")
            if kind == "synthetic" and not row.get("reference_text"):
                raise ValueError(f"line {number}: synthetic audio needs reference_text")
            if "reference_text" in row and not isinstance(row["reference_text"], str):
                raise ValueError(f"line {number}: reference_text must be a string")
            if "boundary_cut" in row and not isinstance(row["boundary_cut"], bool):
                raise ValueError(f"line {number}: boundary_cut must be boolean")
            for key in ("start", "end", "reference_start", "reference_end"):
                if key in row and (
                    isinstance(row[key], bool)
                    or not isinstance(row[key], (float, int))
                    or not math.isfinite(row[key])
                ):
                    raise ValueError(f"line {number}: {key} must be finite")
            for key in ("source_id", "speaker_id", "reference_audio"):
                if key in row and (not isinstance(row[key], str) or not row[key]):
                    raise ValueError(f"line {number}: {key} must be a nonempty string")
            sample_keys = {"start_sample", "end_sample", "timebase_hz"}
            if sample_keys & row.keys():
                if (
                    not sample_keys <= row.keys()
                    or any(
                        isinstance(row[k], bool) or not isinstance(row[k], int) for k in sample_keys
                    )
                    or not (0 <= row["start_sample"] < row["end_sample"] and row["timebase_hz"] > 0)
                ):
                    raise ValueError(f"line {number}: invalid sample boundaries/timebase")
                for key in ("start", "end"):
                    if (
                        key in row
                        and abs(row[key] - row[key + "_sample"] / row["timebase_hz"]) > 1e-8
                    ):
                        raise ValueError("second and sample boundaries disagree")
            yield row


def resolve_audio(row: dict, manifest: Path) -> Path:
    return (manifest.parent / row["audio"]).resolve(strict=True)


class SourceHashes:
    """Disk-backed invocation cache; changed stat invalidates a file digest."""

    def __init__(self, db):
        self.db = db
        db.execute(
            "CREATE TEMP TABLE IF NOT EXISTS sources_cache "
            "(path TEXT PRIMARY KEY,size INTEGER,mtime INTEGER,hash TEXT)"
        )

    def __call__(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        saved = self.db.execute(
            "SELECT hash FROM sources_cache WHERE path=? AND size=? AND mtime=?", key
        ).fetchone()
        if saved:
            return saved[0]
        digest = sha256(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("source changed while hashing")
        self.db.execute("INSERT OR REPLACE INTO sources_cache VALUES(?,?,?,?)", (*key, digest))
        return digest


def score(
    manifest: Path, db_path: Path, asr=None, quality=None, *, stages=None, retry_errors=False
) -> dict:
    from .stages import Stage, score_stages

    if stages is None:
        stages = [Stage("asr", lambda: asr)]
        if quality is not None:
            stages.append(Stage("quality", lambda: quality))
    return score_stages(manifest, db_path, stages, retry_errors=retry_errors)


def select(db_path: Path, policy_path: Path, out: Path, profile=None) -> dict:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    from .metrics import profile_decision, validate_profiles

    if profile:
        validate_profiles(policy)
    else:
        validate_policy(policy)
    # A fresh directory makes outputs and their policy an immutable selection snapshot.
    out.mkdir(parents=True, exist_ok=False)
    counts = {"accept": 0, "reject": 0, "review": 0, "error": 0}
    try:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA temp_store=FILE")
            source_hash = SourceHashes(db)
            if db.execute("SELECT complete FROM state").fetchone() != (1,):
                raise ValueError("scoring run is incomplete; resume score before select")
            db.execute("CREATE TEMP TABLE accepted_pcm (hash TEXT PRIMARY KEY, id TEXT)")
            config = db.execute("SELECT json FROM config WHERE id=1").fetchone()
            with (
                (out / "decisions.jsonl").open("w", encoding="utf-8") as audit,
                (out / "accepted.jsonl").open("w", encoding="utf-8") as accepted,
            ):
                for (encoded,) in db.execute("SELECT json FROM records ORDER BY seq"):
                    row = json.loads(encoded)
                    if not row.get("error"):
                        try:
                            if source_hash(Path(row["audio"])) != row["source_sha256"]:
                                row["error"] = "source_changed_since_scoring"
                            if row.get("reference_audio") and source_hash(
                                Path(row["reference_audio"])
                            ) != row.get("reference_sha256"):
                                row["error"] = "reference_changed_since_scoring"
                        except OSError:
                            row["error"] = "source_unavailable_since_scoring"
                    status, reasons = (
                        profile_decision(row, policy, profile) if profile else decision(row, policy)
                    )
                    if status == "accept":
                        prior = db.execute(
                            "SELECT id FROM accepted_pcm WHERE hash=?", (row["pcm_sha256"],)
                        ).fetchone()
                        if prior:
                            status, reasons = "reject", [f"exact_pcm_duplicate:{prior[0]}"]
                        else:
                            db.execute(
                                "INSERT INTO accepted_pcm VALUES (?, ?)",
                                (row["pcm_sha256"], row["id"]),
                            )
                    row.update(status=status, reasons=reasons)
                    audit.write(dump(row) + "\n")
                    if status == "accept":
                        accepted.write(dump(row) + "\n")
                    counts[status] += 1
            (out / "summary.json").write_text(
                dump(
                    {
                        "counts": counts,
                        "policy": policy,
                        "run": json.loads(config[0]),
                        "profile": profile,
                        "stages": dict(db.execute("SELECT name,identity FROM stages")),
                    }
                )
                + "\n"
            )
            (out / "_SUCCESS").touch()
    except Exception:
        (out / "_FAILED").touch()
        raise
    return counts


def prepare(manifest: Path, output: Path, vad, *, diarizer_factory=None) -> int:
    """VAD proposals from <=30s blocks, not speaker-pure or word-aligned TTS segments."""
    if callable(vad) and not hasattr(vad, "identity") or hasattr(vad, "stream"):
        from .preparing import prepare_stream

        factory = (lambda: vad) if hasattr(vad, "identity") else vad
        return prepare_stream(manifest, output, factory, diarizer_factory)
    manifest = manifest.resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fd, temporary = tempfile.mkstemp(prefix=".prepare-", suffix=".jsonl", dir=output.parent)
    os.close(fd)
    try:
        count = _prepare(manifest, Path(temporary), vad)
        os.link(temporary, output)  # Atomic creation; fails if output appeared meanwhile.
        return count
    finally:
        Path(temporary).unlink(missing_ok=True)


def _prepare(manifest: Path, output: Path, vad) -> int:
    count = 0
    audio_identity = sha256(Path(inspect.getfile(decode)))
    with output.open("w", encoding="utf-8") as f:
        for row in entries(manifest):
            if row.get("kind") == "synthetic" or row.get("reference_text"):
                raise ValueError("prepare requires untranscribed podcast episodes")
            path = resolve_audio(row, manifest)
            length = duration(path)
            begin, stop = row.get("start", 0.0), row.get("end", length)
            if not 0 <= begin < stop <= length + 0.1:
                raise ValueError("episode span is outside the source")
            block_start = begin
            while block_start < stop - 1e-6:
                block_end = min(block_start + MAX_SECONDS, stop)
                audio = decode(path, block_start, block_end)
                for start, end in vad(audio):
                    if not 0 <= start < end <= len(audio) / 16000 + 0.001:
                        raise ValueError("VAD returned invalid timestamps")
                    end = min(end, len(audio) / 16000)
                    absolute_start, absolute_end = block_start + start, block_start + end
                    cut = (block_start > begin and start < 0.1) or (
                        block_end < stop and end > block_end - block_start - 0.1
                    )
                    record = {
                        "id": f"{row['id']}:{round(absolute_start * 16000)}:"
                        f"{round(absolute_end * 16000)}",
                        "audio": str(path),
                        "language": row["language"],
                        "kind": "podcast",
                        "start": absolute_start,
                        "end": absolute_end,
                        "boundary_cut": cut,
                        "source_id": row["id"],
                        "vad": vad.identity,
                        "analysis_audio_adapter_sha256": audio_identity,
                    }
                    f.write(dump(record) + "\n")
                    count += 1
                block_start = block_end
    return count
