"""Streaming manifest processing; SQLite commits one completed record at a time."""

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .audio import MAX_SECONDS, decode, duration, sha256, signals
from .metrics import LANGUAGES, agreement, decision, validate_policy


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
            for key in ("start", "end"):
                if key in row and (isinstance(row[key], bool)
                                   or not isinstance(row[key], (float, int))
                                   or not math.isfinite(row[key])):
                    raise ValueError(f"line {number}: {key} must be finite")
            yield row


def resolve_audio(row: dict, manifest: Path) -> Path:
    return (manifest.parent / row["audio"]).resolve(strict=True)


@lru_cache(maxsize=64)
def _source_hash(path: str, size: int, mtime: int) -> str:
    return sha256(Path(path))


def source_hash(path: Path) -> str:
    stat = path.stat()
    return _source_hash(str(path), stat.st_size, stat.st_mtime_ns)


def package_versions() -> dict:
    packages = {}
    for name in ("mlx-whisper", "mlx", "numpy", "rapidfuzz", "onnxruntime"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return packages


def implementation_identity() -> dict:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    return {"python_sources_sha256": digest.hexdigest(),
            "ffmpeg": ffmpeg.stdout.splitlines()[0]}


def score(manifest: Path, db_path: Path, asr, quality=None) -> dict:
    _source_hash.cache_clear()
    manifest = manifest.resolve(strict=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"manifest_sha256": sha256(manifest), "manifest_path": str(manifest),
              "asr": asr.identity,
              "quality": quality.identity if quality else None,
              "package_version": __version__, "dependencies": package_versions(),
              "implementation": implementation_identity()}
    counts = {"scored": 0, "resumed": 0, "errors": 0}
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, json TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS records "
                   "(seq INTEGER PRIMARY KEY, id TEXT UNIQUE, "
                   "source_hash TEXT, json TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS state (complete INTEGER NOT NULL)")
        saved = db.execute("SELECT json FROM config WHERE id=1").fetchone()
        if saved and saved[0] != dump(config):
            raise ValueError("run identity changed; use a new --out database")
        db.execute("INSERT OR IGNORE INTO config VALUES (1, ?)", (dump(config),))
        db.execute("DELETE FROM state")
        db.execute("INSERT INTO state VALUES (0)")
        db.execute("CREATE TEMP TABLE seen (id TEXT PRIMARY KEY)")
        db.commit()
        for seq, row in enumerate(entries(manifest)):
            try:
                db.execute("INSERT INTO seen VALUES (?)", (row["id"],))
            except sqlite3.IntegrityError as e:
                raise ValueError(f"duplicate manifest id: {row['id']}") from e
            saved = db.execute("SELECT source_hash, json FROM records WHERE id=?",
                               (row["id"],)).fetchone()
            path, file_hash = None, None
            try:
                path = resolve_audio(row, manifest)
                file_hash = source_hash(path)
            except (OSError, ValueError) as e:
                if saved:
                    raise ValueError(f"source became unavailable: {row['id']}") from e
                error = f"{type(e).__name__}: {e}"
            if saved:
                if saved[0] != file_hash:
                    raise ValueError(f"source content changed: {row['id']}; use a new run")
                saved_json = json.loads(saved[1])
                if saved_json.get("audio") and saved_json["audio"] != str(path):
                    raise ValueError(f"source location changed: {row['id']}; use a new run")
                counts["resumed"] += 1
                if saved_json.get("error"):
                    counts["errors"] += 1
                continue
            result = {"id": row["id"], "input": row, "language": row["language"],
                      "boundary_cut": row.get("boundary_cut", False),
                      "source_sha256": file_hash}
            try:
                if path is None:
                    raise ValueError(error)
                start = row.get("start", 0.0)
                end = row.get("end")
                if end is None:
                    end = duration(path)
                audio = decode(path, start, end)
                result.update(audio=str(path), start=start, end=end, **signals(audio))
                if result["silent"]:
                    result["text"] = ""
                else:
                    result.update(asr(audio, row["language"]))
                    if row.get("reference_text"):
                        result.update(agreement(row["reference_text"], result["text"],
                                                row["language"]))
                    if quality:
                        result.update(quality(audio))
                # Non-finite model scores cannot become valid JSON or accepted records.
                serialized = dump(result)
            except Exception as e:
                result = {"id": row["id"], "input": row, "language": row["language"],
                          "source_sha256": file_hash, "error": f"{type(e).__name__}: {e}"}
                serialized = dump(result)
                counts["errors"] += 1
            db.execute("INSERT INTO records VALUES (?, ?, ?, ?)",
                       (seq, row["id"], file_hash, serialized))
            db.commit()
            counts["scored"] += 1
            if counts["scored"] % 100 == 0:
                print(dump(counts), file=sys.stderr, flush=True)
        db.execute("UPDATE state SET complete=1")
        db.commit()
    return counts


def select(db_path: Path, policy_path: Path, out: Path) -> dict:
    _source_hash.cache_clear()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy(policy)
    # A fresh directory makes outputs and their policy an immutable selection snapshot.
    out.mkdir(parents=True, exist_ok=False)
    counts = {"accept": 0, "reject": 0, "review": 0, "error": 0}
    try:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA temp_store=FILE")
            if db.execute("SELECT complete FROM state").fetchone() != (1,):
                raise ValueError("scoring run is incomplete; resume score before select")
            db.execute("CREATE TEMP TABLE accepted_pcm (hash TEXT PRIMARY KEY, id TEXT)")
            config = db.execute("SELECT json FROM config WHERE id=1").fetchone()
            with (out / "decisions.jsonl").open("w", encoding="utf-8") as audit, \
                 (out / "accepted.jsonl").open("w", encoding="utf-8") as accepted:
                for (encoded,) in db.execute("SELECT json FROM records ORDER BY seq"):
                    row = json.loads(encoded)
                    if not row.get("error"):
                        try:
                            if source_hash(Path(row["audio"])) != row["source_sha256"]:
                                row["error"] = "source_changed_since_scoring"
                        except OSError:
                            row["error"] = "source_unavailable_since_scoring"
                    status, reasons = decision(row, policy)
                    if status == "accept":
                        prior = db.execute("SELECT id FROM accepted_pcm WHERE hash=?",
                                           (row["pcm_sha256"],)).fetchone()
                        if prior:
                            status, reasons = "reject", [f"exact_pcm_duplicate:{prior[0]}"]
                        else:
                            db.execute("INSERT INTO accepted_pcm VALUES (?, ?)",
                                       (row["pcm_sha256"], row["id"]))
                    row.update(status=status, reasons=reasons)
                    audit.write(dump(row) + "\n")
                    if status == "accept":
                        accepted.write(dump(row) + "\n")
                    counts[status] += 1
            (out / "summary.json").write_text(dump({"counts": counts, "policy": policy,
                                                    "run": json.loads(config[0])}) + "\n")
            (out / "_SUCCESS").touch()
    except Exception:
        (out / "_FAILED").touch()
        raise
    return counts


def prepare(manifest: Path, output: Path, vad) -> int:
    """VAD proposals from <=30s blocks, not speaker-pure or word-aligned TTS segments."""
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
                    cut = ((block_start > begin and start < 0.1)
                           or (block_end < stop and end > block_end - block_start - 0.1))
                    record = {
                        "id": f"{row['id']}:{round(absolute_start*16000)}:"
                              f"{round(absolute_end*16000)}",
                        "audio": str(path), "language": row["language"], "kind": "podcast",
                        "start": absolute_start, "end": absolute_end, "boundary_cut": cut,
                        "source_id": row["id"], "vad": vad.identity,
                    }
                    f.write(dump(record) + "\n")
                    count += 1
                block_start = block_end
    return count
