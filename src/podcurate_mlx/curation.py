"""Deterministic human review and portable, group-disjoint FLAC export."""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import audio
from .audio import info, resampling_warning, sha256
from .metrics import profile_decision, validate_profiles
from .pipeline import SourceHashes, dump


def calibrate(
    database, out, *, per_group=20, seed="0", labels=None, policy_path=None, profile=None
):
    if per_group < 1:
        raise ValueError("per_group must be positive")
    out.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("SELECT complete FROM state").fetchone() != (1,):
            raise ValueError("scoring run is incomplete")
        db.execute("PRAGMA temp_store=FILE")
        db.create_function(
            "rank", 1, lambda value: hashlib.sha256((seed + "\0" + value).encode()).hexdigest()
        )
        groups = db.execute(
            "SELECT DISTINCT json_extract(json,'$.language'), "
            "COALESCE(json_extract(json,'$.kind'),'podcast') FROM records"
        ).fetchall()
        count = 0
        with (out / "review.jsonl").open("w", encoding="utf8") as f:
            for language, kind in sorted(groups):
                for (encoded,) in db.execute(
                    "SELECT json FROM records WHERE "
                    "json_extract(json,'$.language')=? AND "
                    "COALESCE(json_extract(json,'$.kind'),'podcast')=? "
                    "ORDER BY rank(id),id LIMIT ?",
                    (language, kind, per_group),
                ):
                    row = json.loads(encoded)
                    row.update(human_label=None, human_notes="")
                    f.write(dump(row) + "\n")
                    count += 1
        report = {
            "review_records": count,
            "seed": seed,
            "per_language_source": per_group,
            "thresholds_validated": False,
            "sampling": "sha256(seed + NUL + id), stratified language and source kind",
        }
        if labels is not None:
            if policy_path is None or profile is None:
                raise ValueError("labels require --policy and --profile")
            policy = json.loads(policy_path.read_text())
            validate_profiles(policy)
            db.execute("CREATE TEMP TABLE labels(id TEXT PRIMARY KEY,label TEXT)")
            stats = {}
            with labels.open(encoding="utf8") as f:
                for line in f:
                    label = json.loads(line)
                    if label.get("human_label") not in {"accept", "reject"}:
                        continue
                    try:
                        db.execute(
                            "INSERT INTO labels VALUES(?,?)", (label["id"], label["human_label"])
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate human label: {label['id']}") from exc
                    saved = db.execute(
                        "SELECT json FROM records WHERE id=?", (label["id"],)
                    ).fetchone()
                    if not saved:
                        raise ValueError(f"unknown labeled record: {label['id']}")
                    row = json.loads(saved[0])
                    for field in ("source_sha256", "pcm_sha256"):
                        if not label.get(field) or label[field] != row.get(field):
                            raise ValueError(
                                f"stale or missing label provenance: {label['id']} {field}"
                            )
                    if label.get("input") != row.get("input"):
                        raise ValueError(f"stale or missing label input metadata: {label['id']}")
                    status, _ = profile_decision(row, policy, profile)
                    group = row["language"] + "/" + row.get("kind", "podcast")
                    cell = stats.setdefault(
                        group,
                        {
                            "labels": 0,
                            "accepted": 0,
                            "false_accept": 0,
                            "false_reject": 0,
                            "review_or_error": 0,
                            "human_accept": 0,
                            "human_reject": 0,
                            "accepted_seconds": 0.0,
                        },
                    )
                    cell["labels"] += 1
                    cell["human_" + label["human_label"]] += 1
                    cell["accepted"] += status == "accept"
                    cell["false_accept"] += status == "accept" and label["human_label"] == "reject"
                    cell["false_reject"] += status == "reject" and label["human_label"] == "accept"
                    cell["review_or_error"] += status in {"review", "error"}
                    if status == "accept":
                        cell["accepted_seconds"] += row.get("duration_s", 0)
            for cell in stats.values():
                cell["accept_precision"] = (
                    (cell["accepted"] - cell["false_accept"]) / cell["accepted"]
                    if cell["accepted"]
                    else None
                )
                cell["false_accept_rate"] = (
                    cell["false_accept"] / cell["human_reject"] if cell["human_reject"] else None
                )
            report.update(
                evaluation=stats,
                profile=profile,
                policy=policy,
                labels_sha256=sha256(labels),
                caveat="Labeled sample evaluation only; no held-out quality claim.",
            )
        (out / "report.json").write_text(dump(report) + "\n", encoding="utf8")
        return report


def _root(db, key):
    db.execute("INSERT OR IGNORE INTO groups VALUES(?,?)", (key, key))
    while True:
        parent = db.execute("SELECT parent FROM groups WHERE key=?", (key,)).fetchone()[0]
        if parent == key:
            break
        grandparent = db.execute("SELECT parent FROM groups WHERE key=?", (parent,)).fetchone()[0]
        db.execute("UPDATE groups SET parent=? WHERE key=?", (grandparent, key))
        key = grandparent
    return key


def _verify_flac(path, expected_samples, rate):
    details = info(path)
    if int(details["sample_rate"]) != rate:
        raise ValueError("export sample rate mismatch")
    # Reopen and fully decode the exported <=30 s interval, checking actual frames.
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s32le", "pipe:1"],
        capture_output=True,
        check=True,
    )
    samples = len(p.stdout) // (4 * int(details["channels"]))
    if abs(samples - expected_samples) > 1:
        raise ValueError(f"export frame count mismatch: {samples} vs {expected_samples}")
    return samples, int(details["channels"])


def _verifier_identity():
    """Bind cached full-decode results to the implementation and media tools."""
    return {
        "curation_sha256": sha256(Path(__file__)),
        "audio_sha256": sha256(Path(audio.__file__)),
        **{
            tool: subprocess.run(
                [tool, "-version"], capture_output=True, text=True, check=True
            ).stdout.strip()
            for tool in ("ffmpeg", "ffprobe")
        },
    }


def _check_cached_metadata(metadata, expected, frames):
    samples = metadata.get("num_samples")
    channels = metadata.get("channels")
    if (
        any(metadata.get(key) != value for key, value in expected.items())
        or type(samples) is not int
        or abs(samples - frames) > 1
        or type(channels) is not int
        or channels < 1
    ):
        raise ValueError("completed export metadata changed")


def export(selection, out, *, sample_rate, seed="0", train=0.9, validation=0.05):
    if sample_rate != "preserve":
        try:
            sample_rate = int(sample_rate)
        except (ValueError, TypeError) as exc:
            raise ValueError("sample_rate must be preserve or an integer") from exc
        if not 8000 <= sample_rate <= 192000:
            raise ValueError("export sample rate must be 8000..192000")
    if not (0 < train <= 1 and 0 <= validation <= 1 and train + validation <= 1):
        raise ValueError("invalid split fractions")
    if not (selection / "_SUCCESS").is_file():
        raise ValueError("selection is incomplete")
    accepted = selection / "accepted.jsonl"
    config = {
        "selection_sha256": sha256(accepted),
        "sample_rate": sample_rate,
        "seed": seed,
        "train": train,
        "validation": validation,
        "algorithm": "flac-groups-v3",
        "verifier": _verifier_identity(),
    }
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out / "export.sqlite") as db:
        db.execute("PRAGMA temp_store=FILE")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS config(json TEXT);
            CREATE TABLE IF NOT EXISTS groups(key TEXT PRIMARY KEY, parent TEXT);
            CREATE TABLE IF NOT EXISTS items(seq INTEGER PRIMARY KEY,id TEXT UNIQUE,
                                            json TEXT,key TEXT);
            CREATE TABLE IF NOT EXISTS outputs(id TEXT PRIMARY KEY,json TEXT,hash TEXT);
            CREATE TEMP TABLE sources(path TEXT PRIMARY KEY,hash TEXT,rate INTEGER);
        """)
        saved = db.execute("SELECT json FROM config").fetchone()
        if saved and saved[0] != dump(config):
            raise ValueError("export identity changed; use a new output directory")
        if not saved:
            db.execute("INSERT INTO config VALUES(?)", (dump(config),))
        source_hash = SourceHashes(db)
        with accepted.open(encoding="utf8") as f:
            for seq, line in enumerate(f):
                row = json.loads(line)
                if row.get("status") != "accept":
                    raise ValueError("export accepts only accepted selection records")
                if "source_sample_rate" not in row:
                    raise ValueError(
                        "selection lacks source sample-rate provenance; rescore and select"
                    )
                prior = db.execute("SELECT id,json FROM items WHERE seq=?", (seq,)).fetchone()
                if prior:
                    if prior != (row["id"], line.strip()):
                        raise ValueError("export input rows changed")
                    # The item and all its group unions were committed together.
                    continue
                source = row.get("input", {})
                keys = ["audio:" + row["source_sha256"], "pcm:" + row["pcm_sha256"]]
                if source.get("source_id"):
                    keys.append("source:" + source["source_id"])
                if source.get("speaker_id"):
                    keys.append("speaker:" + source["speaker_id"])
                if row.get("reference_sha256"):
                    keys.append("audio:" + row["reference_sha256"])
                roots = sorted({_root(db, key) for key in keys})
                for key in roots[1:]:
                    db.execute("UPDATE groups SET parent=? WHERE key=?", (roots[0], key))
                db.execute(
                    "INSERT OR IGNORE INTO items VALUES(?,?,?,?)",
                    (seq, row["id"], line.strip(), keys[0]),
                )
                db.commit()
        counts = {"train": 0, "validation": 0, "test": 0}
        conversions = Counter()
        output_rates = Counter()
        for _seq, identifier, encoded, key in db.execute("SELECT * FROM items ORDER BY seq"):
            row = json.loads(encoded)
            saved = db.execute("SELECT json,hash FROM outputs WHERE id=?", (identifier,)).fetchone()
            metadata = json.loads(saved[0]) if saved else None
            # All input unions finish before the first output is committed, so a
            # saved output contains the final group for this immutable selection.
            group = (
                metadata["group_id"]
                if saved
                else hashlib.sha256(_root(db, key).encode()).hexdigest()
            )
            fraction = int(hashlib.sha256((seed + "\0" + group).encode()).hexdigest(), 16) / 2**256
            split = (
                "train"
                if fraction < train
                else "validation"
                if fraction < train + validation
                else "test"
            )
            counts[split] += 1
            path = Path(row["audio"])
            cached = db.execute(
                "SELECT hash,rate FROM sources WHERE path=?", (str(path),)
            ).fetchone()
            digest = source_hash(path)
            if digest != row["source_sha256"]:
                raise ValueError(f"source changed before export: {identifier}")
            if cached and cached[0] == digest:
                source_rate = cached[1]
            elif saved:
                # The source SHA is unchanged; its initially probed rate remains valid.
                source_rate = metadata.get("source_sample_rate")
            else:
                source_rate = int(info(path)["sample_rate"])
            if type(source_rate) is not int or source_rate <= 0:
                raise ValueError("source sample rate must be positive")
            if row["source_sample_rate"] != source_rate:
                raise ValueError(f"source sample rate differs from scoring: {identifier}")
            db.execute(
                "INSERT OR IGNORE INTO sources VALUES(?,?,?)", (str(path), digest, source_rate)
            )
            rate = source_rate if sample_rate == "preserve" else sample_rate
            notice = resampling_warning(source_rate, rate)
            if notice and (source_rate, rate) not in conversions:
                print(f"warning: export: {notice}", file=sys.stderr)
            conversions[source_rate, rate] += 1
            output_rates[rate] += 1
            frames = round((row["end"] - row["start"]) * rate)
            target = out / split / (hashlib.sha256(identifier.encode()).hexdigest() + ".flac")
            target.parent.mkdir(exist_ok=True)
            if saved:
                if not target.exists() or sha256(target) != saved[1]:
                    raise ValueError(f"completed export file changed: {target.name}")
                _check_cached_metadata(
                    metadata,
                    {
                        "id": identifier,
                        "audio": str(target.relative_to(out)),
                        "split": split,
                        "sample_rate": rate,
                        "source_sample_rate": source_rate,
                        "source_sha256": digest,
                    },
                    frames,
                )
                continue
            temporary = target.with_suffix(".partial.flac")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(row["start"]),
                    "-i",
                    str(path),
                    "-t",
                    str(row["end"] - row["start"]),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ar",
                    str(rate),
                    "-c:a",
                    "flac",
                    str(temporary),
                ],
                capture_output=True,
                check=True,
            )
            samples, channels = _verify_flac(temporary, frames, rate)
            os.replace(temporary, target)
            source = row.get("input", {})
            metadata = {
                "id": identifier,
                "audio": str(target.relative_to(out)),
                "split": split,
                "group_id": group,
                "language": row["language"],
                "kind": row.get("kind", "podcast"),
                "text": source.get("reference_text") or row["text"],
                "asr_text": row["text"],
                "sample_rate": rate,
                "source_sample_rate": source_rate,
                "analysis_sample_rate": row.get("analysis_sample_rate", 16000),
                "sample_rate_warnings": [notice] if notice else [],
                "num_samples": samples,
                "channels": channels,
                "source_sha256": digest,
                "source_id": source.get("source_id", digest),
                "source_start_sample": source.get("start_sample", round(row["start"] * 16000)),
                "source_end_sample": source.get("end_sample", round(row["end"] * 16000)),
                "source_timebase_hz": source.get("timebase_hz", 16000),
                "speaker_id": source.get("speaker_id"),
                "reference_source_sha256": row.get("reference_sha256"),
            }
            db.execute(
                "INSERT INTO outputs VALUES(?,?,?)", (identifier, dump(metadata), sha256(target))
            )
            db.commit()
        temporary = out / ".metadata.jsonl"
        with temporary.open("w", encoding="utf8") as f:
            for (encoded,) in db.execute(
                "SELECT outputs.json FROM outputs JOIN items USING(id) ORDER BY seq"
            ):
                f.write(encoded + "\n")
        os.replace(temporary, out / "metadata.jsonl")
        warnings = []
        if len(output_rates) > 1:
            warning = (
                "Mixed output sample rates: "
                + ", ".join(f"{rate} Hz" for rate in sorted(output_rates))
                + ". Use an explicit --sample-rate if training requires one rate."
            )
            print(f"warning: export: {warning}", file=sys.stderr)
            warnings.append(warning)
        summary = {
            **config,
            "counts": counts,
            "sample_rate_conversions": [
                {
                    "source_sample_rate": source_rate,
                    "sample_rate": rate,
                    "records": count,
                    "warning": resampling_warning(source_rate, rate),
                }
                for (source_rate, rate), count in sorted(conversions.items())
            ],
            "sample_rate_warnings": warnings,
            "unknown_speakers_disjoint_guarantee": False,
            "selection": json.loads((selection / "summary.json").read_text()),
        }
        (out / "run.json").write_text(dump(summary) + "\n", encoding="utf8")
        (out / "_SUCCESS").touch()
        return counts
