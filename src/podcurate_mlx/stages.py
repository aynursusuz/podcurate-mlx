"""One model at a time, durable per-stage results, fail-closed provenance checks."""

import gc
import hashlib
import inspect
import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, decode, duration, info, resampling_warning, sha256, signals
from .metrics import agreement, normalize


@dataclass
class Stage:
    name: str
    factory: object
    languages: tuple = ("en", "zh", "ja", "tr")


def bounds(row, path):
    if "start_sample" in row:
        return row["start_sample"] / row["timebase_hz"], row["end_sample"] / row["timebase_hz"]
    return row.get("start", 0.0), row.get("end", duration(path))


def model_identity(model):
    identity = dict(model.identity)
    module = inspect.getmodule(type(model))
    path = getattr(module, "__file__", None)
    # Adapter source is scoped to this stage, never the whole package.
    if path and "podcurate_mlx" in str(path):
        identity["adapter_sha256"] = sha256(Path(path))
    identity["agreement_sha256"] = (
        hashlib.sha256(
            (inspect.getsource(agreement) + inspect.getsource(normalize)).encode()
        ).hexdigest()
        if (type(model).__name__ == "WhisperASR")
        else None
    )
    if type(model).__name__ == "WhisperASR":
        identity["rapidfuzz_version"] = version("rapidfuzz")
    return identity


def release(model):
    if hasattr(model, "close"):
        model.close()
    gc.collect()


def score_stages(manifest, db_path, stages, retry_errors=False):
    from .pipeline import SourceHashes, dump, entries, resolve_audio

    manifest = manifest.resolve(strict=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": 2,
        "manifest_sha256": sha256(manifest),
        "manifest_path": str(manifest),
        "audio_adapter_sha256": sha256(Path(inspect.getfile(decode))),
        "analysis_sample_rate": 16000,
        "orchestration_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__,
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
    }
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS state(complete INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS base(seq INTEGER PRIMARY KEY, id TEXT UNIQUE, json TEXT);
            CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY, id TEXT UNIQUE,
                                              source_hash TEXT, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY, identity TEXT);
            CREATE TABLE IF NOT EXISTS results(id TEXT, stage TEXT, json TEXT,
                                               PRIMARY KEY(id, stage));
            CREATE TABLE IF NOT EXISTS references_cache(key TEXT PRIMARY KEY, embedding BLOB);
            CREATE TEMP TABLE seen(id TEXT PRIMARY KEY);
            CREATE TEMP TABLE changed(id TEXT PRIMARY KEY);
            CREATE TEMP TABLE source_rates(path TEXT, hash TEXT, rate INTEGER,
                                           PRIMARY KEY(path, hash));
        """)
        warned_rates = set()
        source_hash = SourceHashes(db)
        saved = db.execute("SELECT json FROM config WHERE id=1").fetchone()
        if saved and saved[0] != dump(config):
            raise ValueError("run identity changed; use a new --out database")
        db.execute("INSERT OR IGNORE INTO config VALUES(1,?)", (dump(config),))
        db.execute("DELETE FROM state")
        db.execute("INSERT INTO state VALUES(0)")
        db.commit()

        def publish(seq, row):
            merged = dict(row)
            stage_info, errors = {}, []
            for name, encoded in db.execute(
                "SELECT stage,json FROM results WHERE id=?", (row["id"],)
            ):
                result = json.loads(encoded)
                stage_info[name] = result["provenance"]
                if result.get("error"):
                    errors.append(f"{name}: {result['error']}")
                else:
                    merged.update(result["metrics"])
            if errors:
                merged["error"] = "; ".join(errors)
            merged["stages"] = stage_info
            db.execute(
                "INSERT OR REPLACE INTO records VALUES(?,?,?,?)",
                (seq, row["id"], row.get("source_sha256"), dump(merged)),
            )

        def infer(model, name, seq, row):
            prior = db.execute(
                "SELECT json FROM results WHERE id=? AND stage=?", (row["id"], name)
            ).fetchone()
            if prior and not (retry_errors and json.loads(prior[0]).get("error")):
                return
            if row.get("error") or row.get("silent"):
                return
            start = time.monotonic()
            try:
                if source_hash(Path(row["audio"])) != row["source_sha256"]:
                    raise ValueError("source changed during scoring")
                if (
                    row.get("reference_audio")
                    and source_hash(Path(row["reference_audio"])) != row["reference_sha256"]
                ):
                    raise ValueError("reference changed during scoring")
                audio = decode(Path(row["audio"]), row["start"], row["end"])
                if name == "asr":
                    metrics = model(audio, row["language"])
                    if row["input"].get("reference_text"):
                        metrics.update(
                            agreement(
                                row["input"]["reference_text"], metrics["text"], row["language"]
                            )
                        )
                elif name.startswith("alignment"):
                    merged = json.loads(
                        db.execute("SELECT json FROM records WHERE id=?", (row["id"],)).fetchone()[
                            0
                        ]
                    )
                    text = row["input"].get("reference_text") or merged.get("text")
                    if not text:
                        raise ValueError("alignment requires reference_text or successful ASR")
                    metrics = model(audio, text, row["language"])
                    metrics["alignment_text_source"] = (
                        "reference" if row["input"].get("reference_text") else "asr"
                    )
                elif name == "diarization" and row["input"].get("prepared_diarization"):
                    prior_identity = row["input"].get("diarization_identity")
                    if prior_identity != model_identity(model):
                        raise ValueError("prepared diarization model identity differs")
                    if row["input"].get("source_sha256") != row["source_sha256"]:
                        raise ValueError("prepared source hash differs")
                    metrics = row["input"]["prepared_diarization"]
                elif name == "speaker":
                    embedding = None
                    if row.get("reference_audio"):
                        a, b = row["reference_start"], row["reference_end"]
                        key = dump([row["reference_sha256"], a, b, model.identity])
                        cached = db.execute(
                            "SELECT embedding FROM references_cache WHERE key=?", (key,)
                        ).fetchone()
                        if cached:
                            embedding = np.frombuffer(cached[0], dtype=np.float32).copy()
                        else:
                            embedding = np.asarray(
                                model.embed(decode(Path(row["reference_audio"]), a, b)),
                                dtype=np.float32,
                            ).reshape(-1)
                            db.execute(
                                "INSERT INTO references_cache VALUES(?,?)",
                                (key, embedding.tobytes()),
                            )
                    metrics = model(audio, reference_embedding=embedding)
                else:
                    metrics = model(audio)
                result = {"metrics": metrics}
                dump(result)  # Reject NaN before persisting any apparently valid metric.
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
            result["provenance"] = {
                "status": "error" if "error" in result else "ok",
                "seconds": time.monotonic() - start,
            }
            db.execute(
                "INSERT OR REPLACE INTO results VALUES(?,?,?)", (row["id"], name, dump(result))
            )
            db.execute("INSERT OR IGNORE INTO changed VALUES(?)", (row["id"],))
            publish(seq, row)
            db.commit()

        def register(spec, model):
            identity = dump(model_identity(model))
            saved = db.execute("SELECT identity FROM stages WHERE name=?", (spec.name,)).fetchone()
            if saved and saved[0] != identity:
                raise ValueError(f"run identity changed for stage {spec.name}; use a new database")
            db.execute("INSERT OR IGNORE INTO stages VALUES(?,?)", (spec.name, identity))
            db.commit()

        if not stages or stages[0].name != "asr":
            raise ValueError("the first stage must be asr (cached successful results are reused)")
        model = stages[0].factory()
        try:
            register(stages[0], model)
            for seq, item in enumerate(entries(manifest)):
                try:
                    db.execute("INSERT INTO seen VALUES(?)", (item["id"],))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate manifest id: {item['id']}") from exc
                saved = db.execute("SELECT json FROM base WHERE id=?", (item["id"],)).fetchone()
                saved = json.loads(saved[0]) if saved else None
                row = {
                    "id": item["id"],
                    "input": item,
                    "language": item["language"],
                    "kind": item.get("kind", "podcast"),
                    "boundary_cut": item.get("boundary_cut", False),
                }
                try:
                    if (
                        {"vad", "prepared_diarization", "analysis_audio_adapter_sha256"}
                        & item.keys()
                        and item.get("analysis_audio_adapter_sha256")
                        != config["audio_adapter_sha256"]
                    ):
                        raise ValueError("prepared audio adapter differs; run prepare again")
                    path = resolve_audio(item, manifest)
                    digest = source_hash(path)
                    row.update(audio=str(path), source_sha256=digest)
                    if saved and (
                        (
                            saved.get("source_sha256") is not None
                            and saved["source_sha256"] != digest
                        )
                        or (saved.get("audio") is not None and saved["audio"] != str(path))
                    ):
                        raise RuntimeError(f"source content changed: {item['id']}; use a new run")
                    if item.get("reference_audio"):
                        ref = (manifest.parent / item["reference_audio"]).resolve(strict=True)
                        a = item.get("reference_start", 0.0)
                        b = item.get("reference_end", min(duration(ref), a + 30.0))
                        row.update(
                            reference_audio=str(ref),
                            reference_sha256=source_hash(ref),
                            reference_start=a,
                            reference_end=b,
                        )
                        if (
                            saved
                            and saved.get("reference_sha256") is not None
                            and saved["reference_sha256"] != row["reference_sha256"]
                        ):
                            raise RuntimeError(f"reference content changed: {item['id']}")
                    if saved and not (retry_errors and saved.get("error")):
                        row = saved
                    else:
                        a, b = bounds(item, path)
                        cached_rate = db.execute(
                            "SELECT rate FROM source_rates WHERE path=? AND hash=?",
                            (str(path), digest),
                        ).fetchone()
                        rate = cached_rate[0] if cached_rate else int(info(path)["sample_rate"])
                        if rate <= 0:
                            raise ValueError("source sample rate must be positive")
                        db.execute(
                            "INSERT OR IGNORE INTO source_rates VALUES(?,?,?)",
                            (str(path), digest, rate),
                        )
                        row.update(source_sample_rate=rate, analysis_sample_rate=SAMPLE_RATE)
                        row.update(start=a, end=b, **signals(decode(path, a, b)))
                        if row["silent"]:
                            row["text"] = ""
                    rate = row.get("source_sample_rate")
                    notice = resampling_warning(rate, SAMPLE_RATE) if rate is not None else None
                    if notice and rate not in warned_rates:
                        print(
                            f"warning: analysis: {notice} Source files are unchanged.",
                            file=sys.stderr,
                        )
                        warned_rates.add(rate)
                except RuntimeError:
                    raise
                except Exception as exc:
                    if saved and not (retry_errors and saved.get("error")):
                        raise ValueError(f"source became unavailable: {item['id']}") from exc
                    row["error"] = f"{type(exc).__name__}: {exc}"
                if not saved or (retry_errors and saved.get("error")):
                    db.execute(
                        "INSERT OR REPLACE INTO base VALUES(?,?,?)", (seq, item["id"], dump(row))
                    )
                    db.execute("INSERT OR IGNORE INTO changed VALUES(?)", (item["id"],))
                infer(model, "asr", seq, row)
                publish(seq, row)
                db.commit()
        except RuntimeError as exc:
            if "changed" in str(exc):
                raise ValueError(str(exc)) from exc
            raise
        finally:
            release(model)
            del model
        for spec in stages[1:]:
            if not db.execute(
                "SELECT 1 FROM base WHERE json_extract(json,'$.language') IN ("
                + ",".join("?" for _ in spec.languages)
                + ") LIMIT 1",
                spec.languages,
            ).fetchone():
                continue
            model = spec.factory()
            try:
                register(spec, model)
                for seq, encoded in db.execute("SELECT seq,json FROM base ORDER BY seq"):
                    row = json.loads(encoded)
                    if row["language"] in spec.languages:
                        infer(model, spec.name, seq, row)
            finally:
                release(model)
                del model
        db.execute("UPDATE state SET complete=1")
        db.commit()
        scored = db.execute("SELECT count(*) FROM changed").fetchone()[0]
        total = db.execute("SELECT count(*) FROM records").fetchone()[0]
        errors = db.execute(
            "SELECT count(*) FROM records WHERE json_extract(json,'$.error') IS NOT NULL"
        ).fetchone()[0]
        return {"scored": scored, "resumed": total - scored, "errors": errors}
