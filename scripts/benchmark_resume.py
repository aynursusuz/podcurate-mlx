#!/usr/bin/env python3
"""Bounded cached-resume benchmark, explicitly using preseeded STUB results.

One generated, non-speech WAV is shared by all manifest IDs. This measures real
score_stages cache traversal/source checks; it is NOT million-audio inference.
Use PYTHONPATH to select the source tree and a fresh --out directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from podcurate_mlx import audio, pipeline, stages


class Stub:
    calls = {"asr": 0, "quality": 0}
    forbid_inference = False

    def __init__(self, name):
        self.name = name
        self.identity = {"backend": "preseeded-CPU-STUB-" + name, "version": 1}

    def __call__(self, pcm, language=None):
        self.calls[self.name] += 1
        if self.forbid_inference:
            raise SystemExit("Unexpected inference during cached resume; benchmark aborted")
        checksum = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
        if self.name == "asr":
            return {"text": "STUB generated tone", "benchmark_asr_pcm_sha256": checksum}
        return {"benchmark_energy": float(np.mean(pcm.astype(np.float64) ** 2)),
                "benchmark_quality_pcm_sha256": checksum}


def specifications():
    return [stages.Stage(name, lambda name=name: Stub(name)) for name in Stub.calls]


def provenance():
    repo = Path(stages.__file__).resolve().parents[2]
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "sqlite": sqlite3.sqlite_version,
            "git_revision": git.stdout.strip() if git.returncode == 0 else None,
            "runner_sha256": audio.sha256(Path(__file__)),
            "sources_sha256": {m.__name__: audio.sha256(Path(m.__file__))
                               for m in (audio, pipeline, stages)}}


def disk_guard(root, limit):
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used >= limit * 0.9 or shutil.disk_usage(root).free < 1_000_000_000:
        raise SystemExit(f"Disk guard stopped benchmark: {used} bytes in {root}")
    return used


def record_digest(db):
    checksum, count = hashlib.sha256(), 0
    for (encoded,) in db.execute("SELECT json FROM records ORDER BY seq"):
        checksum.update(encoded.encode() + b"\n")
        count += 1
    return {"records": count, "sha256": checksum.hexdigest()}


def seed(root, count, limit):
    start = time.perf_counter()
    samples = 4000
    pcm = np.round(4096 * np.sin(2 * np.pi * 220 * np.arange(samples) / 16000)).astype("<i2")
    with wave.open(str(root / "fixture.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())
    item = {"id": "cache-0000000", "audio": "fixture.wav", "language": "en",
            "kind": "podcast", "start": 0.0, "end": samples / 16000}
    manifest, database = root / "manifest.jsonl", root / "scores.sqlite"
    manifest.write_text(pipeline.dump(item) + "\n")
    bootstrap = stages.score_stages(manifest, database, specifications())
    if bootstrap["errors"]:
        raise SystemExit("Real one-record bootstrap failed")
    checksum = hashlib.sha256()
    with sqlite3.connect(database) as db:
        config = json.loads(db.execute("SELECT json FROM config").fetchone()[0])
        base = json.loads(db.execute("SELECT json FROM base").fetchone()[0])
        record = json.loads(db.execute("SELECT json FROM records").fetchone()[0])
        results = []
        for name, encoded in db.execute("SELECT stage,json FROM results ORDER BY stage"):
            value = json.loads(encoded)
            value["provenance"] = {"status": "ok", "seconds": 0.0, "benchmark_preseeded": True}
            record["stages"][name] = value["provenance"]
            results.append((name, pipeline.dump(value)))
        for table in ("base", "records", "results"):
            db.execute(f"DELETE FROM {table}")
        db.execute("UPDATE state SET complete=0")
        db.commit()
        with manifest.open("w") as stream:
            for index in range(count):
                item["id"] = f"cache-{index:07d}"
                base.update(id=item["id"], input=item)
                record.update(id=item["id"], input=item)
                encoded = pipeline.dump(record)
                stream.write(pipeline.dump(item) + "\n")
                checksum.update(encoded.encode() + b"\n")
                db.execute("INSERT INTO base VALUES(?,?,?)",
                           (index, item["id"], pipeline.dump(base)))
                db.execute("INSERT INTO records VALUES(?,?,?,?)",
                           (index, item["id"], base["source_sha256"], encoded))
                db.executemany("INSERT INTO results VALUES(?,?,?)",
                               ((item["id"], name, value) for name, value in results))
                if (index + 1) % 1000 == 0:
                    db.commit()
                    disk_guard(root, limit)
                if (index + 1) % 100000 == 0:
                    print(json.dumps({"seeded": index + 1}), flush=True)
        config["manifest_sha256"] = audio.sha256(manifest)
        db.execute("UPDATE config SET json=?", (pipeline.dump(config),))
        db.execute("UPDATE state SET complete=1")
        db.commit()
    report = {"records": count, "unique_audio_files": 1, "samples_per_file": samples,
              "scope": "Synthetic cached rows copied from deterministic STUB bootstrap outputs",
              "bootstrap": bootstrap, "bootstrap_stub_calls": Stub.calls,
              "records_sha256": checksum.hexdigest(), "seed_seconds": time.perf_counter() - start,
              "disk_bytes": disk_guard(root, limit), "provenance": provenance()}
    (root / "seed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seed_complete": True, "seconds": report["seed_seconds"],
                      "disk_bytes": report["disk_bytes"]}), flush=True)


def measure(root, limit):
    seed_info = json.loads((root / "seed.json").read_text())
    expected_sources = seed_info["provenance"]["sources_sha256"]
    if provenance()["sources_sha256"] != expected_sources:
        raise SystemExit("Source tree changed since seeding; use a fresh benchmark")
    counters = {"source_check_calls": 0, "source_hash_computations": 0,
                "ffmpeg_decode_calls": 0, "ffprobe_calls": 0, "ffmpeg_version_calls": 0,
                "records_changed_rows": 0}
    original_check, original_hash = pipeline.SourceHashes.__call__, pipeline.sha256
    original_popen, original_connect = subprocess.Popen, sqlite3.connect

    def check(owner, path):
        counters["source_check_calls"] += 1
        if counters["source_check_calls"] % 10000 == 0:
            disk_guard(root, limit)
        if counters["source_check_calls"] % 100000 == 0:
            print(json.dumps({"source_checks": counters["source_check_calls"]}), flush=True)
        return original_check(owner, path)

    def source_hash(path):
        counters["source_hash_computations"] += 1
        return original_hash(path)

    def popen(command, *args, **kwargs):
        name = Path(command[0]).name
        if name == "ffmpeg":
            key = "ffmpeg_version_calls" if "-version" in command else "ffmpeg_decode_calls"
            counters[key] += 1
        elif name == "ffprobe":
            counters["ffprobe_calls"] += 1
        return original_popen(command, *args, **kwargs)

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            write = "records" in sql and sql.lstrip().lower().startswith(
                ("insert", "replace", "update", "delete"))
            before = self.total_changes if write else 0
            result = super().execute(sql, parameters)
            if write:
                counters["records_changed_rows"] += self.total_changes - before
            return result

    def connect(*args, **kwargs):
        kwargs.setdefault("factory", Connection)
        return original_connect(*args, **kwargs)

    Stub.forbid_inference = True
    pipeline.SourceHashes.__call__, pipeline.sha256 = check, source_hash
    subprocess.Popen, sqlite3.connect = popen, connect
    start = time.perf_counter()
    try:
        result = stages.score_stages(
            root / "manifest.jsonl", root / "scores.sqlite", specifications())
        elapsed = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        pipeline.SourceHashes.__call__, pipeline.sha256 = original_check, original_hash
        subprocess.Popen, sqlite3.connect = original_popen, original_connect
    verify_start = time.perf_counter()
    with sqlite3.connect(root / "scores.sqlite") as db:
        actual = record_digest(db)
        counts = {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                  for name in ("base", "records", "results")}
    checks = {"all_resumed": result == {"scored": 0, "resumed": seed_info["records"], "errors": 0},
              "records_unchanged": actual == {"records": seed_info["records"],
                                               "sha256": seed_info["records_sha256"]},
              "no_stub_inference": not any(Stub.calls.values()),
              "no_decode_or_records_write": not any(counters[key] for key in
                  ("ffmpeg_decode_calls", "ffprobe_calls", "records_changed_rows"))}
    report = {"scope": "Preseeded cached resume; NOT million-file hashing or speech inference",
              "wall_seconds": elapsed, "verification_seconds": time.perf_counter() - verify_start,
              "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
              "rss_scope": "Fresh measurement child process; excludes seeding",
              "result": result, "counters": counters, "stub_inference_calls": Stub.calls,
              "table_counts": counts, "checks": checks, "records_digest": actual,
              "database_bytes": (root / "scores.sqlite").stat().st_size,
              "disk_bytes": disk_guard(root, limit), "provenance": provenance(),
              "instrumentation": "Wall includes call counters and periodic disk-budget checks"}
    (root / "resume.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit("Cached-resume verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1000000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--disk-budget-gb", type=float, default=6.0)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--measure-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.records < 1 or not 0 < args.disk_budget_gb <= 6:
        parser.error("Positive records and a disk budget within 0..6 GB are required")
    if args.wait_for:
        while not args.wait_for.is_file():
            time.sleep(1)
    root, limit = args.out.resolve(), args.disk_budget_gb * 1_000_000_000
    root.mkdir(parents=True, exist_ok=args.measure_only)
    (root / "tmp").mkdir(exist_ok=True)
    os.environ["SQLITE_TMPDIR"] = str(root / "tmp")
    if args.measure_only:
        measure(root, limit)
    else:
        seed(root, args.records, limit)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--measure-only",
                        "--out", str(root), "--disk-budget-gb", str(args.disk_budget_gb)],
                       check=True)


if __name__ == "__main__":
    main()
