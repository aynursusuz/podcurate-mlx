#!/usr/bin/env python3
"""CPU-only score/resume benchmark; neural stages are STUBS, not a model benchmark.
Run with PYTHONPATH pointing to the source tree under test. Repeated records
reference one generated WAV: this measures orchestration/IO, not unique-audio
throughput. Use a fresh --out directory per version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import resource
import sqlite3
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from podcurate_mlx import audio, pipeline, stages


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {
            key: Path(item).name if key in {"audio", "reference_audio"} else canonical(item)
            for key, item in value.items() if key != "seconds"
        }
    return value


def output_digest(database, connect):
    checksum, count = hashlib.sha256(), 0
    with connect(database) as db:
        for (encoded,) in db.execute("SELECT json FROM records ORDER BY seq"):
            normalized = json.dumps(canonical(json.loads(encoded)), sort_keys=True,
                                    ensure_ascii=False, allow_nan=False)
            checksum.update(normalized.encode() + b"\n")
            count += 1
    return {"records": count, "canonical_sha256": checksum.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=1.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.records < 1 or not np.isfinite(args.seconds) or not 0.1 <= args.seconds <= 30:
        parser.error("--records must be positive; --seconds must be within 0.1..30")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    count = round(args.seconds * 16000)
    pcm = np.round(4096 * np.sin(2 * np.pi * 220 * np.arange(count) / 16000)).astype("<i2")
    with wave.open(str(root / "fixture.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(pcm.tobytes())
    manifest = root / "manifest.jsonl"
    with manifest.open("w") as stream:
        for index in range(args.records):
            stream.write(json.dumps({
                "id": f"io-benchmark-{index:09d}", "audio": "fixture.wav",
                "language": "en", "kind": "podcast", "start": 0.0, "end": count / 16000,
                "reference_audio": "fixture.wav", "reference_start": 0.0,
                "reference_end": count / 16000,
            }) + "\n")
    counters = {}
    original_connect, original_popen = sqlite3.connect, subprocess.Popen

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            before = self.total_changes
            result = super().execute(sql, parameters)
            words = " ".join(sql.lower().split())
            if words.startswith(("insert", "replace", "update", "delete")) and (
                re.search(r"\b(?:into|update|from)\s+records\b", words)
            ):
                counters["records_write_statements"] += 1
                counters["records_changed_rows"] += self.total_changes - before
            return result

        def commit(self):
            counters["sqlite_commit_calls"] += 1
            return super().commit()

    def connect(*values, **options):
        options.setdefault("factory", Connection)
        return original_connect(*values, **options)

    def popen(command, *values, **options):
        executable = Path(command[0]).name if not isinstance(command, str) else command.split()[0]
        if executable == "ffmpeg":
            key = "ffmpeg_version_calls" if "-version" in command else "ffmpeg_decode_calls"
            counters[key] += 1
        elif executable == "ffprobe":
            counters["ffprobe_calls"] += 1
        return original_popen(command, *values, **options)

    class Stub:
        def __init__(self, name):
            self.name = name
            self.identity = {"backend": "benchmark-STUB-" + name, "version": 1}
            counters["stub_factory_calls"][name] += 1

        def __call__(self, pcm, language=None):
            counters["stub_inference_calls"][self.name] += 1
            checksum = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
            result = {"benchmark_" + self.name + "_pcm_sha256": checksum}
            if self.name == "asr":
                result["text"] = f"STUB samples={len(pcm)} language={language}"
            else:
                result["benchmark_quality_rms"] = float(
                    np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
            return result

    measurements = {}
    sqlite3.connect, subprocess.Popen = connect, popen
    try:
        for phase in ("cold", "cached_resume"):
            counters = {key: 0 for key in ("records_write_statements", "records_changed_rows",
                        "sqlite_commit_calls", "ffmpeg_version_calls",
                        "ffmpeg_decode_calls", "ffprobe_calls")}
            counters.update(stub_factory_calls={"asr": 0, "quality": 0},
                            stub_inference_calls={"asr": 0, "quality": 0})
            specs = [stages.Stage(name, lambda name=name: Stub(name))
                     for name in ("asr", "quality")]
            start = time.perf_counter()
            result = stages.score_stages(manifest, root / "scores.sqlite", specs)
            elapsed = time.perf_counter() - start
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            measurements[phase] = {"wall_seconds": elapsed, "calls": dict(counters),
                "result": result,
                "process_lifetime_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "output": output_digest(root / "scores.sqlite", original_connect)}
    finally:
        sqlite3.connect, subprocess.Popen = original_connect, original_popen
    repo = Path(stages.__file__).resolve().parents[2]
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True)
    with original_connect(root / "scores.sqlite") as db:
        run_config = json.loads(db.execute("SELECT json FROM config WHERE id=1").fetchone()[0])
    report = {"scope": "CPU orchestration/IO only; ASR and quality are deterministic STUBS",
        "fixture": {"records": args.records, "unique_audio_files": 1, "samples": count,
                    "sample_rate": 16000, "wav_sha256": digest(root / "fixture.wav"),
                    "explicit_end_and_reference_end": True},
        "provenance": {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "ffmpeg": run_config.get("ffmpeg"),
            "sqlite": sqlite3.sqlite_version,
            "git_revision": git.stdout.strip() if not git.returncode else None,
            "git_dirty": bool(dirty.stdout.strip()), "runner_sha256": digest(__file__),
            "sources_sha256": {module.__name__: digest(module.__file__)
                               for module in (audio, pipeline, stages)}},
        "measurements": measurements,
        "digest_exclusions": ["seconds fields (stage timings)",
                              "audio/reference_audio directory prefixes"],
        "rss_scope": "Cumulative process lifetime; resume memory is not isolated"}
    report["outputs_identical"] = (
        measurements["cold"]["output"] == measurements["cached_resume"]["output"])
    (root / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["outputs_identical"]:
        raise SystemExit("Cold and resumed canonical outputs differ")


if __name__ == "__main__":
    main()
