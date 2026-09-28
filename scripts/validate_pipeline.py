#!/usr/bin/env python3
"""Opt-in real-model transport validation on a small, local four-language cohort.

Requires installed project extras, FFmpeg, and complete pinned model snapshots in
--hf-cache. This never creates human labels or treats its permissive test policy
as calibrated. FLEURS natural read speech is not relabeled as synthetic speech.
"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

STAGES = "asr,quality,diarization,speaker,alignment_qwen,alignment_tr"
LANGUAGES = ("en", "zh", "ja", "tr")
DISCLAIMER = (
    "TRANSPORT-TEST-ONLY. Permissive bounds exercise software paths; they are not "
    "production thresholds, human quality judgments, speaker verification thresholds, "
    "or evidence of downstream training quality. FLEURS is natural read speech, "
    "not podcast-domain or synthetic-speech quality validation."
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def jsonl(path):
    with path.open(encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mlx = sys.modules.get("mlx.core")
    return {"ru_maxrss_raw": raw, "native_unit": "bytes" if sys.platform == "darwin" else "KiB",
            "peak_rss_bytes": int(raw if sys.platform == "darwin" else raw * 1024),
            "scope": "process lifetime high-water mark, not a per-stage delta",
            "mlx_peak_memory_bytes": int(mlx.get_peak_memory()) if mlx is not None else None,
            "mlx_memory_scope": "MLX allocator process high-water mark; excludes non-MLX "
                                "allocations and is not RSS or total physical device memory"}


def stage_snapshot(database, stage):
    with sqlite3.connect(database) as db:
        rows = db.execute("SELECT id,json FROM results WHERE stage=? ORDER BY id", (stage,))
        result = hashlib.sha256()
        count = 0
        for identifier, encoded in rows:
            result.update((identifier + "\0" + encoded + "\n").encode())
            count += 1
        return {"count": count, "serialized_results_sha256": result.hexdigest()}


def cohort(manifest, output):
    rows = []
    for original in jsonl(manifest):
        require(len(rows) < 64, "This runner accepts at most 64 validation fixtures")
        row = dict(original)
        path = (manifest.parent / row["audio"]).resolve(strict=True)
        require(row["language"] in LANGUAGES, "Unsupported fixture language")
        require(row.get("kind", "podcast") == "podcast", "Use natural-speech fixtures here")
        if row.get("sha256"):
            require(digest(path) == row["sha256"], f"Fixture checksum mismatch: {row['id']}")
        row.update(audio=str(path), kind="podcast", validation_source_type="natural_read_speech")
        row["validation_kind_note"] = "podcast schema branch; not evidence of podcast provenance"
        row.pop("reference_audio", None)
        rows.append(row)
    require({row["language"] for row in rows} == set(LANGUAGES), "All four languages are required")
    require(len({row["id"] for row in rows}) == len(rows), "Fixture IDs must be unique")
    for language in LANGUAGES:
        group = [row for row in rows if row["language"] == language]
        group[0].update(reference_audio=group[0]["audio"],
                        reference_control="same_audio_positive_control")
        if len(group) >= 3:
            group[1].update(reference_audio=group[2]["audio"],
                            reference_control="different_file_no_speaker_label")
    with output.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def transport_policy():
    profiles = {}
    for profile in ("tts", "asr"):
        languages = {}
        for language in LANGUAGES:
            bounds = {"duration_s": {"min": 0, "max": 30},
                      "reference_cer": {"min": 0, "max": 1000},
                      "avg_logprob": {"min": -1000000, "max": 1000000},
                      "compression_ratio": {"min": 0, "max": 1000000}}
            if profile == "tts":
                bounds.update(overlap_ratio={"min": 0, "max": 1},
                              speaker_consistency={"min": -1.00001, "max": 1.00001},
                              speaker_similarity={"min": -1.00001, "max": 1.00001},
                              dnsmos_ovrl={"min": -1000, "max": 1000})
            languages[language] = {"podcast": bounds}
        profiles[profile] = {"languages": languages}
    return {"profiles": profiles}


def exported_hashes(directory):
    paths = sorted(directory.rglob("*.flac")) + [directory / "metadata.jsonl"]
    return {str(path.relative_to(directory)): digest(path) for path in paths}


def inspect_results(database):
    report = {"models": {}, "stages": {}, "records": []}
    with sqlite3.connect(database) as db:
        report["models"] = {name: json.loads(encoded) for name, encoded in db.execute(
            "SELECT name,identity FROM stages ORDER BY name")}
        for name, encoded in db.execute("SELECT stage,json FROM results ORDER BY stage,id"):
            result = json.loads(encoded)
            cell = report["stages"].setdefault(name, {"ok": 0, "error": 0, "inference_seconds": 0})
            cell[result["provenance"]["status"]] += 1
            cell["inference_seconds"] += result["provenance"]["seconds"]
        keys = ("id", "language", "error", "text", "reference_cer", "reference_wer",
                "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "diar_speakers", "overlap_ratio",
                "speaker_similarity", "speaker_consistency", "speaker_status",
                "alignment_status", "alignment_reasons", "alignment_coverage", "alignment_score")
        for (encoded,) in db.execute("SELECT json FROM records ORDER BY seq"):
            row = json.loads(encoded)
            report["records"].append({**{key: row.get(key) for key in keys},
                                      "reference_control": row["input"].get("reference_control")})
    return report


def retry_control(database, target, manifest, factories, baseline, total):
    """Inject one explicit test error in an isolated backup, never in the export source."""
    from podcurate_mlx.pipeline import score

    with sqlite3.connect(database) as source, sqlite3.connect(target) as copy:
        source.backup(copy)
        identifier = copy.execute("SELECT id FROM results WHERE stage='quality' ORDER BY id LIMIT 1"
                                  ).fetchone()[0]
        injected = {"error": "VALIDATION FAULT INJECTION: transient quality failure",
                    "provenance": {"status": "error", "seconds": 0}}
        copy.execute("UPDATE results SET json=? WHERE id=? AND stage='quality'",
                     (json.dumps(injected), identifier))
    result = score(manifest, target, stages=factories(), retry_errors=True)
    require(result == {"scored": 1, "resumed": total - 1, "errors": 0},
            f"Isolated retry did not recompute exactly one record: {result}")
    require(stage_snapshot(target, "asr") == baseline, "Retry changed cached ASR results")
    return {"injected_stage": "quality", "id": identifier, "isolated_database": str(target),
            "injection_is_a_test_error_not_a_model_measurement": True, "result": result,
            "asr_results_preserved": True}


def run(args, report):
    # Set cache configuration before importing Hugging Face or model adapter modules.
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.hf_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PODCURATE_CACHE_DIR"] = str(args.out / "model-cache")
    from podcurate_mlx import __version__
    from podcurate_mlx.cli import WHISPER_MODEL, WHISPER_REVISION, default_stages
    from podcurate_mlx.curation import calibrate, export
    from podcurate_mlx.pipeline import score, select

    manifest = args.out / "fixture-manifest.jsonl"
    rows = cohort(args.fixtures, manifest)
    total = len(rows)
    report.update(fixtures={"source_manifest": str(args.fixtures),
                            "source_manifest_sha256": digest(args.fixtures), "records": total,
                            "manifest": str(manifest), "manifest_sha256": digest(manifest)},
                  phases={}, hf_cache=str(args.hf_cache), offline=True,
                  package_source_version=__version__)
    options = SimpleNamespace(model=WHISPER_MODEL, revision=WHISPER_REVISION,
                              offline=True, dnsmos_model=args.dnsmos_model, stages=STAGES)
    database = args.out / "scores.sqlite"

    def timed(name, operation):
        started = time.perf_counter()
        print(f"START {name}", flush=True)
        result = operation()
        report["phases"][name] = {"seconds": time.perf_counter() - started,
                                  "result": result, **peak_rss()}
        write_json(args.out / "validation.json", report)
        print(f"DONE {name}: {json.dumps(result, ensure_ascii=False)}", flush=True)
        return result

    timed("asr_only", lambda: score(manifest, database, stages=default_stages(options)[:1]))
    baseline = stage_snapshot(database, "asr")
    require(baseline["count"] == total, "ASR result count differs from fixture count")
    added = timed("add_remaining_stages", lambda: score(
        manifest, database, stages=default_stages(options)))
    require(stage_snapshot(database, "asr") == baseline, "Adding stages changed ASR results")
    report["asr_cache_preserved_after_stage_add"] = baseline
    report["inference"] = inspect_results(database)
    require(added["errors"] == 0, "Model scoring errors occurred; inspect inference report")
    require(set(report["inference"]["stages"]) == set(STAGES.split(",")),
            "Not all six stages produced results")
    for spec in default_stages(options):
        expected = sum(row["language"] in spec.languages for row in rows)
        require(report["inference"]["stages"][spec.name]["ok"] == expected,
                f"Stage result count differs from eligible fixture count: {spec.name}")
    resumed = timed("resume_all", lambda: score(manifest, database, stages=default_stages(options)))
    require(resumed == {"scored": 0, "resumed": total, "errors": 0},
            f"Resume unexpectedly recomputed records: {resumed}")
    require(stage_snapshot(database, "asr") == baseline, "Resume changed ASR results")
    timed("retry_isolated_fault", lambda: retry_control(
        database, args.out / "retry-control.sqlite", manifest,
        lambda: default_stages(options), baseline, total))

    timed("review_sample", lambda: calibrate(database, args.out / "review", per_group=2,
                                             seed="transport-test-2026-09-28"))
    timed("repeat_review_sample", lambda: calibrate(database, args.out / "review-repeat",
                                                    per_group=2, seed="transport-test-2026-09-28"))
    review = args.out / "review" / "review.jsonl"
    require(digest(review) == digest(args.out / "review-repeat" / "review.jsonl"),
            "Review sampling was not deterministic")
    require(all(row["human_label"] is None for row in jsonl(review)), "Human labels were generated")
    policy = args.out / "TRANSPORT-TEST-ONLY.policy.json"
    write_json(policy, transport_policy())
    (args.out / "TRANSPORT-TEST-ONLY.txt").write_text(DISCLAIMER + "\n")
    report["selections"] = {}
    total_exported = 0
    for profile in ("tts", "asr"):
        selected = args.out / f"selected-{profile}"
        counts = timed(f"select_{profile}", lambda p=profile, s=selected: select(
            database, policy, s, profile=p))
        output = args.out / f"export-{profile}"
        exported = timed(f"export_{profile}", lambda s=selected, o=output: export(
            s, o, sample_rate=24000, seed="transport-test", train=0.7, validation=0.15))
        before = exported_hashes(output)
        repeated = timed(f"resume_export_{profile}", lambda s=selected, o=output: export(
            s, o, sample_rate=24000, seed="transport-test", train=0.7, validation=0.15))
        require(exported == repeated and before == exported_hashes(output),
                "Export resume changed FLAC or metadata bytes")
        metadata = list(jsonl(output / "metadata.jsonl"))
        require(len(metadata) == counts["accept"] == sum(exported.values()),
                "Accepted/exported/metadata counts differ")
        group_splits = {}
        for item in metadata:
            require(item["sample_rate"] == 24000 and item["num_samples"] > 0,
                    "Exported metadata has invalid audio parameters")
            group_splits.setdefault(item["group_id"], set()).add(item["split"])
        require(all(len(splits) == 1 for splits in group_splits.values()), "Group split leakage")
        total_exported += len(metadata)
        report["selections"][profile] = {
            "counts": counts, "exports": exported, "sample_rate": 24000,
            "export_file_hashes": before, "export_resume_bytes_unchanged": True,
            "flac_transport_exercised": bool(metadata),
            "decisions": [{"id": row["id"], "status": row["status"], "reasons": row["reasons"]}
                          for row in jsonl(selected / "decisions.jsonl")],
        }
    require(total_exported > 0,
            "No profile accepted a fixture; actual FLAC export was not exercised")
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True, help="Local FLEURS manifest.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="New validation directory")
    parser.add_argument("--hf-cache", type=Path, required=True, help="Populated local HF hub cache")
    parser.add_argument("--dnsmos-model", type=Path, required=True, help="Pinned local P.835 ONNX")
    parser.add_argument("--lock", type=Path, help="Optional shared exclusive GPU lock file")
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve(strict=True)
    args.hf_cache = args.hf_cache.resolve(strict=True)
    args.dnsmos_model = args.dnsmos_model.resolve(strict=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": DISCLAIMER, "platform": platform.platform(),
              "python": sys.version, "versions": {name: importlib.metadata.version(name)
              for name in ("podcurate-mlx", "mlx", "mlx-audio", "mlx-whisper", "numpy")}}
    started = time.perf_counter()
    try:
        with contextlib.ExitStack() as stack:
            if args.lock:
                args.lock.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(args.lock.open("a"))
                print(f"WAIT exclusive GPU lock: {args.lock}", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
            run(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(total_seconds=time.perf_counter() - started, resources=peak_rss())
        write_json(args.out / "validation.json", report)


if __name__ == "__main__":
    main()
