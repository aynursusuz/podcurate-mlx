"""Bounded episode preparation with deterministic replay of unfinished episodes."""

import inspect
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from .audio import sha256, stream
from .stages import bounds, model_identity, release


def prepare_stream(manifest, output, vad_factory, diarizer_factory=None):
    from .pipeline import dump, entries, resolve_audio

    manifest, output = manifest.resolve(strict=True), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    database = output.with_suffix(output.suffix + ".sqlite")
    audio_identity = sha256(Path(inspect.getfile(stream)))
    config = dump(
        {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "algorithm": "stream-replay-v1",
            "timebase_hz": 16000,
            "diarization_enabled": diarizer_factory is not None,
            "implementation": sha256(Path(__file__)),
            "audio_adapter_sha256": audio_identity,
        }
    )
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS config(json TEXT);
            CREATE TABLE IF NOT EXISTS models(name TEXT PRIMARY KEY, json TEXT);
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, hash TEXT, complete INTEGER);
            CREATE TABLE IF NOT EXISTS spans(episode TEXT, a INTEGER, b INTEGER, cut INTEGER);
            CREATE INDEX IF NOT EXISTS spans_episode_a ON spans(episode,a);
            CREATE TABLE IF NOT EXISTS probabilities(episode TEXT, a INTEGER, b INTEGER,
                 step INTEGER, pcm BLOB);
            CREATE INDEX IF NOT EXISTS probability_range ON probabilities(episode,a,b);
            CREATE TABLE IF NOT EXISTS segments(episode TEXT, a INTEGER, json TEXT,
                PRIMARY KEY(episode,a));
            CREATE TEMP TABLE seen(id TEXT PRIMARY KEY);
        """)
        saved = db.execute("SELECT json FROM config").fetchone()
        if saved and saved[0] != config:
            raise ValueError("prepare identity changed; use a new output")
        if not saved:
            db.execute("INSERT INTO config VALUES(?)", (config,))
        db.commit()

        def check_model(name, model):
            identity = dump(model_identity(model))
            saved = db.execute("SELECT json FROM models WHERE name=?", (name,)).fetchone()
            if saved and saved[0] != identity:
                raise ValueError(f"prepare {name} changed; use a new output")
            db.execute("INSERT OR IGNORE INTO models VALUES(?,?)", (name, identity))
            db.commit()

        # Validate identities even when every episode has already completed.
        for name, factory in (("vad", vad_factory), ("diarization", diarizer_factory)):
            if factory:
                model = factory()
                try:
                    check_model(name, model)
                finally:
                    release(model)
                    del model

        for row in entries(manifest):
            if row.get("kind") == "synthetic" or row.get("reference_text"):
                raise ValueError("prepare requires untranscribed podcast episodes")
            try:
                db.execute("INSERT INTO seen VALUES(?)", (row["id"],))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate manifest id: {row['id']}") from exc
            path = resolve_audio(row, manifest)
            digest = sha256(path)
            saved = db.execute(
                "SELECT hash,complete FROM episodes WHERE id=?", (row["id"],)
            ).fetchone()
            if saved and saved[0] != digest:
                raise ValueError(f"source content changed: {row['id']}")
            if saved and saved[1]:
                continue
            # No neural hidden-state serialization: replay this one incomplete episode.
            for table in ("spans", "probabilities", "segments"):
                db.execute(f"DELETE FROM {table} WHERE episode=?", (row["id"],))
            db.execute("INSERT OR REPLACE INTO episodes VALUES(?,?,0)", (row["id"], digest))
            db.commit()
            a, b = bounds(row, path)
            offset = round(a * 16000)
            vad = vad_factory()
            try:
                check_model("vad", vad)
                vad_identity = model_identity(vad)
                for begin, end, cut in vad.stream(stream(path, a, b)):
                    if not (
                        isinstance(begin, int)
                        and isinstance(end, int)
                        and 0 <= begin < end
                        and end - begin <= 480000
                    ):
                        raise ValueError("invalid streaming VAD interval")
                    db.execute(
                        "INSERT INTO spans VALUES(?,?,?,?)",
                        (row["id"], offset + begin, offset + end, int(cut)),
                    )
                    db.commit()
            finally:
                release(vad)
                del vad
            diar_identity = None
            source_speakers = set()
            diar_covered_end = offset
            if diarizer_factory:
                diar = diarizer_factory()
                try:
                    check_model("diarization", diar)
                    diar_identity = model_identity(diar)
                    for chunk in diar.stream(stream(path, a, b)):
                        p = np.asarray(chunk["probabilities"], dtype=np.float32)
                        step = chunk["frame_samples"]
                        begin = offset + chunk["start_sample"]
                        if p.ndim != 2 or p.shape[1] != 8 or not np.isfinite(p).all():
                            raise ValueError("invalid diarization probabilities")
                        if step != 160 or np.any(p < 0) or np.any(p > 1):
                            raise ValueError("invalid diarization frame configuration")
                        diar_covered_end = max(diar_covered_end, begin + len(p) * step)
                        threshold = diar_identity.get("activity_threshold", 0.5)
                        source_speakers.update(np.flatnonzero(np.any(p > threshold, axis=0)))
                        # Persist intervals of at most 30 s for bounded range queries.
                        for index in range(0, len(p), 3000):
                            part = p[index : index + 3000]
                            part_begin = begin + index * step
                            db.execute(
                                "INSERT INTO probabilities VALUES(?,?,?,?,?)",
                                (
                                    row["id"],
                                    part_begin,
                                    part_begin + len(part) * step,
                                    step,
                                    part.tobytes(),
                                ),
                            )
                        db.commit()
                finally:
                    release(diar)
                    del diar
            for begin, end, cut in db.execute(
                "SELECT a,b,cut FROM spans WHERE episode=? ORDER BY a", (row["id"],)
            ):
                record = {
                    **row,
                    "id": f"{row['id']}:{begin}:{end}",
                    "audio": str(path),
                    "source_id": row.get("source_id", row["id"]),
                    "start_sample": begin,
                    "end_sample": end,
                    "timebase_hz": 16000,
                    "start": begin / 16000,
                    "end": end / 16000,
                    "boundary_cut": bool(cut),
                    "vad": vad_identity,
                    "source_sha256": digest,
                    "analysis_audio_adapter_sha256": audio_identity,
                }
                if diar_identity:
                    if end > diar_covered_end:
                        tail = end - diar_covered_end
                        if tail >= 160:
                            raise ValueError(
                                "missing diarization coverage beyond final partial frame"
                            )
                        record["boundary_cut"] = True
                        record["prepared_diarization"] = {
                            "diarization_status": "review",
                            "diarization_reasons": ["unscored_diarization_tail"],
                            "unscored_tail_samples": tail,
                            "diar_speakers": None,
                            "overlap_ratio": None,
                            "speaker_turns": [],
                            "speaker_id_scope": "source_local",
                        }
                    else:
                        record["prepared_diarization"] = summarize(
                            db, row["id"], begin, end, diar_identity.get("activity_threshold", 0.5)
                        )
                    record["prepared_diarization"]["speaker_capacity_reached"] = (
                        len(source_speakers) == 8
                    )
                    record["diarization_identity"] = diar_identity
                db.execute("INSERT INTO segments VALUES(?,?,?)", (row["id"], begin, dump(record)))
            db.execute("UPDATE episodes SET complete=1 WHERE id=?", (row["id"],))
            db.commit()
        # Publish only after every episode has finished. Re-running completed work is idempotent.
        fd, name = tempfile.mkstemp(prefix=".prepare-", dir=output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf8") as f:
                for row in entries(manifest):
                    for (encoded,) in db.execute(
                        "SELECT json FROM segments WHERE episode=? ORDER BY a", (row["id"],)
                    ):
                        f.write(encoded + "\n")
            if output.exists():
                if sha256(output) != sha256(Path(name)):
                    raise ValueError("existing prepare output differs; refusing overwrite")
            else:
                os.link(name, output)
        finally:
            Path(name).unlink(missing_ok=True)
        return db.execute("SELECT count(*) FROM segments").fetchone()[0]


def summarize(db, episode, begin, end, threshold=0.5):
    """Intersect original frame intervals; never rebin speakers into shared offset bins."""
    speakers, opened, turns = set(), {}, []
    overlap = covered = 0
    previous_end = begin
    for a, step, raw in db.execute(
        "SELECT a,step,pcm FROM probabilities WHERE episode=? AND a>=? AND a<? AND b>? ORDER BY a",
        (episode, begin - 480000, end, begin),
    ):
        probabilities = np.frombuffer(raw, dtype=np.float32).reshape(-1, 8)
        for index, frame in enumerate(probabilities):
            start, stop = max(begin, a + index * step), min(end, a + (index + 1) * step)
            if stop <= start:
                continue
            if start != previous_end:
                raise ValueError("missing or overlapping diarization frames")
            previous_end = stop
            current = set(int(i) for i in np.flatnonzero(frame > threshold))
            speakers.update(current)
            for speaker in list(opened):
                if speaker not in current:
                    turns.append(
                        {
                            "start": (opened.pop(speaker) - begin) / 16000,
                            "end": (start - begin) / 16000,
                            "speaker": f"speaker_{speaker}",
                        }
                    )
            for speaker in current:
                opened.setdefault(speaker, start)
            overlap += (stop - start) if len(current) > 1 else 0
            covered += stop - start
    if covered != end - begin:
        raise ValueError("missing diarization coverage")
    for speaker, start in opened.items():
        turns.append(
            {
                "start": (start - begin) / 16000,
                "end": (end - begin) / 16000,
                "speaker": f"speaker_{speaker}",
            }
        )
    return {
        "diar_speakers": len(speakers),
        "overlap_ratio": overlap / (end - begin),
        "overlap_seconds": overlap / 16000,
        "speaker_turns": sorted(turns, key=lambda item: (item["start"], item["speaker"])),
        "speaker_capacity_reached": len(speakers) == 8,
        "speaker_id_scope": "source_local",
    }
