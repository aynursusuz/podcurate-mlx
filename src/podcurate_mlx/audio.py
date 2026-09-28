"""Bounded FFmpeg reads. Source files are never rewritten."""

import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0


def resampling_warning(source_rate: int, target_rate: int) -> str | None:
    """Describe conversion limits, without inventing a perceptual-quality score."""
    if source_rate == target_rate:
        return None
    change = f"{source_rate} -> {target_rate} Hz"
    if source_rate < target_rate:
        return f"{change}: upsampling does not restore missing frequency detail."
    return f"{change}: downsampling limits bandwidth to below {target_rate / 2:g} Hz."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def duration(path: Path) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = float(p.stdout.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("audio duration must be positive and finite")
    return value


def decode(path: Path, start: float, end: float) -> np.ndarray:
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("expected finite 0 <= start < end")
    if end - start > MAX_SECONDS + 1e-6:
        raise ValueError("audio span exceeds 30 seconds; run prepare first")
    p = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(end - start),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(p.stdout, dtype="<f4").copy()
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("empty or non-finite decoded audio")
    if abs(audio.size / SAMPLE_RATE - (end - start)) > 0.1:
        raise ValueError("decoded duration differs from requested span by more than 0.1 s")
    return audio


def signals(audio: np.ndarray) -> dict:
    """Metrics on the unnormalized 16k mono analysis view, not a calibrated MOS."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return {
        "duration_s": len(audio) / SAMPLE_RATE,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "peak": float(np.abs(audio).max()),
        "full_scale_ratio": float(np.mean(np.abs(audio) >= 1.0)),
        "silent": rms == 0,
        "pcm_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
    }


def stream(path: Path, start=0.0, end=None, block_samples=16000 * 20):
    """Decode one continuous stream. RAM is bounded by block_samples, even for days of audio."""
    import tempfile

    if not math.isfinite(start) or start < 0 or (end is not None and end <= start):
        raise ValueError("invalid stream boundaries")
    if not 0 < block_samples <= 480000:
        raise ValueError("block_samples must be within 1..480000")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(path)]
    if end is not None:
        command += ["-t", str(end - start)]
    command += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    # Disk-backed stderr avoids a pipe deadlock without buffering an entire episode.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while raw := process.stdout.read(block_samples * 4):
                audio = np.frombuffer(raw, dtype="<f4").copy()
                if not np.isfinite(audio).all():
                    raise ValueError("nonfinite audio")
                yield audio
            if process.wait():
                errors.seek(0)
                raise RuntimeError(errors.read(4096).decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


def info(path: Path) -> dict:
    import json

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration_ts,time_base",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]
