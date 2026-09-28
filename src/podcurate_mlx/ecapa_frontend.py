# Vendored from aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX
# Revision: e749e6e08557f4a3ceb6ce3bf6f0b79efe592a77. Apache-2.0.
# See ecapa_license.py for the full license.
"""Independent SpeechBrain-compatible log-mel frontend.

The implementation intentionally does not call SpeechBrain. Export validation
compares it with SpeechBrain's own Fbank module so frontend drift is caught
before artifacts are released.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160


def periodic_hamming(length: int = WIN_LENGTH) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)
    return (
        np.float32(0.54)
        - np.float32(0.46)
        * np.cos(np.float32(2.0 * np.pi) * positions / np.float32(length))
    ).astype(np.float32)


def speechbrain_filterbank(n_mels: int) -> np.ndarray:
    """Return SpeechBrain's triangular filters as ``[frequency, mel]``."""

    def hz_to_mel(value: float) -> float:
        return 2595.0 * np.log10(1.0 + value / 700.0)

    def mel_to_hz(value: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, value / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(SAMPLE_RATE / 2.0),
        n_mels + 2,
        dtype=np.float32,
    )
    hz_points = mel_to_hz(mel_points).astype(np.float32)
    centers = hz_points[1:-1]
    # SpeechBrain deliberately uses the lower-side band for both sides of each
    # triangle. This is not the usual asymmetric HTK/librosa construction.
    bands = (hz_points[1:] - hz_points[:-1])[:-1]
    frequencies = np.linspace(
        0.0, SAMPLE_RATE // 2, N_FFT // 2 + 1, dtype=np.float32
    )
    slopes = (frequencies[:, None] - centers[None, :]) / bands[None, :]
    return np.maximum(0.0, np.minimum(slopes + 1.0, -slopes + 1.0)).astype(
        np.float32
    )


def compute_fbank(audio: np.ndarray, n_mels: int) -> np.ndarray:
    """Compute SpeechBrain Fbank features as ``[frames, n_mels]``."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("audio must contain at least one sample")

    padded = np.pad(samples, (N_FFT // 2, N_FFT // 2), mode="constant")
    if padded.size < N_FFT:
        padded = np.pad(padded, (0, N_FFT - padded.size), mode="constant")

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[
        ::HOP_LENGTH
    ]
    windowed = frames * periodic_hamming()[None, :]
    spectrum = np.fft.rfft(windowed, n=N_FFT, axis=-1)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
        np.float32
    )
    mel = power @ speechbrain_filterbank(n_mels)
    decibels = np.float32(10.0) * np.log10(
        np.maximum(mel, np.float32(1e-10))
    )
    floor = np.max(decibels) - np.float32(80.0)
    return np.maximum(decibels, floor).astype(np.float32)


def sentence_mean_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected [frames, mels], got {values.shape}")
    return (values - values.mean(axis=0, keepdims=True)).astype(np.float32)


def deterministic_audio(seconds: float = 3.0) -> np.ndarray:
    count = int(round(seconds * SAMPLE_RATE))
    time_axis = np.arange(count, dtype=np.float32) / np.float32(SAMPLE_RATE)
    generator = np.random.default_rng(20260808)
    signal = (
        0.12 * np.sin(2.0 * np.pi * 173.0 * time_axis)
        + 0.06 * np.sin(2.0 * np.pi * 271.0 * time_axis)
        + 0.01 * generator.standard_normal(count)
    )
    return signal.astype(np.float32)
