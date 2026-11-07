"""Bounded-memory decoding; overlapping windows are never stored separately."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SAMPLE_RATE, STRIDE_SECONDS, WINDOW_SECONDS


def inspect_audio(path: Path) -> tuple[int, int]:
    """Validate the entire managed copy, including truncated/invalid payloads."""
    try:
        with sf.SoundFile(path) as audio:
            if audio.frames <= 0 or audio.samplerate <= 0 or audio.channels <= 0:
                raise ValueError("Recording is empty.")
            decoded = 0
            for block in audio.blocks(blocksize=65_536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError("Recording contains non-finite audio samples.")
                decoded += len(block)
            if decoded != audio.frames:
                raise ValueError("Recording is truncated or could not be fully decoded.")
            return decoded, audio.samplerate
    except (sf.LibsndfileError, OSError) as error:
        raise ValueError(f"Cannot decode {path.name}: {error}. Use a valid WAV or FLAC file.") from error


def windows(frames: int, sample_rate: int):
    for start_frame in range(0, frames, STRIDE_SECONDS * sample_rate):
        yield start_frame / sample_rate, min(frames, start_frame + WINDOW_SECONDS * sample_rate) / sample_rate


def read_window(path: str, start: float, end: float) -> np.ndarray:
    with sf.SoundFile(path) as audio:
        audio.seek(round(start * audio.samplerate))
        samples = audio.read(round((end - start) * audio.samplerate), dtype="float32", always_2d=True)
        if len(samples) == 0:
            raise ValueError("Assigned audio window is empty.")
        mono = samples.mean(axis=1)
        if audio.samplerate != SAMPLE_RATE:
            divisor = gcd(audio.samplerate, SAMPLE_RATE)
            mono = resample_poly(mono, SAMPLE_RATE // divisor, audio.samplerate // divisor)
    length = WINDOW_SECONDS * SAMPLE_RATE
    result = np.zeros(length, dtype=np.float32)
    result[:min(length, len(mono))] = mono[:length]
    if not np.isfinite(result).all():
        raise ValueError("Audio window contains non-finite samples.")
    return result
