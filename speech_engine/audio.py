"""
speech_engine/audio.py — Audio I/O and preprocessing pipeline.

Responsibilities:
  - Load audio files (WAV, FLAC, OGG; MP3/MP4 via torchaudio+ffmpeg backend)
  - Convert multi-channel audio to mono
  - Resample to 16 kHz (the universal standard for ASR)
  - Peak-normalize amplitude
  - Save processed audio
  - Extract file metadata without loading the full waveform

Why 16 kHz?
  Human speech occupies 80 Hz – 8 kHz. The Nyquist theorem requires a sample
  rate at least twice the highest frequency of interest, so 16 kHz is the
  minimum that captures all speech content without redundancy. Whisper, wav2vec2,
  and every modern ASR system uses this rate.

Why peak normalization?
  Raw recordings vary wildly in volume (microphone gain, distance, environment).
  Normalizing to a fixed peak level makes the model see consistent amplitude
  regardless of recording conditions, which speeds up training and reduces
  the model's need to learn volume invariance from data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

TARGET_SR = 16_000  # Hz — standard for all LinguaForge ASR models


@dataclass
class AudioMetadata:
    """Snapshot of audio file properties, both before and after preprocessing."""
    path: str
    sample_rate: int
    num_samples: int
    num_channels: int
    duration_s: float
    file_size_bytes: int

    def __str__(self) -> str:
        size_kb = self.file_size_bytes / 1024
        return (
            f"  Path          {self.path}\n"
            f"  Duration      {self.duration_s:.3f} s\n"
            f"  Sample rate   {self.sample_rate:,} Hz\n"
            f"  Samples       {self.num_samples:,}\n"
            f"  Channels      {self.num_channels}\n"
            f"  File size     {size_kb:.1f} KB"
        )


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file into a float32 NumPy array.

    Returns:
        waveform: shape (num_channels, num_samples), dtype float32, values in [-1, 1]
        sample_rate: original sample rate of the file

    soundfile is the primary loader (handles WAV, FLAC, OGG natively and is fast).
    torchaudio is the fallback for formats that require an ffmpeg backend (MP3, MP4, M4A).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # soundfile returns (samples, channels) — transpose to (channels, samples)
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data.T, sr  # (C, N)
    except Exception:
        # Fallback: torchaudio with whatever backend is installed (ffmpeg, sox, etc.)
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        return waveform.numpy(), int(sr)  # (C, N)


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Average all channels into a single mono channel.

    Args:
        waveform: (C, N) or (N,)
    Returns:
        (N,) float32 array

    Averaging is preferred over selecting channel 0 because it preserves
    energy from both channels and reduces noise from either channel alone.
    """
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        return waveform.mean(axis=0)
    raise ValueError(f"Expected 1D or 2D waveform, got shape {waveform.shape}")


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample a mono waveform to target_sr using a sinc filter.

    Args:
        waveform: (N,) float32 array
        orig_sr:  original sample rate
        target_sr: desired output sample rate (default: 16000)
    Returns:
        (M,) float32 array where M = len(waveform) * target_sr / orig_sr

    torchaudio.functional.resample uses a Kaiser-windowed sinc filter, which
    is significantly higher quality than scipy.signal.resample for speech
    (it has better stopband attenuation and avoids aliasing artifacts).
    """
    if orig_sr == target_sr:
        return waveform

    tensor = torch.from_numpy(waveform.astype(np.float32))
    # resample expects (C, N) or (N,); our waveform is already 1D here
    resampled = F.resample(
        tensor,
        orig_freq=orig_sr,
        new_freq=target_sr,
        resampling_method="sinc_interp_kaiser",
    )
    return resampled.numpy()


def normalize_amplitude(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize so the loudest sample reaches target_peak.

    Args:
        waveform:    (N,) float32 array
        target_peak: 0.95 leaves slight headroom to avoid clipping on re-encode

    Returns:
        (N,) float32 array with peak == target_peak

    Silent recordings (peak < 1e-6) are returned unchanged to avoid
    amplifying background noise into the signal range.
    """
    peak = float(np.abs(waveform).max())
    if peak < 1e-6:
        return waveform
    return (waveform / peak * target_peak).astype(np.float32)


def save_audio(waveform: np.ndarray, path: str | Path, sample_rate: int = TARGET_SR) -> None:
    """Write a float32 mono waveform to a WAV file.

    Creates parent directories automatically if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.astype(np.float32), sample_rate)


def get_metadata(path: str | Path) -> AudioMetadata:
    """Read file metadata without loading the full audio into memory.

    Uses soundfile's info() call which reads only the file header.
    For large files (hours of audio) this is much faster than load_audio().
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    info = sf.info(str(path))
    return AudioMetadata(
        path=str(path),
        sample_rate=info.samplerate,
        num_samples=info.frames,
        num_channels=info.channels,
        duration_s=info.duration,
        file_size_bytes=path.stat().st_size,
    )


def record_audio(duration_s: float, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Record audio from the default system microphone.

    Args:
        duration_s:  length of recording in seconds
        sample_rate: recording sample rate (default: 16000)

    Returns:
        (N,) float32 mono waveform

    Requires: pip install sounddevice
    The recorded audio is mono — sounddevice captures a single channel.
    """
    if not _SOUNDDEVICE_AVAILABLE:
        raise ImportError(
            "sounddevice is required for microphone recording.\n"
            "Install it with: pip install sounddevice"
        )
    n_samples = int(duration_s * sample_rate)
    print(f"  Recording {duration_s:.1f}s at {sample_rate} Hz — speak now...")
    audio = sd.rec(n_samples, samplerate=sample_rate, channels=1, dtype="float32", blocking=True)
    print("  Recording complete.")
    return audio.flatten()  # (N,)


def preprocess(
    path: str | Path,
    target_sr: int = TARGET_SR,
) -> tuple[np.ndarray, AudioMetadata]:
    """Full preprocessing pipeline for a single audio file.

    Pipeline:
        load → mono → resample → normalize

    Returns:
        waveform: (N,) float32 array, peak ≈ 0.95, sample rate = target_sr
        metadata: AudioMetadata reflecting the post-processing state

    This is the function that dataset.py and the inference API will call.
    All four steps are intentionally separate functions so they can be tested
    independently and composed differently in specialized contexts (e.g.,
    streaming audio doesn't use load_audio at all).
    """
    raw_meta = get_metadata(path)
    waveform, sr = load_audio(path)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sr, target_sr)
    waveform = normalize_amplitude(waveform)

    processed_meta = AudioMetadata(
        path=raw_meta.path,
        sample_rate=target_sr,
        num_samples=len(waveform),
        num_channels=1,
        duration_s=len(waveform) / target_sr,
        file_size_bytes=raw_meta.file_size_bytes,
    )
    return waveform, processed_meta
