"""
speech_engine/features.py — Log-Mel spectrogram computation and visualization.

This module implements the feature extraction stage that sits between raw audio
and the ASR model. The output of this module IS the input to every model in
speech_engine/ (CTC, seq2seq, Transformer).

Why log-Mel spectrograms?
--------------------------
A raw waveform (sequence of amplitude samples) is hard for a neural network to
learn from directly — small shifts in time produce large changes in the signal
even when the speech content is identical. The spectrogram converts the 1D
time-domain signal into a 2D time-frequency representation where:

  - X axis = time (one column per frame, every 10 ms)
  - Y axis = frequency (one row per Mel bin)
  - Pixel value = energy at that frequency at that moment

The Mel scale is a non-linear frequency scale that matches human perception:
we hear differences between 100 Hz and 200 Hz much more clearly than between
3000 Hz and 3100 Hz, even though both are 100 Hz apart. Mel bands provide a
perceptually motivated frequency spacing, not a complete model of human hearing.

Log compression (log10) reduces the dynamic range from ~60 dB to ~6 units,
making gradient-based training much more stable.

Default parameters (Whisper-compatible):
-----------------------------------------
  n_fft=400     → 25 ms analysis window at 16 kHz  (400 / 16000 = 0.025 s)
  hop_length=160 → 10 ms frame step at 16 kHz      (160 / 16000 = 0.010 s)
  n_mels=80     → 80 perceptual frequency bands
  fmax=8000     → upper frequency limit (Nyquist for 16 kHz)

These settings are inspired by common log-Mel speech frontends. Sharing settings
does not establish comparable recognition quality. All LinguaForge weights are
trained locally; no third-party pretrained checkpoint is used.

Pipeline (implemented explicitly below):
-----------------------------------------
  waveform (N,)
    └─ torch.stft + Hann window → complex STFT (n_fft/2+1, T+1)
         └─ squared magnitude → power spectrum (n_fft/2+1, T)
              └─ Mel filterbank (matrix multiply) → mel energies (n_mels, T)
                   └─ log10 + 1e-10 floor → log mel (n_mels, T)
                        └─ dynamic range clamp + normalize → output (n_mels, T)
"""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.functional

# ── Whisper-compatible constants ───────────────────────────────────────────────
SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400  # 25 ms window
HOP_LENGTH = 160  # 10 ms hop
FMAX = 8_000.0  # Hz

# Maximum number of frames for a fixed-length model input.
# 3000 frames × 10 ms/frame = 30 seconds (Whisper's context window).
MAX_FRAMES = 3_000


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmax: float,
) -> torch.Tensor:
    """Build a (n_mels, n_fft//2+1) Mel filterbank matrix.

      The filterbank is a set of n_mels triangular filters, linearly spaced
      on the Mel scale between 0 Hz and fmax. Each filter integrates the
      power spectrum over a small frequency range and produces one Mel bin.

      norm="slaney" area-normalizes each filter so that wider filters (at high
    frequencies) don't dominate just because they cover more frequency bins.
    """
    # torchaudio returns (n_freqs, n_mels) — transpose to (n_mels, n_freqs)
    fbanks = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=fmax,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    return fbanks.T  # (n_mels, n_fft//2+1)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmax: float = FMAX,
) -> np.ndarray:
    """Convert a mono float32 waveform to a log-Mel spectrogram.

    This uses a conventional 80-bin log-Mel frontend with dynamic-range
    compression. It is shared by training and local inference; matching another
    system's frontend settings does not imply matching its model or accuracy.

    Args:
        waveform:    (N,) float32 array, sample rate must match `sample_rate`
        sample_rate: sample rate of the waveform in Hz
        n_mels:      number of Mel frequency bins
        n_fft:       FFT size in samples (controls frequency resolution)
        hop_length:  hop size in samples (controls time resolution)
        fmax:        upper frequency limit in Hz

    Returns:
        (n_mels, T) float32 array where T ≈ len(waveform) / hop_length
        Values use affine scaling and are not bounded to [-1, 1].

    Raises:
        ValueError: if waveform is not 1-dimensional
    """
    if waveform.ndim != 1:
        raise ValueError(
            f"Expected a 1D (mono) waveform, got shape {waveform.shape}. "
            "Call audio.to_mono() first."
        )

    tensor = torch.from_numpy(waveform.astype(np.float32))

    # ── Step 1: Short-Time Fourier Transform ─────────────────────────────────
    # The STFT splits the signal into overlapping frames of length n_fft.
    # A Hann window is applied to each frame before the FFT to reduce
    # "spectral leakage" — energy from one frequency bleeding into adjacent bins.
    # center=True pads the signal by n_fft//2 on each side so the first
    # and last frames are centered on the first/last audio sample.
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode="reflect",  # reflect padding avoids discontinuities at edges
        normalized=False,
        return_complex=True,
    )
    # stft shape: (n_fft//2+1, T+1) complex
    # We only need n_fft//2+1 frequency bins because the FFT of a real signal
    # is symmetric — the upper half carries no new information.

    # ── Step 2: Power spectrum ───────────────────────────────────────────────
    # Take the squared magnitude of each complex STFT coefficient.
    # This gives the energy at each (time, frequency) cell.
    # Drop the last frame (Whisper convention) to get a clean T:
    magnitudes = stft[..., :-1].abs() ** 2  # (n_fft//2+1, T)

    # ── Step 3: Mel filterbank ───────────────────────────────────────────────
    # Matrix multiply the filterbank (n_mels, n_fft//2+1) with the power
    # spectrum (n_fft//2+1, T) to produce (n_mels, T) Mel energies.
    # Each row of the filterbank is one triangular filter; the matrix multiply
    # computes the weighted sum of power spectrum bins for each Mel bin.
    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate, fmax).to(tensor.device)
    mel_spec = mel_fb @ magnitudes  # (n_mels, T)

    # ── Step 4: Log compression ──────────────────────────────────────────────
    # We use log base 10 rather than natural log because the values map nicely:
    #   log10(1e-10) = -10  (very quiet)
    #   log10(1.0)   =  0   (max energy for normalized audio)
    # The 1e-10 floor prevents log(0) = -inf for completely silent frames.
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()

    # ── Step 5: Dynamic range compression ───────────────────────────────────
    # Clamp values to the top 8 log10 units (~80 dB) of dynamic range.
    # Anything quieter than 80 dB below the peak is treated as silence.
    # This prevents very quiet background noise from dominating gradient updates.
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)

    # ── Step 6: Affine scaling (not bounded to [-1, 1]) ───────────────────────
    # After step 5, values are in [max-8, max].
    # For normalized audio, max ≈ 0 dB → range is approximately [-8, 0].
    # (x + 4) / 4 maps [-8, 0] → [-1, 1].
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.numpy().astype(np.float32)  # (n_mels, T)


def visualize_spectrogram(
    spec: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    title: str = "Log-Mel Spectrogram",
    save_path: str | Path | None = None,
    show: bool = False,
) -> "Any":
    """Render a spectrogram as a labelled matplotlib figure.

    Args:
        spec:        (n_mels, T) array from log_mel_spectrogram()
        sample_rate: audio sample rate (used to label the time axis)
        hop_length:  STFT hop length in samples (used to label the time axis)
        title:       figure title text
        save_path:   if given, the figure is saved to this path (PNG / PDF / SVG)
        show:        if True, calls plt.show() — only works in interactive sessions

    Returns:
        matplotlib Figure — you can further customize it before saving.
    """
    import matplotlib.pyplot as plt  # lazy: not needed at import time

    n_mels, n_frames = spec.shape
    duration_s = n_frames * hop_length / sample_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    img = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",  # low Mel bin = bottom of image = low frequency
        extent=[0, duration_s, 0, n_mels],
        cmap="magma",
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
    )

    cbar = fig.colorbar(img, ax=ax, pad=0.01, shrink=0.95)
    cbar.set_label("Log magnitude (normalized)", color="#aaaaaa", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel(f"Mel bin (0 – {n_mels})", color="#aaaaaa", fontsize=10)
    ax.set_title(title, color="#e0e0e0", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # Parameter summary below the figure
    frame_ms = hop_length / sample_rate * 1000
    param_text = (
        f"n_mels={n_mels}  ·  n_fft={N_FFT} ({N_FFT / sample_rate * 1000:.0f}ms window)  ·  "
        f"hop={hop_length} ({frame_ms:.0f}ms/frame)  ·  "
        f"frames={n_frames}  ·  duration={duration_s:.2f}s  ·  shape={spec.shape}"
    )
    fig.text(
        0.5,
        0.01,
        param_text,
        ha="center",
        fontsize=7.5,
        color="#555555",
        fontfamily="monospace",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="#0d0d0d")

    if show:
        plt.show()

    return fig


def pad_or_truncate(spec: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Pad with zeros or truncate so the spectrogram has exactly max_frames columns.

    During training, all examples in a batch must have the same shape.
    Whisper uses max_frames=3000 (30 seconds at 10 ms/frame).

    Zeros are padding placeholders, not acoustically equivalent to silence.
    Callers must retain real lengths and mask padding in their model/loss.

    Args:
        spec:       (n_mels, T) spectrogram array
        max_frames: target number of time frames

    Returns:
        (n_mels, max_frames) array
    """
    _, n_frames = spec.shape
    if n_frames == max_frames:
        return spec
    if n_frames > max_frames:
        return spec[:, :max_frames]
    pad_width = max_frames - n_frames
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
