"""Log-Mel spectrogram features.

Defaults: 16 kHz audio, 25 ms Hann window (n_fft=400), 10 ms hop (160 samples),
80 Mel bands up to 8 kHz.

    waveform → STFT → power → Mel filterbank → log10 → clamp and normalise

The output, shape (80, T), is the input to every speech model in the package.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.functional

# Frontend constants
SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400  # 25 ms window
HOP_LENGTH = 160  # 10 ms hop
FMAX = 8_000.0  # Hz

# Maximum number of frames for a fixed-length model input.
# 3000 frames × 10 ms/frame = 30 seconds.
MAX_FRAMES = 3_000


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmax: float,
) -> torch.Tensor:
    """(n_mels, n_fft//2+1) matrix of triangular filters spaced evenly on the Mel scale.

    Slaney area normalisation keeps the wider high-frequency filters from
    dominating.
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

    The same function is used for training and local inference.

    Args:
        waveform:    (N,) float32 array at ``sample_rate``
        sample_rate: sample rate in Hz
        n_mels:      number of Mel bins
        n_fft:       FFT size in samples
        hop_length:  hop size in samples
        fmax:        upper frequency limit in Hz

    Returns:
        (n_mels, T) float32 array, T ≈ len(waveform) / hop_length. Values are
        affinely scaled, not bounded to [-1, 1].

    Raises:
        ValueError: if waveform is not 1-dimensional
    """
    if waveform.ndim != 1:
        raise ValueError(
            f"Expected a 1D (mono) waveform, got shape {waveform.shape}. "
            "Call audio.to_mono() first."
        )

    tensor = torch.from_numpy(waveform.astype(np.float32))

    # Step 1: Short-Time Fourier Transform
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

    # Step 2: Power spectrum
    # Take the squared magnitude of each complex STFT coefficient.
    # This gives the energy at each (time, frequency) cell.
    # Drop the last frame so T = len(waveform) // hop_length:
    magnitudes = stft[..., :-1].abs() ** 2  # (n_fft//2+1, T)

    # Step 3: Mel filterbank
    # Matrix multiply the filterbank (n_mels, n_fft//2+1) with the power
    # spectrum (n_fft//2+1, T) to produce (n_mels, T) Mel energies.
    # Each row of the filterbank is one triangular filter; the matrix multiply
    # computes the weighted sum of power spectrum bins for each Mel bin.
    mel_fb = _mel_filterbank(n_mels, n_fft, sample_rate, fmax).to(tensor.device)
    mel_spec = mel_fb @ magnitudes  # (n_mels, T)

    # Step 4: Log compression
    # We use log base 10 rather than natural log because the values map nicely:
    #   log10(1e-10) = -10  (very quiet)
    #   log10(1.0)   =  0   (max energy for normalized audio)
    # The 1e-10 floor prevents log(0) = -inf for completely silent frames.
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()

    # Step 5: Dynamic range compression
    # Clamp values to the top 8 log10 units (~80 dB) of dynamic range.
    # Anything quieter than 80 dB below the peak is treated as silence.
    # This prevents very quiet background noise from dominating gradient updates.
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)

    # Step 6: Affine scaling (not bounded to [-1, 1])
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
        the matplotlib Figure.
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
    """Zero-pad or truncate a spectrogram to exactly ``max_frames`` columns.

    Zero padding isn't silence, so callers should keep the real lengths and mask
    the padding.

    Args:
        spec:       (n_mels, T) spectrogram
        max_frames: target number of frames

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
