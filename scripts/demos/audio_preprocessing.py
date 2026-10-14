"""
scripts/demos/audio_preprocessing.py — LinguaForge Phase 1: Audio Preprocessing Demo

Demonstrates the full Phase 1 pipeline:
  load → mono → resample → normalize → log-Mel spectrogram → visualize

Usage:
    # With your own audio file:
    python scripts/demos/audio_preprocessing.py path/to/audio.wav

    # With the built-in synthetic test signal (no audio file needed):
    python scripts/demos/audio_preprocessing.py

    # Save spectrogram to a custom path:
    python scripts/demos/audio_preprocessing.py --output my_spectrogram.png

    # Show spectrogram interactively (requires a display):
    python scripts/demos/audio_preprocessing.py --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add the project root to sys.path so we can import speech_engine without
# installing the package (useful during early development).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.audio import (
    get_metadata,
    load_audio,
    normalize_amplitude,
    resample,
    save_audio,
    to_mono,
)
from speech_engine.features import (
    HOP_LENGTH,
    MAX_FRAMES,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_truncate,
    visualize_spectrogram,
)

# ── ANSI colours for terminal output ──────────────────────────────────────────
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _section(title: str) -> None:
    print(f"\n{_CYAN}── {title} {'─' * max(0, 52 - len(title))}{_RESET}")


def _step(n: int, total: int, label: str) -> None:
    print(f"  {_BLUE}[{n}/{total}]{_RESET} {label}", end="", flush=True)


def _ok(detail: str = "") -> None:
    suffix = f"  {_DIM}{detail}{_RESET}" if detail else ""
    print(f"  {_GREEN}✓{_RESET}{suffix}")


# ── Synthetic test signal ──────────────────────────────────────────────────────


def generate_synthetic_signal(duration_s: float = 3.0, sample_rate: int = 22_050) -> np.ndarray:
    """Generate a multi-formant test signal that loosely resembles a vowel sound.

    We use multiple sine waves at speech-like frequencies:
      200 Hz  — fundamental (voice pitch, male)
      800 Hz  — first formant F1  (vowel quality)
      2400 Hz — second formant F2
      3600 Hz — third formant F3

    Amplitude modulation at 4 Hz creates a "syllable beat" effect that makes
    the spectrogram visually interesting and tests that the pipeline handles
    amplitude variation correctly.

    The signal is generated at 22050 Hz (CD quality half-rate) so the demo
    also exercises the resampler.
    """
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), dtype=np.float32)

    signal = (
        0.50 * np.sin(2 * np.pi * 200 * t)  # fundamental
        + 0.30 * np.sin(2 * np.pi * 800 * t)  # F1
        + 0.20 * np.sin(2 * np.pi * 2400 * t)  # F2
        + 0.10 * np.sin(2 * np.pi * 3600 * t)  # F3
    )

    # Amplitude modulation: creates 4 Hz "syllable" beats
    envelope = (1.0 + np.sin(2 * np.pi * 4 * t)) / 2.0
    return (signal * envelope).astype(np.float32)


# ── Main demo ─────────────────────────────────────────────────────────────────


def run_demo(
    audio_path: str | None,
    output_path: str,
    show: bool,
) -> None:
    TOTAL_STEPS = 6

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'═' * 62}{_RESET}")
    print(f"{_BOLD}  LinguaForge — Phase 1: Audio Preprocessing{_RESET}")
    print(f"{_BOLD}{'═' * 62}{_RESET}")

    # ── Input: real file or synthetic signal ──────────────────────────────────
    _section("Input")

    if audio_path is None:
        print(f"  No audio file provided — generating synthetic test signal.")
        print(f"  Frequency components: 200 Hz (F0) · 800 Hz (F1) · 2400 Hz (F2) · 3600 Hz (F3)")
        print(f"  Amplitude modulation: 4 Hz syllable envelope")
        print(f"  Sample rate: 22050 Hz  →  will be resampled to {SAMPLE_RATE} Hz")
        print(f"  Duration: 3.0 s\n")

        synthetic = generate_synthetic_signal(duration_s=3.0, sample_rate=22_050)

        # Save to disk first so we can demonstrate load_audio()
        synth_path = PROJECT_ROOT / "data" / "samples" / "synthetic_test.wav"
        save_audio(synthetic, synth_path, sample_rate=22_050)
        print(f"  Saved synthetic signal → {_DIM}{synth_path.relative_to(PROJECT_ROOT)}{_RESET}")
        audio_path = str(synth_path)
    else:
        audio_path = str(Path(audio_path).resolve())
        print(f"  File: {audio_path}")

    # ── Step 1: Metadata ───────────────────────────────────────────────────────
    _section("Metadata")
    t0 = time.perf_counter()
    meta = get_metadata(audio_path)
    elapsed = time.perf_counter() - t0
    print(meta)
    print(f"  {_DIM}(read in {elapsed * 1000:.1f} ms — no audio loaded){_RESET}")

    # ── Steps 2–5: Full pipeline, step by step ────────────────────────────────
    _section("Pipeline")

    # Step 2: Load
    _step(1, TOTAL_STEPS, "Load audio ")
    t0 = time.perf_counter()
    waveform, sr = load_audio(audio_path)
    _ok(
        f"{sr:,} Hz  ·  {waveform.shape[0]} ch  ·  {waveform.shape[-1]:,} samples  "
        f"·  {(time.perf_counter() - t0) * 1000:.1f} ms"
    )

    # Step 3: Mono
    _step(2, TOTAL_STEPS, "Convert to mono ")
    n_ch_before = waveform.shape[0] if waveform.ndim > 1 else 1
    waveform = to_mono(waveform)
    _ok(f"{n_ch_before} ch → 1 ch  ·  shape now {waveform.shape}")

    # Step 4: Resample
    _step(3, TOTAL_STEPS, f"Resample {sr:,} Hz → {SAMPLE_RATE:,} Hz ")
    t0 = time.perf_counter()
    peak_before = float(np.abs(waveform).max())
    waveform = resample(waveform, sr, SAMPLE_RATE)
    _ok(
        f"{len(waveform):,} samples  ·  {len(waveform) / SAMPLE_RATE:.3f} s  "
        f"·  {(time.perf_counter() - t0) * 1000:.1f} ms"
    )

    # Step 5: Normalize
    _step(4, TOTAL_STEPS, "Normalize amplitude ")
    peak_after_resample = float(np.abs(waveform).max())
    waveform = normalize_amplitude(waveform)
    peak_after = float(np.abs(waveform).max())
    _ok(f"peak {peak_before:.4f} → {peak_after:.4f}")

    # Step 6: Log-Mel spectrogram
    _step(5, TOTAL_STEPS, "Log-Mel spectrogram ")
    t0 = time.perf_counter()
    spec = log_mel_spectrogram(
        waveform,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    spec_time = (time.perf_counter() - t0) * 1000
    _ok(f"shape {spec.shape}  ·  {spec_time:.1f} ms")

    # Step 6: Pad/truncate (show what the model will see)
    _step(6, TOTAL_STEPS, f"Pad/truncate to {MAX_FRAMES} frames ")
    spec_padded = pad_or_truncate(spec, max_frames=MAX_FRAMES)
    _ok(
        f"shape {spec_padded.shape}  ({MAX_FRAMES} × 10 ms = {MAX_FRAMES * 10 / 1000:.0f}s context)"
    )

    # ── Spectrogram detail ────────────────────────────────────────────────────
    _section("Spectrogram Info")
    n_mels, n_frames = spec.shape
    duration_s = n_frames * HOP_LENGTH / SAMPLE_RATE
    ms_per_frame = HOP_LENGTH / SAMPLE_RATE * 1000
    print(f"  Shape      {spec.shape}   (n_mels × frames)")
    print(f"  Duration   {duration_s:.3f} s  ({n_frames} frames × {ms_per_frame:.0f} ms/frame)")
    print(f"  Mel bins   {n_mels}  covering 0 – {int(SAMPLE_RATE / 2):,} Hz")
    print(f"  n_fft      {N_FFT}  ({N_FFT / SAMPLE_RATE * 1000:.0f} ms window)")
    print(f"  hop        {HOP_LENGTH}  ({HOP_LENGTH / SAMPLE_RATE * 1000:.0f} ms step)")
    print(f"  Value range   min={spec.min():.4f}  max={spec.max():.4f}  mean={spec.mean():.4f}")

    # ── Pipeline: what the model will actually receive ─────────────────────────
    _section("What the Model Sees")
    print(f"  Input tensor shape: ({N_MELS}, {MAX_FRAMES})")
    print(f"  dtype: float32")
    print(f"  Values: normalized log-Mel energies in approx [-1.0, 1.0]")
    print(f"  Each column = 10 ms of audio  ·  Each row = one Mel frequency band")
    print(f"  The model reads this like an image: time across, frequency up")

    # ── Save spectrogram ───────────────────────────────────────────────────────
    _section("Output")
    out = Path(output_path)
    print(f"  Generating spectrogram figure...")
    title = f"LinguaForge Phase 1 — Log-Mel Spectrogram"
    fig = visualize_spectrogram(
        spec,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        title=title,
        save_path=out,
        show=show,
    )

    print(f"  {_GREEN}Saved → {out}{_RESET}")
    if not show:
        print(f"  Open it to see the spectrogram image.")

    plt_close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'─' * 62}{_RESET}")
    print(f"{_GREEN}Phase 1 complete.{_RESET}  All functions working correctly.")
    print(f"\nNext step: Phase 2 — build the dataset loader and tokenizer.")
    print(f"  Files to create: speech_engine/dataset.py, speech_engine/tokenizer.py")
    print()


def plt_close(fig) -> None:
    """Close figure to free memory — matplotlib keeps figures in memory by default."""
    import matplotlib.pyplot as plt

    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LinguaForge Phase 1: Audio Preprocessing Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to an audio file (WAV, FLAC, OGG). "
        "If omitted, a synthetic test signal is generated.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(PROJECT_ROOT / "data" / "samples" / "spectrogram.png"),
        help="Where to save the spectrogram PNG (default: data/samples/spectrogram.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the spectrogram interactively (requires a display / GUI backend).",
    )

    args = parser.parse_args()
    run_demo(audio_path=args.audio, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
