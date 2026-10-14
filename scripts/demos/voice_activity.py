"""
scripts/demos/voice_activity.py — LinguaForge Phase 8: VAD demo

Generates synthetic audio (silence + voiced-speech analogs + silence),
runs the hand-rolled VAD, and prints an ASCII timeline and feature report.

No external audio files needed.  Requires only numpy.

Usage:
    .venv/bin/python3 scripts/demos/voice_activity.py
    .venv/bin/python3 scripts/demos/voice_activity.py --wav path/to/audio.wav
    .venv/bin/python3 scripts/demos/voice_activity.py --stream
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_engine.vad import VADState, VoiceActivityDetector


# ── Synthetic audio generator ─────────────────────────────────────────────────


def _voiced(sr: int, dur: float, f0: float = 120.0, n_harmonics: int = 12) -> np.ndarray:
    """
    Synthetic voiced-speech analog: sum of harmonics with 1/k amplitude
    envelope (simulates vocal-tract resonance) plus low-level noise.
    """
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    signal = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        signal += np.sin(2 * math.pi * f0 * k * t) / k
    signal /= np.max(np.abs(signal) + 1e-9)
    signal += np.random.randn(len(t)) * 0.03
    return signal.astype(np.float32)


def _unvoiced(sr: int, dur: float, center: float = 4000.0) -> np.ndarray:
    """
    Synthetic unvoiced-speech analog: band-pass filtered noise around
    a centre frequency (simulates fricatives like /s/ /f/ /sh/).
    """
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t)).astype(np.float32)
    # Simple resonant peak via two-pole IIR
    bw = center / 3
    r = 1 - math.pi * bw / sr
    pole = 2 * math.pi * center / sr
    a1, a2 = -2 * r * math.cos(pole), r**2
    out = np.zeros_like(noise)
    for i in range(2, len(noise)):
        out[i] = noise[i] - a1 * out[i - 1] - a2 * out[i - 2]
    out *= 0.4 / (np.max(np.abs(out)) + 1e-9)
    return out


def _silence(sr: int, dur: float, noise_amp: float = 0.003) -> np.ndarray:
    """Background noise — white noise at very low amplitude."""
    return (np.random.randn(int(sr * dur)) * noise_amp).astype(np.float32)


def build_test_signal(sr: int = 16_000) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Build a ~7-second test signal with known ground truth segments.

    Layout (seconds):
      0.0 – 0.5  silence
      0.5 – 1.8  voiced speech #1 ("el gato")
      1.8 – 2.3  silence
      2.3 – 2.6  unvoiced burst ("curioso")
      2.6 – 2.8  tiny pause
      2.8 – 4.5  voiced speech #2 ("exploró la casa vieja")
      4.5 – 5.1  silence
      5.1 – 6.0  voiced speech #3 ("con cuidado")
      6.0 – 6.5  silence
    """
    np.random.seed(42)
    segments_gt = [(0.5, 1.8), (2.3, 2.6), (2.8, 4.5), (5.1, 6.0)]

    parts = [
        _silence(sr, 0.5),
        _voiced(sr, 1.3, f0=130.0),
        _silence(sr, 0.5),
        _unvoiced(sr, 0.3),
        _silence(sr, 0.2),
        _voiced(sr, 1.7, f0=115.0),
        _silence(sr, 0.6),
        _voiced(sr, 0.9, f0=145.0),
        _silence(sr, 0.5),
    ]
    return np.concatenate(parts), segments_gt


# ── ASCII visualiser ──────────────────────────────────────────────────────────


def _ascii_timeline(
    audio: np.ndarray,
    sr: int,
    detected: list,  # list[SpeechSegment]
    ground_truth: list[tuple[float, float]],
    width: int = 80,
) -> str:
    total = len(audio) / sr
    bar_w = width - 20

    def to_x(t: float) -> int:
        return min(bar_w - 1, int(t / total * bar_w))

    lines = []
    lines.append(f"\n{'─' * width}")
    lines.append("VAD TIMELINE  (each char ≈ {:.0f}ms)".format(total / bar_w * 1000))
    lines.append("─" * width)

    # Ground truth
    gt_bar = [" "] * bar_w
    for s, e in ground_truth:
        for x in range(to_x(s), to_x(e)):
            gt_bar[x] = "▓"
    lines.append(f"  GROUND TRUTH │{''.join(gt_bar)}│")

    # Detected
    det_bar = [" "] * bar_w
    for seg in detected:
        for x in range(to_x(seg.start), to_x(seg.end)):
            det_bar[x] = "▓"
    lines.append(f"  DETECTED     │{''.join(det_bar)}│")

    # Diff
    diff_bar = []
    for g, d in zip(gt_bar, det_bar):
        if g == d == "▓":
            diff_bar.append("✓")
        elif g == "▓" and d != "▓":
            diff_bar.append("!")  # missed
        elif g != "▓" and d == "▓":
            diff_bar.append("?")  # false alarm
        else:
            diff_bar.append(" ")
    lines.append(f"  DIFF         │{''.join(diff_bar)}│")

    # Time axis
    n_labels = 6
    axis = [" "] * bar_w
    for i in range(n_labels + 1):
        t = total * i / n_labels
        x = to_x(t)
        lbl = f"{t:.1f}s"
        for j, c in enumerate(lbl):
            if x + j < bar_w:
                axis[x + j] = c
    lines.append(f"               │{''.join(axis)}│")
    lines.append("─" * width)
    lines.append("  ✓ true positive  ! missed  ? false alarm")
    lines.append("─" * width)
    return "\n".join(lines)


def _feature_plot(features: dict, width: int = 80) -> str:
    bar_w = width - 22
    lines = ["\nFEATURE OVERVIEW (per frame)", "─" * width]
    for name, values in [
        ("energy (RMS)", features["energy"]),
        ("ZCR (norm)", features["zcr"] / (features["zcr"].max() + 1e-9)),
        ("flatness", features["flatness"]),
    ]:
        v_norm = values / (values.max() + 1e-9)
        step = max(1, len(v_norm) // bar_w)
        sampled = v_norm[::step][:bar_w]
        bar = "".join("█" if v > 0.5 else ("▄" if v > 0.25 else "·") for v in sampled)
        lines.append(f"  {name:<18}│{bar:<{bar_w}}│")
    lines.append("─" * width)
    return "\n".join(lines)


# ── Streaming demo ────────────────────────────────────────────────────────────


def run_streaming_demo(audio: np.ndarray, sr: int, vad: VoiceActivityDetector) -> None:
    stream = vad.stream()
    chunk_len = sr // 10  # 100ms chunks
    print("\nStreaming demo (100ms chunks):")
    print("─" * 60)

    prev_state = VADState.SILENCE
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i : i + chunk_len]
        state, segs = stream.push(chunk)
        t = i / sr
        if state != prev_state:
            print(f"  {t:.2f}s  {prev_state.name:<10} → {state.name}")
            prev_state = state
        for seg in segs:
            print(f"           completed segment: {seg}")

    final = stream.flush()
    for seg in final:
        print(f"  [flush] {seg}")
    print("─" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", default=None, help="Path to a .wav file (optional)")
    p.add_argument("--sr", type=int, default=16_000)
    p.add_argument("--stream", action="store_true", help="Also run streaming demo")
    args = p.parse_args()

    print("=" * 80)
    print("LinguaForge  Phase 8 — Voice Activity Detection")
    print("Hand-rolled: energy + ZCR + spectral flatness + 4-state machine")
    print("=" * 80)

    sr = args.sr

    if args.wav:
        # Load wav file (stdlib struct + wave module only)
        import struct, wave

        with wave.open(args.wav, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            width = wf.getsampwidth()
            data = wf.readframes(wf.getnframes())
        fmt = {1: "b", 2: "h", 4: "i"}[width]
        raw = struct.unpack(f"{len(data) // width}{fmt}", data)
        sig = np.array(raw, dtype=np.float32)
        if n_ch > 1:
            sig = sig.reshape(-1, n_ch).mean(axis=1)  # downmix to mono
        sig /= 2 ** (8 * width - 1)  # normalise to [-1, 1]
        audio = sig
        ground_truth = []
        print(f"\nLoaded: {args.wav}  ({sr}Hz, {len(audio) / sr:.2f}s)")
    else:
        print("\nUsing synthetic test signal (7s)…")
        audio, ground_truth = build_test_signal(sr)

    # ── Batch detect ──────────────────────────────────────────────────────────
    vad = VoiceActivityDetector(sr=sr)
    detected = vad.detect(audio)
    features = vad.frame_features(audio)

    print(f"\nDetected {len(detected)} speech segment(s):")
    for seg in detected:
        print(f"  {seg}")

    if ground_truth:
        # Compute simple recall / precision over detected frames
        total_secs = len(audio) / sr
        step = 0.01
        ts = np.arange(0, total_secs, step)

        gt_mask = np.zeros(len(ts), bool)
        det_mask = np.zeros(len(ts), bool)
        for s, e in ground_truth:
            gt_mask |= (ts >= s) & (ts < e)
        for seg in detected:
            det_mask |= (ts >= seg.start) & (ts < seg.end)

        tp = np.sum(gt_mask & det_mask)
        fn = np.sum(gt_mask & ~det_mask)
        fp = np.sum(~gt_mask & det_mask)
        recall = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1 = 2 * recall * prec / (recall + prec + 1e-9)
        print(f"\nVs. ground truth:  precision={prec:.2f}  recall={recall:.2f}  F1={f1:.2f}")

    print(_ascii_timeline(audio, sr, detected, ground_truth))
    print(_feature_plot(features))

    if args.stream:
        run_streaming_demo(audio, sr, vad)

    print("\nPhase 8 complete.")


if __name__ == "__main__":
    main()
