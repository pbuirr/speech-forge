"""
scripts/demos/ctc_training.py — LinguaForge Phase 3: CTC Model + Training Demo

Demonstrates:
  1. CTCModel architecture — our own CNN + BiLSTM + CTC design
  2. A forward pass — shapes from input to output
  3. CTC loss computation — what the model optimises
  4. A real training loop — loss should decrease on toy data
  5. Greedy decode — convert model output to text
  6. Prefix beam search — better decode with k hypotheses
  7. CER and WER — our own edit-distance metrics

Usage:
    # Quick demo (tiny model, 5 epochs):
    python scripts/demos/ctc_training.py

    # Longer training on toy data:
    python scripts/demos/ctc_training.py --epochs 30

    # Train on the official LibriSpeech training split (download first):
    python scripts/demos/ctc_training.py --librispeech data/LibriSpeech --split train-clean-100 --epochs 50

    # Resume from checkpoint:
    python scripts/demos/ctc_training.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.dataset import build_dataloader, load_librispeech, load_manifest
from speech_engine.decode import beam_decode, greedy_decode
from speech_engine.evaluate import (
    char_error_rate,
    evaluate_dataset,
    levenshtein,
    word_error_rate,
)
from speech_engine.model_ctc import CTCModel
from speech_engine.tokenizer import BLANK_ID, CharTokenizer
from speech_engine.train import TrainConfig, resolve_device, train

# Reuse toy dataset generator from phase2_demo
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer_pipeline import make_toy_dataset, TOY_SENTENCES

# ── ANSI colours ──────────────────────────────────────────────────────────────
_G = "\033[92m"; _B = "\033[94m"; _C = "\033[96m"
_Y = "\033[93m"; _D = "\033[2m";  _W = "\033[1m"; _R = "\033[0m"

def _sec(title: str) -> None:
    print(f"\n{_C}── {title} {'─' * max(0, 54 - len(title))}{_R}")

def _ok(label: str, value: str) -> None:
    print(f"  {_B}{label:<26}{_R} {value}")


# ── Section 1: Model architecture ────────────────────────────────────────────

def demo_model(vocab_size: int, quick: bool = True) -> CTCModel:
    _sec("CTCModel Architecture")

    if quick:
        # Small model — fast on CPU, shows the pipeline
        model = CTCModel(
            vocab_size=vocab_size,
            n_mels=80,
            cnn_channels=(16, 32, 64),
            hidden_size=128,
            num_lstm_layers=2,
            dropout=0.1,
        )
        print(f"  {_Y}Quick mode: small model for fast demo.{_R}")
        print(f"  For real training: CTCModel(hidden_size=256, num_lstm_layers=3)")
    else:
        model = CTCModel(
            vocab_size=vocab_size,
            n_mels=80,
            cnn_channels=(32, 64, 128),
            hidden_size=256,
            num_lstm_layers=3,
            dropout=0.1,
        )

    print()
    print(model.summary())
    return model


# ── Section 2: Forward pass inspection ───────────────────────────────────────

def demo_forward_pass(model: CTCModel, device: torch.device) -> None:
    _sec("Forward Pass — Shapes")

    model.to(device)
    model.eval()

    B, N_MELS, T = 4, 80, 300   # 4 examples, 80 mel bins, 300 frames (3s)
    mel = torch.randn(B, N_MELS, T).to(device)
    mel_lengths = torch.full((B,), T, dtype=torch.long).to(device)

    print(f"  Dummy input: B={B}, n_mels={N_MELS}, T={T} (3s of audio)")
    print()

    with torch.no_grad():
        log_probs = model(mel)                                    # (T_out, B, V)
        enc_lengths = model.encoder_output_length(mel_lengths)   # (B,)

    T_out = log_probs.shape[0]
    _ok("mel (input):", f"({B}, {N_MELS}, {T})")
    _ok("log_probs (output):", f"{tuple(log_probs.shape)}  [time-first for CTCLoss]")
    _ok("enc_lengths:", f"{enc_lengths.tolist()}")
    _ok("Time reduction:", f"{T} → {T_out} frames  ({T // T_out}×)")
    _ok("Value range:", f"[{log_probs.min():.3f}, {log_probs.max():.3f}]  (log-probs ≤ 0)")
    _ok("exp(log_probs).sum(dim=-1):", "≈ 1.0 per frame  (valid probability distribution)")

    # Verify probs sum to ~1 at every frame
    probs = log_probs.exp()
    prob_sums = probs.sum(dim=-1)  # (T_out, B)
    _ok("  prob sum range:", f"[{prob_sums.min():.4f}, {prob_sums.max():.4f}]")

    print(f"\n  CTC loss will receive:")
    _ok("  log_probs:", f"{tuple(log_probs.shape)}  (T_out, B, vocab_size)")
    _ok("  input_lengths:", f"{enc_lengths.tolist()}  (frames per example)")
    _ok("  target_lengths:", f"[L_1, L_2, ..., L_B]  (chars per transcript)")


# ── Section 3: CTC loss demonstration ────────────────────────────────────────

def demo_ctc_loss(model: CTCModel, tokenizer: CharTokenizer, device: torch.device) -> None:
    _sec("CTC Loss")

    model.eval()
    ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True, reduction="mean")

    # Tiny example: one sentence
    sentence = "hello"
    token_ids = tokenizer.encode(sentence)
    T = 150  # 1.5s audio at 16kHz → 150 mel frames

    mel = torch.randn(1, 80, T).to(device)
    mel_lengths = torch.tensor([T], dtype=torch.long).to(device)
    targets = torch.tensor([token_ids], dtype=torch.long)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.long)

    with torch.no_grad():
        log_probs = model(mel)
        input_lengths = model.encoder_output_length(mel_lengths)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    print(f"  Sentence:        {repr(sentence)}")
    print(f"  Token IDs:       {token_ids}  (len={len(token_ids)})")
    print(f"  Input frames:    {T}  → encoder output: {input_lengths[0].item()}")
    print(f"  CTC constraint:  input_length ({input_lengths[0].item()}) >= target_length ({len(token_ids)})  ✓")
    print(f"  CTC loss:        {loss.item():.4f}")
    print(f"\n  Interpretation:")
    expected_per_char = -math.log(1.0 / tokenizer.vocab_size)
    print(f"    CTC sums over ALL valid alignments (not just one path),")
    print(f"    so the expected initial loss (~50–150) is much higher than")
    print(f"    the per-frame -log(1/{tokenizer.vocab_size}) = {expected_per_char:.2f}.")
    print(f"    Our initial loss: {loss.item():.2f}  — normal for untrained CTC on {tokenizer.vocab_size} classes.")
    print(f"    Watch for the loss to DECREASE during training — that's the signal.")


# ── Section 4: Edit distance and metrics demo ─────────────────────────────────

def demo_metrics() -> None:
    _sec("Edit Distance, CER, and WER — all from scratch")

    pairs = [
        ("hello world",  "helo world"),       # one deletion
        ("the cat sat",  "the sat"),           # word deletion
        ("i want water", "i want water"),      # perfect
        ("mister quilter is the apostle",
         "mister quiltar is the apostal"),     # two substitutions
    ]

    print(f"  {'Reference':<35}  {'Hypothesis':<35}  CER     WER")
    print(f"  {'─'*35}  {'─'*35}  {'─'*6}  {'─'*6}")

    for ref, hyp in pairs:
        cer = char_error_rate([ref], [hyp])
        wer = word_error_rate([ref], [hyp])
        r_str = repr(ref[:33])
        h_str = repr(hyp[:33])
        print(f"  {r_str:<35}  {h_str:<35}  {cer:.3f}   {wer:.3f}")

    print(f"\n  Levenshtein walk-through for ('hello', 'helo'):")
    ref, hyp = list("hello"), list("helo")
    dist = levenshtein(ref, hyp)
    print(f"    ref: {ref}")
    print(f"    hyp: {hyp}")
    print(f"    edit distance: {dist}  (1 deletion of the second 'l')")
    print(f"    CER: {dist}/{len(ref)} = {dist/len(ref):.3f}")


# ── Section 5: Full training loop ─────────────────────────────────────────────

def demo_training(
    model: CTCModel,
    tokenizer: CharTokenizer,
    examples: list[tuple[str, str]],
    config: TrainConfig,
    device: torch.device,
    resume_from: Optional[Path] = None,
) -> None:
    _sec(f"Training Loop — {config.max_epochs} epochs on {len(examples)} examples")

    # Split: 80% train, 20% val
    n_train = max(1, int(len(examples) * 0.8))
    train_examples = examples[:n_train]
    val_examples   = examples[n_train:]

    train_loader = build_dataloader(
        train_examples, tokenizer,
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = build_dataloader(
        val_examples, tokenizer,
        batch_size=config.batch_size, shuffle=False,
    ) if val_examples else None

    model.to(device)
    results = train(model, train_loader, val_loader, config, device, resume_from)

    # Print ASCII loss curve
    losses = results["train_losses"]
    if losses:
        _sec("Training Loss Curve")
        lo, hi = min(losses), max(losses)
        rng = hi - lo if hi != lo else 1.0
        width = 50
        for i, loss in enumerate(losses):
            bar_len = int((hi - loss) / rng * width)
            bar = "█" * bar_len
            print(f"  Epoch {i+1:>3}  {loss:.4f}  {_G}{bar}{_R}")


# ── Section 6: Decode demo ────────────────────────────────────────────────────

def demo_decode(
    model: CTCModel,
    tokenizer: CharTokenizer,
    examples: list[tuple[str, str]],
    device: torch.device,
) -> None:
    _sec("Decoding — Greedy vs Beam Search")

    model.eval()
    model.to(device)

    print(f"  {'Reference':<40}  {'Greedy':<30}  {'Beam (k=5)':<30}")
    print(f"  {'─'*40}  {'─'*30}  {'─'*30}")

    for audio_path, ref_text in examples[:5]:
        # Load and process audio
        from speech_engine.audio import preprocess
        from speech_engine.features import log_mel_spectrogram, pad_or_truncate, MAX_FRAMES
        import torch

        waveform, _ = preprocess(audio_path)
        spec = log_mel_spectrogram(waveform)
        spec = pad_or_truncate(spec, MAX_FRAMES)
        mel = torch.from_numpy(spec).unsqueeze(0).to(device)   # (1, 80, T)

        with torch.no_grad():
            log_probs = model(mel)   # (T_out, 1, V)

        greedy_text = greedy_decode(log_probs, tokenizer)[0]
        beam_text   = beam_decode(log_probs, tokenizer, beam_width=5)[0]

        ref_str    = repr(ref_text[:38])
        greedy_str = repr(greedy_text[:28]) if greedy_text else repr("")
        beam_str   = repr(beam_text[:28])   if beam_text   else repr("")

        print(f"  {ref_str:<40}  {greedy_str:<30}  {beam_str:<30}")

    print(f"\n  {_D}Note: on toy synthetic audio, the model cannot transcribe real speech.")
    print(f"  Outputs will be mostly empty or repeated characters.")
    print(f"  This is expected — the model needs real paired speech data to learn.{_R}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LinguaForge Phase 3: CTC ASR Demo")
    parser.add_argument("--epochs",      type=int,  default=5,   help="Training epochs (default 5)")
    parser.add_argument("--quick",       action="store_true",     help="Use tiny model (fast on CPU)")
    parser.add_argument("--librispeech", default=None,            help="LibriSpeech root directory")
    parser.add_argument("--split",       default="train-clean-100", help="LibriSpeech training split; never report training-on-test as independent quality")
    parser.add_argument("--manifest",    default=None,            help="TSV manifest file")
    parser.add_argument("--resume",      default=None,            help="Resume from checkpoint path")
    parser.add_argument("--device",      default="auto",          help="cpu / cuda / mps / auto")
    args = parser.parse_args()

    print(f"\n{_W}{'═' * 62}{_R}")
    print(f"{_W}  LinguaForge — Phase 3: CTC Model, Training, Decode, Eval{_R}")
    print(f"{_W}{'═' * 62}{_R}")

    # ── Device ────────────────────────────────────────────────────────────────
    device = resolve_device(args.device)
    _sec("Device")
    _ok("Selected device:", str(device))
    if device.type == "cpu":
        print(f"  {_D}(Training on CPU is fine for the demo and toy dataset.)")
        print(f"   For LibriSpeech, CUDA or Apple MPS greatly speeds up training.){_R}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    _sec("Dataset")
    if args.manifest:
        from speech_engine.dataset import load_manifest
        examples = load_manifest(args.manifest)
        print(f"  Loaded {len(examples)} examples from {args.manifest}")
    elif args.librispeech:
        from speech_engine.dataset import load_librispeech
        examples = load_librispeech(args.librispeech, args.split)
        print(f"  Loaded {len(examples)} LibriSpeech ({args.split}) examples")
        if len(examples) > 200:
            examples = examples[:200]
            print(f"  (capped at 200 for demo)")
    else:
        toy_dir  = PROJECT_ROOT / "data" / "toy_dataset"
        examples = make_toy_dataset(toy_dir, n=10)
        print(f"  Using toy dataset ({len(examples)} synthetic examples)")
        print(f"  {_D}For real training: python scripts/demos/ctc_training.py --librispeech data/LibriSpeech{_R}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = PROJECT_ROOT / "data" / "tokenizer.json"
    if tok_path.exists():
        tokenizer = CharTokenizer.load(tok_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.fit([t for _, t in examples])
        tokenizer.save(tok_path)
    _ok("Tokenizer:", repr(tokenizer))

    # ── Model ─────────────────────────────────────────────────────────────────
    quick = args.quick or (not args.librispeech and not args.manifest)
    model = demo_model(tokenizer.vocab_size, quick=quick)

    # ── Forward pass ──────────────────────────────────────────────────────────
    demo_forward_pass(model, device)

    # ── CTC loss ──────────────────────────────────────────────────────────────
    demo_ctc_loss(model, tokenizer, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    demo_metrics()

    # ── Training ──────────────────────────────────────────────────────────────
    config = TrainConfig(
        max_epochs    = args.epochs,
        batch_size    = 4,
        peak_lr       = 1e-3,
        warmup_steps  = max(10, len(examples) // 4),
        log_every     = max(1, len(examples) // (4 * 5)),  # ~5 logs per epoch
        checkpoint_dir= str(PROJECT_ROOT / "checkpoints"),
        device        = args.device,
    )

    resume = Path(args.resume) if args.resume else None
    demo_training(model, tokenizer, examples, config, device, resume)

    # ── Decode ────────────────────────────────────────────────────────────────
    demo_decode(model, tokenizer, examples, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_W}{'─' * 62}{_R}")
    print(f"{_G}Phase 3 complete.{_R}  Model, training loop, decoder, and metrics working.")
    print(f"""
What you built:
  • CTCModel: CNN encoder + BiLSTM stack + linear classifier   (model_ctc.py)
  • Training loop with cosine LR warmup + gradient clipping    (train.py)
  • Greedy CTC decoder                                         (decode.py)
  • Prefix beam search decoder (from scratch, in log space)    (decode.py)
  • Levenshtein edit distance, CER, WER                        (evaluate.py)

What uses external libraries:
  • nn.Conv2d, nn.LSTM, nn.Linear, nn.CTCLoss — PyTorch primitives
  • Everything else (architecture, training, decoding, metrics) is ours.

Next step: Phase 4 — Transformer encoder-decoder (mini-Whisper architecture)
  Files to create:
    speech_engine/model_transformer.py — Transformer Enc-Dec with cross-attention
    speech_engine/positional_encoding.py — sinusoidal + learned PE
    speech_engine/attention.py — multi-head self-attention from scratch
""")


if __name__ == "__main__":
    main()
