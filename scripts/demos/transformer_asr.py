"""Demo: attention, positional encodings and the Transformer speech model on toy data.

Usage:
    python scripts/demos/transformer_asr.py
    python scripts/demos/transformer_asr.py --epochs 30 --d-model 256
    python scripts/demos/transformer_asr.py --librispeech data/LibriSpeech --epochs 50
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.attention import MultiHeadAttention
from speech_engine.dataset import build_dataloader
from speech_engine.evaluate import char_error_rate, word_error_rate
from speech_engine.model_transformer import (
    TransformerASR,
    make_causal_mask,
)
from speech_engine.positional_encoding import LearnedPositionalEmbedding, sinusoidal_encoding
from speech_engine.tokenizer import BOS_ID, EOS_ID, PAD_ID, CharTokenizer
from speech_engine.train import TrainConfig, resolve_device
from speech_engine.train_transformer import train_transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer_pipeline import make_toy_dataset

# ANSI colours
_G = "\033[92m"
_B = "\033[94m"
_C = "\033[96m"
_Y = "\033[93m"
_D = "\033[2m"
_W = "\033[1m"
_R = "\033[0m"


def _sec(title: str) -> None:
    print(f"\n{_C}── {title} {'─' * max(0, 54 - len(title))}{_R}")


def _ok(label: str, value: str) -> None:
    print(f"  {_B}{label:<28}{_R} {value}")


# Section 1: Attention mechanics


def demo_attention(d_model: int, n_heads: int, device: torch.device) -> None:
    _sec("Multi-Head Attention")

    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0).to(device)

    B, T_q, T_k = 2, 6, 10

    query = torch.randn(B, T_q, d_model, device=device)
    key = torch.randn(B, T_k, d_model, device=device)
    value = torch.randn(B, T_k, d_model, device=device)

    # 1. Unconstrained cross-attention (encoder → decoder)
    out, weights = attn(query, key, value)
    _ok("Cross-attn output:", f"{tuple(out.shape)}  (B, T_q, d_model)")
    _ok("Attn weight shape:", f"{tuple(weights.shape)}  (B, n_heads, T_q, T_k)")
    _ok("Weight row sums:", f"{weights[0, 0].sum(dim=-1).tolist()[:3]}  (each ≈ 1.0)")

    # 2. Causal self-attention (decoder self-attn)
    T = 8
    x = torch.randn(B, T, d_model, device=device)
    causal_m = make_causal_mask(T, device)
    out_c, w_c = attn(x, x, x, mask=causal_m)

    print(f"\n  Causal mask (T={T}) — lower triangular, True=attend:")
    mask_row = causal_m[0, 0]  # (T, T)
    for row_i in range(min(T, 5)):
        row = "".join("▓" if mask_row[row_i, j] else "░" for j in range(T))
        future_leak = w_c[0, 0, row_i, row_i + 1 :].sum().item()
        print(f"    pos {row_i}: {row}  future leak={future_leak:.2e}")

    print(f"\n  Parameter count:")
    _ok("  W_q, W_k, W_v, W_o:", f"4 × ({d_model}×{d_model}) = {4 * d_model * d_model:,} params")
    _ok("  Total (this layer):", f"{sum(p.numel() for p in attn.parameters()):,}")


# Section 2: Positional encodings


def demo_positional_encodings(d_model: int) -> None:
    _sec("Positional Encodings — Sinusoidal vs Learned")

    # Sinusoidal
    pe = sinusoidal_encoding(max_len=100, d_model=d_model)  # (1, 100, d_model)
    print(f"  Sinusoidal PE shape: {tuple(pe.shape)}")
    print(f"  First 4 positions, first 8 dims:")
    for pos in range(4):
        vals = pe[0, pos, :8].tolist()
        print(f"    pos {pos}: {[f'{v:+.3f}' for v in vals]}")

    print(f"\n  Key property — dot product of nearby positions:")
    for delta in [0, 1, 5, 20, 50]:
        sim = torch.dot(pe[0, 0], pe[0, delta]).item() / d_model
        print(f"    pos 0 · pos {delta:>2}  =  {sim:+.4f}  {'← same' if delta == 0 else ''}")

    print(f"\n  Learned PE: nn.Embedding({448}, {d_model}) — initialised N(0, 0.02)")
    le = LearnedPositionalEmbedding(d_model=d_model, max_len=448)
    _ok("  Learned PE params:", f"{le.embedding.weight.numel():,}")
    _ok("  Sinusoidal params:", "0  (fixed, registered as buffer)")


# Section 3: TransformerASR architecture


def demo_model(vocab_size: int, d_model: int, quick: bool, device: torch.device) -> TransformerASR:
    _sec("TransformerASR Architecture")

    if quick:
        model = TransformerASR(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=4,
            d_ff=d_model * 4,
            n_enc_layers=2,
            n_dec_layers=2,
            cnn_channels=(16, 32, 64),
            dropout=0.1,
        )
        print(f"  {_Y}Quick mode: small model for fast demo.{_R}")
        print(f"  Full model: d_model=384, n_enc_layers=6, n_dec_layers=6")
    else:
        model = TransformerASR(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=6,
            d_ff=d_model * 4,
            n_enc_layers=6,
            n_dec_layers=6,
        )

    model.to(device)
    print()
    print(model.summary())
    return model


def demo_forward_pass(
    model: TransformerASR, tokenizer: CharTokenizer, device: torch.device
) -> None:
    _sec("Forward Pass — Encoder + Decoder Shapes")

    model.eval()

    B, T, V = 2, 300, tokenizer.vocab_size
    L_tgt = 8  # decoder input length

    mel = torch.randn(B, 80, T, device=device)
    mel_len = torch.full((B,), T, dtype=torch.long, device=device)
    # Decoder input: BOS + random token IDs
    tgt_ids = torch.randint(4, V, (B, L_tgt), device=device)
    tgt_ids[:, 0] = BOS_ID

    with torch.no_grad():
        memory, mem_mask = model.encode(mel, mel_len)
        logits = model.decode(tgt_ids, memory, mem_mask)

    _ok("Input mel:", f"({B}, 80, {T})  — 3s audio")
    _ok("Encoder output:", f"{tuple(memory.shape)}  — (B, T//2, d_model)")
    _ok("Decoder input:", f"({B}, {L_tgt})  — [BOS, c1…c{L_tgt - 1}]")
    _ok("Logits output:", f"{tuple(logits.shape)}  — (B, L, vocab_size)")

    probs = torch.softmax(logits[0, 0], dim=-1)
    _ok("Probs at pos 0:", f"sum={probs.sum().item():.5f}  max={probs.max().item():.4f}")

    print(f"\n  Loss target alignment:")
    print(f"    decoder input  → logits[:, 0, :] predicts target[0] = c1")
    print(f"    decoder input  → logits[:, L-1, :] predicts target[L-1] = EOS")
    print(f"    Cross-entropy averages over all L positions (PAD ignored).")


# Section 4: Loss comparison CTC vs seq2seq


def demo_loss(model: TransformerASR, tokenizer: CharTokenizer, device: torch.device) -> None:
    _sec("Loss — Cross-Entropy (seq2seq) vs CTC")

    model.eval()
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    sentence = "hello world"
    toks = tokenizer.encode(sentence)  # no BOS/EOS
    L = len(toks)
    T = 200

    mel = torch.randn(1, 80, T, device=device)
    dec_in = torch.tensor([[BOS_ID] + toks], dtype=torch.long, device=device)  # (1, L+1)
    target = torch.tensor([toks + [EOS_ID]], dtype=torch.long, device=device)  # (1, L+1)

    with torch.no_grad():
        logits = model(mel, dec_in)  # (1, L+1, V)
    B_, Lp, V = logits.shape
    loss = ce_loss(logits.view(B_ * Lp, V), target.view(B_ * Lp))

    expected = math.log(tokenizer.vocab_size)

    print(f"  Sentence:         {repr(sentence)}")
    print(f"  Token IDs:        {toks}  (len={L})")
    print(f"  Decoder input:    [BOS, {', '.join(str(t) for t in toks)}]")
    print(f"  Target:           [{', '.join(str(t) for t in toks)}, EOS]")
    print(f"  CE loss:          {loss.item():.4f}")
    print(f"  Expected random:  log({tokenizer.vocab_size}) = {expected:.2f}")
    print(f"  Ratio:            {loss.item() / expected:.2f}×  (≈1.0 means properly initialised)")
    print(f"\n  CTC vs seq2seq:")
    print(f"    CTC:     trains encoder only, loss = -log P(transcript | audio frames)")
    print(f"              no explicit decoder, alignment is marginalised over all paths")
    print(f"    Seq2seq: trains encoder+decoder jointly, loss = -log P(y_t | y_<t, audio)")
    print(f"              explicit left-to-right language model in the decoder")


# Section 5: Training loop


def demo_training(
    model: TransformerASR,
    tokenizer: CharTokenizer,
    examples: list,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    _sec(f"Training — {config.max_epochs} epochs on {len(examples)} examples")

    n_train = max(1, int(len(examples) * 0.8))
    train_loader = build_dataloader(examples[:n_train], tokenizer, batch_size=config.batch_size)
    val_loader = (
        build_dataloader(examples[n_train:], tokenizer, batch_size=config.batch_size)
        if len(examples) > n_train
        else None
    )

    return train_transformer(model, train_loader, val_loader, config, device)


# Section 6: Inference


def demo_inference(
    model: TransformerASR,
    tokenizer: CharTokenizer,
    examples: list,
    device: torch.device,
) -> None:
    _sec("Inference — Auto-Regressive Generation")

    model.eval()

    print(f"  {'Reference':<38}  {'Generated':<38}  CER")
    print(f"  {'─' * 38}  {'─' * 38}  {'─' * 5}")

    from speech_engine.audio import preprocess
    from speech_engine.features import MAX_FRAMES, log_mel_spectrogram, pad_or_truncate

    refs, hyps = [], []

    for audio_path, ref_text in examples[:5]:
        waveform, _ = preprocess(audio_path)
        spec = log_mel_spectrogram(waveform)
        spec = pad_or_truncate(spec, MAX_FRAMES)
        mel = torch.from_numpy(spec).unsqueeze(0).to(device)  # (1, 80, T)

        token_ids = model.generate(mel, max_len=100)[0]
        # Strip BOS and EOS
        token_ids = [t for t in token_ids if t not in (BOS_ID, EOS_ID, PAD_ID)]
        hyp_text = tokenizer.decode(token_ids)

        cer = char_error_rate([ref_text], [hyp_text])
        print(f"  {repr(ref_text[:36]):<38}  {repr(hyp_text[:36]):<38}  {cer:.3f}")

        refs.append(ref_text)
        hyps.append(hyp_text)

    cer = char_error_rate(refs, hyps)
    wer = word_error_rate(refs, hyps)
    print(f"\n  Corpus CER: {cer:.3f}   WER: {wer:.3f}")
    print(f"\n  {_D}Note: toy synthetic audio → garbage output is expected.")
    print(f"  With real paired audio (LibriSpeech) the model learns to generate")
    print(f"  coherent text.{_R}")


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="LinguaForge Transformer ASR")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128, dest="d_model")
    parser.add_argument("--librispeech", default=None)
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print(f"\n{_W}{'═' * 64}{_R}")
    print(f"{_W}  LinguaForge — Transformer Encoder-Decoder ASR{_R}")
    print(f"{_W}{'═' * 64}{_R}")

    device = resolve_device(args.device)
    _sec("Device")
    _ok("Selected:", str(device))

    # Dataset
    _sec("Dataset")
    if args.manifest:
        from speech_engine.dataset import load_manifest

        examples = load_manifest(args.manifest)
        print(f"  {len(examples)} examples from {args.manifest}")
    elif args.librispeech:
        from speech_engine.dataset import load_librispeech

        examples = load_librispeech(args.librispeech, args.split)
        examples = examples[:200]
        print(f"  {len(examples)} LibriSpeech ({args.split}) examples")
    else:
        toy_dir = PROJECT_ROOT / "data" / "toy_dataset"
        examples = make_toy_dataset(toy_dir, n=10)
        print(f"  Toy dataset ({len(examples)} synthetic examples)")
        print(f"  {_D}For real training: --librispeech data/LibriSpeech{_R}")

    # Tokenizer
    tok_path = PROJECT_ROOT / "data" / "tokenizer.json"
    tokenizer = CharTokenizer.load(tok_path) if tok_path.exists() else CharTokenizer()
    if not tok_path.exists():
        tokenizer.fit([t for _, t in examples])
        tokenizer.save(tok_path)
    _ok("Tokenizer:", repr(tokenizer))

    quick = not args.librispeech and not args.manifest

    # Sections
    demo_attention(d_model=args.d_model, n_heads=4, device=device)
    demo_positional_encodings(d_model=args.d_model)
    model = demo_model(tokenizer.vocab_size, args.d_model, quick, device)
    demo_forward_pass(model, tokenizer, device)
    demo_loss(model, tokenizer, device)

    # Training
    config = TrainConfig(
        max_epochs=args.epochs,
        batch_size=4,
        peak_lr=1e-3,
        min_lr=1e-6,
        warmup_steps=max(10, len(examples) // 4),
        log_every=max(1, len(examples) // (4 * 5)),
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints"),
        device=args.device,
    )
    results = demo_training(model, tokenizer, examples, config, device)

    # Print loss curve
    losses = results["train_losses"]
    if losses:
        _sec("Training Loss Curve")
        lo, hi = min(losses), max(losses)
        rng = hi - lo if hi != lo else 1.0
        for i, loss in enumerate(losses):
            bar = "█" * int((hi - loss) / rng * 50)
            print(f"  Epoch {i + 1:>3}  {loss:.4f}  {_G}{bar}{_R}")

    # Inference
    demo_inference(model, tokenizer, examples, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
