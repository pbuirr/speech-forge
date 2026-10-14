"""
scripts/translation/train_nmt.py — LinguaForge Phase 7 NMT trainer

Trains a Spanish→English Seq2Seq Transformer on the curated corpus at
data/corpora/es_en.tsv.  Everything is from scratch: BPE tokenisation,
Transformer architecture, beam search — no pretrained weights, no HuggingFace.

Usage:
    # Full train (saves to data/models/nmt_es_en.pt)
    .venv/bin/python3 scripts/translation/train_nmt.py

    # Quick smoke-test (100 epochs, small model)
    .venv/bin/python3 scripts/translation/train_nmt.py --fast

    # Translate after training
    .venv/bin/python3 scripts/translation/train_nmt.py --translate "el gato curioso exploró la casa vieja"
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_engine.bpe import BPEVocab
from speech_engine.nmt import Seq2SeqNMT, collate_pairs, load_checkpoint, nmt_loss, save_checkpoint

# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS_PATH = ROOT / "data" / "corpora" / "es_en.tsv"
GENERATED_CORPUS_PATH = ROOT / "data" / "corpora" / "es_en_generated.tsv"
REAL_CORPUS_PATH = ROOT / "data" / "corpora" / "es_en_real.tsv"

_TEST_SENTENCES = [
    "el gato curioso exploró la casa vieja con cuidado",
    "me gusta leer porque aprendo mucho",
    "¿puede decirme alguien cómo solucionar esta ecuación?",
    "no sé qué hacer con este problema",
    "podría ayudarme con este ejercicio",
    "debería estudiar más para el examen",
    "¿cuánto cuesta el billete de tren?",
    "estoy buscando un trabajo nuevo",
    "tengo mucho sueño hoy",
    "¿dónde está la estación de tren?",
    "hola buenos días",
    "gracias por tu ayuda",
    "no tengo coche",
]


def load_corpus(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0].strip().lower(), parts[1].strip().lower()))
    return pairs


def load_all_corpora() -> list[tuple[str, str]]:
    """Load curated + generated + real OPUS corpora, deduplicate on source."""
    curated = load_corpus(CORPUS_PATH)
    generated = load_corpus(GENERATED_CORPUS_PATH) if GENERATED_CORPUS_PATH.exists() else []
    real = load_corpus(REAL_CORPUS_PATH) if REAL_CORPUS_PATH.exists() else []

    if real:
        print(f"  Real OPUS corpus loaded: {len(real):,} pairs")

    seen_src: set[str] = set()
    combined: list[tuple[str, str]] = []
    # Curated first (highest quality), then real, then generated templates
    for pair in curated + real + generated:
        if pair[0] not in seen_src:
            seen_src.add(pair[0])
            combined.append(pair)
    return combined


# ── Train/test split ──────────────────────────────────────────────────────────

SPLIT_SEED = 42
TEST_RATIO = 0.10
SPLITS_DIR = ROOT / "data" / "splits"


def make_or_load_split(
    pairs: list[tuple[str, str]],
    src: str = "es",
    tgt: str = "en",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Create a reproducible 90/10 train/test split and persist the test set.

    The test set is saved as a TSV to data/splits/{src}_{tgt}_test_seed{SPLIT_SEED}.tsv
    so that:
      - Re-running evaluation always uses the SAME held-out sentences
      - The test set is never contaminated by augmentation or resampling
      - Scores from different training runs are directly comparable

    If the split file already exists, the saved test set is used directly
    (to guarantee comparability across runs even if the corpus grows).
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_path = SPLITS_DIR / f"{src}_{tgt}_test_seed{SPLIT_SEED}.tsv"

    if split_path.exists():
        # Re-use existing test split — guarantees comparability across runs
        test_pairs: list[tuple[str, str]] = []
        test_src_set: set[str] = set()
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                test_pairs.append((parts[0], parts[1]))
                test_src_set.add(parts[0])
        # Train = everything not in the test split
        train_pairs = [(s, t) for s, t in pairs if s not in test_src_set]
        print(f"  Loaded existing split: {len(train_pairs):,} train / {len(test_pairs):,} test")
        print(f"  Split file: {split_path}")
        return train_pairs, test_pairs

    # Create split with fixed seed
    rng = random.Random(SPLIT_SEED)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_test = max(100, int(len(shuffled) * TEST_RATIO))
    test_pairs = shuffled[:n_test]
    train_pairs = shuffled[n_test:]

    # Persist test set
    with split_path.open("w", encoding="utf-8") as f:
        f.write(f"# LinguaForge test split — {src}→{tgt}, seed={SPLIT_SEED}\n")
        f.write(f"# {len(test_pairs)} pairs  (held out from {len(pairs)} total)\n")
        for s, t in test_pairs:
            f.write(f"{s}\t{t}\n")

    print(f"  Created split: {len(train_pairs):,} train / {len(test_pairs):,} test")
    print(f"  Saved test set → {split_path}")
    return train_pairs, test_pairs


# ── BLEU evaluation ───────────────────────────────────────────────────────────


def run_bleu_eval(
    model: Seq2SeqNMT,
    src_vocab: BPEVocab,
    tgt_vocab: BPEVocab,
    test_pairs: list[tuple[str, str]],
    device: torch.device,
    beam: int = 4,
    max_len: int = 100,
    n_samples: int = 25,
) -> dict:
    """
    Translate the held-out test set and compute corpus BLEU.
    Returns the full bleu_detail dict plus sample translations.
    """
    from speech_engine.bleu import bleu_detail, sentence_bleu

    sources = [s for s, _ in test_pairs]
    references = [r for _, r in test_pairs]

    model.eval()
    hypotheses: list[str] = []
    with torch.no_grad():
        for src_str in sources:
            src_ids = src_vocab.encode(src_str, add_bos=True, add_eos=True)
            if not src_ids:
                hypotheses.append("")
                continue
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            out_ids = model.translate(src_t, beam_size=beam, max_len=max_len)
            hypotheses.append(tgt_vocab.decode(out_ids).strip())

    detail = bleu_detail(hypotheses, references)

    # Save n_samples example translations
    rng = random.Random(SPLIT_SEED)
    idx = rng.sample(range(len(test_pairs)), min(n_samples, len(test_pairs)))
    samples = [
        {
            "source": sources[i],
            "reference": references[i],
            "hypothesis": hypotheses[i],
            "sentence_bleu": round(sentence_bleu(hypotheses[i], references[i]) * 100, 2),
        }
        for i in sorted(idx)
    ]

    return {**detail, "samples": samples}


def augment(pairs: list[tuple[str, str]], factor: int = 3) -> list[tuple[str, str]]:
    """Light data augmentation: duplicate with minor perturbations."""
    augmented = list(pairs)
    for _ in range(factor - 1):
        for src, tgt in pairs:
            # Drop final punctuation variant
            s2 = src.rstrip(".,;:?!")
            t2 = tgt.rstrip(".,;:?!")
            if s2 != src:
                augmented.append((s2, t2))
    return augmented


# ── Vocab ─────────────────────────────────────────────────────────────────────


def build_vocabs(pairs: list[tuple[str, str]], vocab_size: int) -> tuple[BPEVocab, BPEVocab]:
    src_corpus = [s for s, _ in pairs]
    tgt_corpus = [t for _, t in pairs]
    src_vocab = BPEVocab()
    tgt_vocab = BPEVocab()
    src_vocab.train(src_corpus, vocab_size=vocab_size)
    tgt_vocab.train(tgt_corpus, vocab_size=vocab_size)
    return src_vocab, tgt_vocab


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    model: Seq2SeqNMT,
    data: list[tuple[list[int], list[int]]],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device | None = None,
    accum_steps: int = 1,
    save_every: int = 0,
    save_path: str = "",
    src_vocab: BPEVocab | None = None,
    tgt_vocab: BPEVocab | None = None,
    raw_pairs: list[tuple[str, str]] | None = None,
    bpe_dropout: float = 0.0,
    epoch_callback=None,
    curriculum_epochs: int | None = None,
    source_contrast_weight: float = 0.0,
) -> None:
    """Train with AdamW, cosine LR, gradient accumulation, curriculum, and BPE dropout.

    - AdamW (Loshchilov & Hutter 2019): decoupled weight decay on non-embedding/norm params.
    - Sentence-length curriculum (Platanios et al. 2019): first 20% of epochs train on
      the shortest pairs only, then switch to full shuffled data. Helps early learning.
    - BPE dropout (Provilkov et al. 2020): if raw_pairs is provided, re-encodes each epoch
      with p=bpe_dropout so the model sees varied segmentations as data augmentation.
    """
    if device is None:
        device = next(model.parameters()).device

    if source_contrast_weight < 0:
        raise ValueError("source_contrast_weight must be nonnegative")
    contrast_rng = random.Random(
        90210
    )  # Keep batch shuffling independent of the auxiliary sampler.
    micro_size = max(1, batch_size // accum_steps)

    # AdamW: weight decay on weight matrices only, not biases/embeddings/layer norms
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if any(nd in name for nd in ("embed", "norm", "bias")):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total_steps = epochs * math.ceil(len(data) / batch_size)
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Sentence-length curriculum: sort ascending by (src+tgt) length for first 20% of epochs
    curriculum_epochs = max(1, epochs // 5) if curriculum_epochs is None else curriculum_epochs
    if curriculum_epochs < 0:
        raise ValueError("curriculum_epochs must be nonnegative")
    data_sorted = sorted(data, key=lambda p: len(p[0]) + len(p[1]))

    model.train()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # BPE dropout: re-encode raw pairs each epoch with stochastic segmentation
        if (
            raw_pairs is not None
            and bpe_dropout > 0.0
            and src_vocab is not None
            and tgt_vocab is not None
        ):
            epoch_data = [
                (
                    src_vocab.encode(s, add_bos=True, add_eos=True, bpe_dropout=bpe_dropout),
                    tgt_vocab.encode(t, add_bos=True, add_eos=True),
                )
                for s, t in raw_pairs
            ]
        elif epoch <= curriculum_epochs:
            epoch_data = data_sorted  # curriculum: short-to-long
        else:
            epoch_data = data

        epoch_data = list(epoch_data)
        if epoch > curriculum_epochs:
            random.shuffle(epoch_data)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(epoch_data), batch_size):
            batch = epoch_data[i : i + batch_size]
            optimizer.zero_grad()
            accum_loss = 0.0

            for k in range(0, len(batch), micro_size):
                micro = batch[k : k + micro_size]
                if not micro:
                    continue
                src_tensor, _, tgt_in, tgt_out, _ = collate_pairs(micro, pad_id=0)
                src_tensor = src_tensor.to(device)
                tgt_in = tgt_in.to(device)
                tgt_out = tgt_out.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(src_tensor, tgt_in)
                    n_micro = math.ceil(len(batch) / micro_size)
                    loss = nmt_loss(logits, tgt_out, pad_id=0) / n_micro
                    if source_contrast_weight > 0 and len(micro) > 1:
                        from speech_engine.model_objectives import conditional_contrast_loss

                        shift = contrast_rng.randrange(1, len(micro))
                        indices = torch.arange(len(micro), device=device).roll(shift)
                        wrong_src = src_tensor[indices]
                        valid = src_tensor.ne(wrong_src).any(1) & tgt_out.ne(tgt_out[indices]).any(
                            1
                        )
                        if valid.any():
                            wrong_logits = model(wrong_src, tgt_in)
                            loss = (
                                loss
                                + source_contrast_weight
                                * conditional_contrast_loss(logits, wrong_logits, tgt_out, valid)
                                / n_micro
                            )
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += accum_loss
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s"
            )

        if save_every > 0 and save_path and epoch % save_every == 0 and epoch < epochs:
            ckpt = save_path.replace(".pt", f"_ep{epoch}.pt")
            save_checkpoint(
                model, src_vocab, tgt_vocab, ckpt, meta={"direction": "es→en", "epoch": epoch}
            )
            print(f"  [checkpoint] saved → {ckpt}")

        if epoch_callback is not None:
            epoch_callback(epoch, model, avg_loss, optimizer, scheduler, scaler)
            model.train()


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model: Seq2SeqNMT,
    data: list[tuple[list[int], list[int]]],
    tgt_vocab: BPEVocab,
    beam: int,
    sample: int = 500,
) -> float:
    device = next(model.parameters()).device
    model.eval()
    exact = 0
    subset = random.sample(data, min(sample, len(data)))
    with torch.no_grad():
        for src_ids, tgt_ids in subset:
            src_t = torch.tensor([src_ids], device=device)
            out = model.translate(src_t, beam_size=beam)
            pred = tgt_vocab.decode(out).strip()
            gold = tgt_vocab.decode(tgt_ids[1:]).strip()  # strip BOS
            if pred == gold:
                exact += 1
    model.train()
    return exact / len(subset)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(CORPUS_PATH))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--vocab", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument(
        "--word_penalty",
        type=float,
        default=0.0,
        help="Beam search word penalty (negative = prefer shorter)",
    )
    p.add_argument(
        "--coverage_penalty",
        type=float,
        default=0.0,
        help="Beam search coverage penalty (0.1-0.4 typical)",
    )
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--no_mps", action="store_true", help="Disable MPS even if available")
    p.add_argument("--fast", action="store_true", help="Quick smoke-test (50 epochs, small model)")
    p.add_argument(
        "--translate",
        default=None,
        help="Translate a sentence (requires --load or existing checkpoint)",
    )
    p.add_argument("--load", default=None)
    p.add_argument("--out", default=str(ROOT / "data" / "models" / "nmt_es_en.pt"))
    p.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--max_pairs", type=int, default=100_000, help="Cap training pairs to avoid OOM")
    p.add_argument(
        "--save_every", type=int, default=25, help="Save checkpoint every N epochs (0 = disabled)"
    )
    p.add_argument("--dropout", type=float, default=0.15, help="Model dropout rate (default 0.15)")
    p.add_argument(
        "--bpe_dropout",
        type=float,
        default=0.1,
        help="BPE dropout p — stochastic segmentation during training (0 = off)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device selection: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
    elif not args.no_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if args.fast:
        args.epochs = 50
        args.d_model = 128
        args.n_layers = 2
        args.d_ff = 256
        args.vocab = 1500
        args.batch = 64
        args.accum = 1
        args.max_pairs = 10_000
        args.save_every = 0

    # ── Load-only / translate mode ────────────────────────────────────────────
    if args.translate and args.load:
        model, sv, tv, meta = load_checkpoint(args.load)
        model = model.to(device)
        src_ids = sv.encode(args.translate.lower())
        src_t = torch.tensor([src_ids], device=device)
        out = model.translate(
            src_t,
            beam_size=args.beam,
            word_penalty=args.word_penalty,
            coverage_penalty=args.coverage_penalty,
        )
        print(f"→ {tv.decode(out)}")
        return

    # ── Load corpora (curated + generated) ───────────────────────────────────
    print("Loading corpora…")
    raw_pairs = (
        load_corpus(Path(args.corpus)) if Path(args.corpus) != CORPUS_PATH else load_all_corpora()
    )
    print(f"  {len(raw_pairs)} unique sentence pairs loaded")

    # ── Train/test split (MUST happen before augmentation) ───────────────────
    print("\nCreating/loading train-test split…")
    train_raw, test_pairs = make_or_load_split(raw_pairs, src="es", tgt="en")
    train_raw_set = set(train_raw)

    # Augment curated pairs only (generated already diverse).
    # Cap the real-corpus contribution to stay within --max_pairs and avoid OOM
    # during BPE training and the encode step on large (440k+) datasets.
    # Only use training split — test set must never enter the training pipeline.
    curated = (
        load_corpus(CORPUS_PATH)
        if CORPUS_PATH.exists() and Path(args.corpus) == CORPUS_PATH
        else []
    )
    curated_train = [p for p in curated if p in train_raw_set]
    curated_train_set = set(curated_train)
    generated_extra = [p for p in train_raw if p not in curated_train_set]
    max_extra = max(0, args.max_pairs - 2 * len(curated_train))
    if len(generated_extra) > max_extra:
        random.shuffle(generated_extra)
        generated_extra = generated_extra[:max_extra]
        print(f"  Capped extra pairs to {max_extra:,} (--max_pairs={args.max_pairs:,})")
    pairs = augment(curated_train, factor=2) + generated_extra
    pairs = list(dict.fromkeys(pairs))  # dedup preserving order
    print(f"  {len(pairs):,} pairs after augmentation (test set strictly excluded)")

    # ── Build vocabularies ────────────────────────────────────────────────────
    # When resuming, load the checkpoint's vocab BEFORE encoding so token IDs
    # align with the loaded model's embedding table. Building a fresh vocab and
    # then overwriting it (old behaviour) caused a mismatch: pairs were encoded
    # with new IDs but the model embedded them with the old IDs → random input.
    print(
        f"\nBuilding Seq2SeqNMT  d_model={args.d_model}  layers={args.n_layers}  heads={args.n_heads}  d_ff={args.d_ff}"
    )
    if args.load and Path(args.load).exists():
        print(f"  Resuming from {args.load} — loading checkpoint vocab…")
        model, src_vocab, tgt_vocab, _ = load_checkpoint(args.load)
        print(f"  src vocab: {src_vocab.vocab_size}  tgt vocab: {tgt_vocab.vocab_size}")
    else:
        print(f"\nTraining BPE vocabularies (vocab_size={args.vocab})…")
        src_vocab, tgt_vocab = build_vocabs(pairs, args.vocab)
        print(f"  src vocab: {src_vocab.vocab_size}  tgt vocab: {tgt_vocab.vocab_size}")

    # ── Encode all pairs ──────────────────────────────────────────────────────
    PAD, BOS, EOS = 0, 1, 2
    print("Encoding pairs…")
    encoded = [
        (
            src_vocab.encode(s, add_bos=True, add_eos=True),
            tgt_vocab.encode(t, add_bos=True, add_eos=True),
        )
        for s, t in pairs
    ]
    print(f"  {len(encoded):,} pairs encoded")

    # ── Build model (fresh-train path only) ───────────────────────────────────
    if not (args.load and Path(args.load).exists()):
        model = Seq2SeqNMT(
            src_vocab_size=src_vocab.vocab_size,
            tgt_vocab_size=tgt_vocab.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    # ── Train ─────────────────────────────────────────────────────────────────
    eff_batch = args.batch  # micro_size = batch // accum, effective batch = batch
    print(
        f"\nTraining for {args.epochs} epochs  batch={eff_batch}  accum={args.accum}  lr={args.lr}…"
    )
    print(f"  (micro-batch = {eff_batch // args.accum}, effective batch = {eff_batch})")
    train(
        model,
        encoded,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        accum_steps=args.accum,
        save_every=args.save_every,
        save_path=args.out,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        raw_pairs=pairs if args.bpe_dropout > 0 else None,
        bpe_dropout=args.bpe_dropout,
    )

    # ── Evaluate (on a random sample — full eval on 35k takes too long) ───────
    acc = evaluate(model, encoded, tgt_vocab, beam=args.beam, sample=500)
    print(f"\nTraining-set exact-match on up to 500 pairs (not generalization): {acc * 100:.1f}%")

    # ── Test sentences ────────────────────────────────────────────────────────
    model.eval()
    print("\nTest translations:")
    with torch.no_grad():
        for sent in _TEST_SENTENCES:
            src_ids = src_vocab.encode(sent.lower())
            if not src_ids:
                continue
            src_t = torch.tensor([src_ids], device=device)
            out = model.translate(src_t, beam_size=args.beam)
            pred = tgt_vocab.decode(out)
            print(f"  [{sent}]")
            print(f"   → {pred}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        src_vocab,
        tgt_vocab,
        out_path,
        meta={"direction": "es→en", "pairs": len(pairs), "n_train": len(pairs)},
    )
    print(f"\nSaved → {out_path}")

    # ── BLEU evaluation on held-out test set ─────────────────────────────────
    if not args.fast and test_pairs:
        import json
        from datetime import datetime, timezone

        print(f"\nEvaluating on {len(test_pairs):,} held-out test pairs…")
        eval_result = run_bleu_eval(
            model,
            src_vocab,
            tgt_vocab,
            test_pairs,
            device,
            beam=args.beam,
            n_samples=25,
        )
        bleu = eval_result["bleu"]
        print(f"\n{'=' * 55}")
        print(f"  ES → EN BLEU: {bleu:.2f}")
        print(f"  Precision (1-4): {' / '.join(f'{p:.1f}' for p in eval_result['precisions'])}")
        print(
            f"  BP: {eval_result['bp']:.4f}  |  Hyp len: {eval_result['hyp_len']:,}  |  Ref len: {eval_result['ref_len']:,}"
        )
        print(f"  Test examples: {eval_result['n_sentences']:,}")
        print(f"{'=' * 55}")

        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        report_path = results_dir / "eval_es_en.json"
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "src_lang": "es",
            "tgt_lang": "en",
            "checkpoint": str(out_path),
            "n_train": len(pairs),
            "n_test": len(test_pairs),
            "beam_size": args.beam,
            **eval_result,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report → {report_path}")

    # ── Translate CLI arg if given ────────────────────────────────────────────
    if args.translate:
        src_ids = src_vocab.encode(args.translate.lower())
        src_t = torch.tensor([src_ids], device=device)
        out = model.translate(src_t, beam_size=args.beam)
        print(f"\n[{args.translate}]\n→ {tgt_vocab.decode(out)}")


if __name__ == "__main__":
    main()
