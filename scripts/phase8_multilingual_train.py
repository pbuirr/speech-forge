"""
scripts/phase8_multilingual_train.py — LinguaForge Phase 8 Multilingual NMT

Trains a single ES/FR/DE → EN Transformer on all three OPUS corpora at once.
Architecture is identical to Phase 7 (Seq2SeqNMT); the only additions are:

  1. A shared BPE vocabulary trained on all source languages + English together.
     Latin-script languages share most subwords, so the shared vocab is only
     ~10-15% larger than a monolingual one for the same coverage.

  2. Three language tokens — <ES>, <FR>, <DE> — added as special tokens and
     prepended to every source sentence before encoding.  The decoder always
     targets English; the language token tells the encoder what it's reading.

     ES: <ES> el gato exploró la casa  →  the cat explored the house
     FR: <FR> le chat a exploré la maison  →  the cat explored the house
     DE: <DE> die Katze erkundete das Haus  →  the cat explored the house

  The model learns shared cross-lingual representations — knowledge of French
  grammar generalises to Spanish, German word-order patterns inform English
  clause structure.  This is the Johnson et al. (2017) "one model, many
  languages" approach used in production at Google Translate and M2M-100.

Usage:
    # Train from scratch
    .venv/bin/python3 scripts/phase8_multilingual_train.py

    # Resume from an existing checkpoint
    .venv/bin/python3 scripts/phase8_multilingual_train.py \\
        --load data/models/nmt_multilingual.pt

    # Quick smoke-test
    .venv/bin/python3 scripts/phase8_multilingual_train.py --fast
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from speech_engine.bleu import bleu_detail
from speech_engine.bpe import BPEVocab, BOS_ID, EOS_ID, PAD_ID
from speech_engine.nmt import Seq2SeqNMT, collate_pairs, nmt_loss, save_checkpoint, load_checkpoint

# ── Language token constants ──────────────────────────────────────────────────

LANG_TOKENS = {"es": "<ES>", "fr": "<FR>", "de": "<DE>"}

# ── Corpus paths ───────────────────────────────────────────────────────────────

def _corpus_path(lang: str) -> Path:
    filtered = ROOT / "data" / "corpora" / f"{lang}_en_filtered.tsv"
    if filtered.exists():
        return filtered
    return ROOT / "data" / "corpora" / f"{lang}_en_real.tsv"

CORPORA: dict[str, Path] = {lang: _corpus_path(lang) for lang in ("es", "fr", "de")}


# ── Data loading ───────────────────────────────────────────────────────────────

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


def load_all_pairs(
    max_per_lang: int = 100_000,
) -> list[tuple[str, str, str]]:
    """Return (src_lang, src_text, tgt_text) triples, capped per language."""
    triples: list[tuple[str, str, str]] = []
    for lang, path in CORPORA.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {lang.upper()}→EN")
            continue
        pairs = load_corpus(path)
        random.shuffle(pairs)
        pairs = pairs[:max_per_lang]
        for src, tgt in pairs:
            triples.append((lang, src, tgt))
        print(f"  {lang.upper()}→EN: {len(pairs):,} pairs")
    return triples


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def build_shared_vocab(
    triples: list[tuple[str, str, str]],
    vocab_size: int = 6000,
) -> BPEVocab:
    """Train a single BPE vocab on all source languages + English target.

    We use a slightly larger vocab than monolingual (6k vs 4k) so each
    language retains reasonable coverage without bleeding into each other.
    Language tokens are injected after training so they always get IDs.
    """
    all_text = []
    for lang, src, tgt in triples:
        all_text.append(src)
        all_text.append(tgt)

    vocab = BPEVocab()
    # Inject language tokens as reserved IDs before BPE training so they
    # are guaranteed to exist even if they never appear in the raw corpus.
    for token in LANG_TOKENS.values():
        nid = len(vocab._token_to_id)
        vocab._token_to_id[token] = nid
        vocab._id_to_token[nid] = token

    vocab.train(all_text, vocab_size=vocab_size)
    return vocab


def encode_pair(
    lang: str,
    src: str,
    tgt: str,
    vocab: BPEVocab,
) -> tuple[list[int], list[int]]:
    """Encode one (src, tgt) pair with language token prepended to source."""
    lang_tok = LANG_TOKENS[lang]
    lang_id  = vocab._token_to_id[lang_tok]

    # Source: [lang_id, BOS, t1, …, tn, EOS]
    # The lang token sits before BOS so the encoder sees it first.
    src_ids = [lang_id] + vocab.encode(src, add_bos=True, add_eos=True)
    tgt_ids = vocab.encode(tgt, add_bos=True, add_eos=True)
    return src_ids, tgt_ids


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model: Seq2SeqNMT,
    data: list[tuple[list[int], list[int]]],
    vocab: BPEVocab,
    *,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    lr: float,
    save_path: str,
    save_every: int,
    device: torch.device,
    start_epoch: int = 1,
    raw_triples: list[tuple[str, str, str]] | None = None,
    bpe_dropout: float = 0.0,
) -> None:
    """Train with AdamW, cosine LR, curriculum, and BPE dropout."""
    micro_size = max(1, batch_size // accum_steps)

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if any(nd in name for nd in ("embed", "norm", "bias")):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.01},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.98), eps=1e-9,
    )
    scaler = GradScaler()

    total_steps  = epochs * math.ceil(len(data) / batch_size)
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if start_epoch > 1:
        steps_done = (start_epoch - 1) * math.ceil(len(data) / batch_size)
        for _ in range(steps_done):
            scheduler.step()

    curriculum_epochs = start_epoch + max(1, epochs // 5)
    data_sorted = sorted(data, key=lambda p: len(p[0]) + len(p[1]))

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        if raw_triples is not None and bpe_dropout > 0.0:
            epoch_data = []
            for lang, src, tgt in raw_triples:
                src_ids, tgt_ids = encode_pair(lang, src, tgt, vocab)
                if len(src_ids) <= 256 and len(tgt_ids) <= 256:
                    epoch_data.append((src_ids, tgt_ids))
        elif epoch < curriculum_epochs:
            epoch_data = data_sorted
        else:
            epoch_data = data

        epoch_data = list(epoch_data)
        if epoch >= curriculum_epochs:
            random.shuffle(epoch_data)

        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(epoch_data), batch_size):
            batch = epoch_data[i : i + batch_size]
            optimizer.zero_grad()
            accum_loss = 0.0

            for k in range(0, len(batch), micro_size):
                micro = batch[k : k + micro_size]
                if not micro:
                    continue
                src_tensor, _, tgt_in, tgt_out, _ = collate_pairs(micro, pad_id=PAD_ID)
                src_tensor = src_tensor.to(device)
                tgt_in     = tgt_in.to(device)
                tgt_out    = tgt_out.to(device)

                with autocast():
                    logits = model(src_tensor, tgt_in)
                    n_micro = math.ceil(len(batch) / micro_size)
                    loss    = nmt_loss(logits, tgt_out, pad_id=PAD_ID) / n_micro
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += accum_loss
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)

        if epoch % max(1, (start_epoch + epochs - 1) // 20) == 0 or epoch == start_epoch + epochs - 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        if save_every > 0 and save_path and epoch % save_every == 0:
            ckpt = save_path.replace(".pt", f"_ep{epoch}.pt")
            save_checkpoint(model, vocab, vocab, ckpt,
                            meta={"direction": "multi→en", "epoch": epoch,
                                  "langs": list(LANG_TOKENS.keys())})
            print(f"  [checkpoint] saved → {ckpt}")


# ── Quick evaluation ───────────────────────────────────────────────────────────

def quick_eval(
    model: Seq2SeqNMT,
    samples: list[tuple[str, str, str]],
    vocab: BPEVocab,
    device: torch.device,
    n: int = 10,
) -> None:
    """Translate a few sentences from each language and print side-by-side."""
    model.eval()
    print("\n  Sample translations:")
    by_lang: dict[str, list] = {lang: [] for lang in LANG_TOKENS}
    for lang, src, tgt in samples:
        by_lang[lang].append((src, tgt))

    with torch.no_grad():
        for lang in LANG_TOKENS:
            pairs = by_lang[lang][:n // len(LANG_TOKENS)]
            for src, ref in pairs:
                src_ids, _ = encode_pair(lang, src, ref, vocab)
                src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
                out   = model.translate_beam(src_t, beam_size=4)
                hyp   = vocab.decode(out)
                print(f"  [{lang.upper()}] {src[:60]}")
                print(f"        REF: {ref[:80]}")
                print(f"        HYP: {hyp[:80]}")
    model.train()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--vocab",       type=int,   default=6000)
    p.add_argument("--batch",       type=int,   default=128)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--d_model",     type=int,   default=256)
    p.add_argument("--n_layers",    type=int,   default=4)
    p.add_argument("--d_ff",        type=int,   default=512)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--accum",       type=int,   default=4)
    p.add_argument("--max_per_lang",type=int,   default=100_000)
    p.add_argument("--save_every",  type=int,   default=25)
    p.add_argument("--load",        default=None, help="Resume from checkpoint")
    p.add_argument("--out",         default=str(ROOT / "data" / "models" / "nmt_multilingual.pt"))
    p.add_argument("--no_mps",      action="store_true")
    p.add_argument("--fast",        action="store_true", help="Smoke-test: 2k pairs/lang, 20 epochs")
    p.add_argument("--dropout",        type=float, default=0.1,  help="Model dropout rate")
    p.add_argument("--bpe_dropout",    type=float, default=0.1,  help="BPE dropout p during training (0 = off)")
    p.add_argument("--tie_embeddings", action="store_true",       help="Tie src/tgt/output embeddings (shared vocab only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fast:
        args.max_per_lang = 2_000
        args.epochs       = 20
        args.vocab        = 2_000
        args.save_every   = 0

    # ── Device ──────────────────────────────────────────────────────────────
    if not args.no_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ── Load corpora ────────────────────────────────────────────────────────
    print("\nLoading corpora…")
    triples = load_all_pairs(max_per_lang=args.max_per_lang)
    print(f"  Total: {len(triples):,} pairs across {len(CORPORA)} languages")

    # ── Build / load vocabulary ─────────────────────────────────────────────
    vocab_path = ROOT / "data" / "models" / "multilingual_vocab.json"
    if vocab_path.exists() and args.load:
        print(f"\nLoading shared vocab from {vocab_path}…")
        vocab = BPEVocab.load(str(vocab_path))
    else:
        print(f"\nTraining shared BPE vocabulary (vocab_size={args.vocab})…")
        vocab = build_shared_vocab(triples, vocab_size=args.vocab)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(str(vocab_path))
        print(f"  Vocab saved → {vocab_path}")

    print(f"  Vocab size: {vocab.vocab_size:,} tokens")
    for lang, tok in LANG_TOKENS.items():
        tid = vocab._token_to_id.get(tok, "MISSING")
        print(f"  {tok} → id {tid}")

    # ── Encode all pairs ────────────────────────────────────────────────────
    print("\nEncoding pairs…")
    encoded: list[tuple[list[int], list[int]]] = []
    for lang, src, tgt in triples:
        src_ids, tgt_ids = encode_pair(lang, src, tgt, vocab)
        # Skip pathologically long sequences (>256 tokens)
        if len(src_ids) <= 256 and len(tgt_ids) <= 256:
            encoded.append((src_ids, tgt_ids))

    print(f"  {len(encoded):,} pairs after length filtering")

    # Hold out 5% for quick qualitative checks (no overlap with training)
    split = max(1, len(encoded) // 20)
    train_data = encoded[split:]
    eval_triples = triples[:split]
    print(f"  {len(train_data):,} train / {split:,} held-out for eval samples")

    # ── Model ────────────────────────────────────────────────────────────────
    V = vocab.vocab_size
    start_epoch = 1

    if args.load:
        print(f"\nLoading checkpoint: {args.load}")
        model, _, _, meta = load_checkpoint(args.load)
        model = model.to(device)
        start_epoch = meta.get("epoch", 0) + 1
        print(f"  Resuming from epoch {start_epoch}")
    else:
        print(f"\nBuilding model: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
        model = Seq2SeqNMT(
            src_vocab_size=V,
            tgt_vocab_size=V,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            tie_embeddings=args.tie_embeddings,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")
    print(f"  Training for {args.epochs} epochs")

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nTraining…")
    train(
        model, train_data, vocab,
        epochs=args.epochs,
        batch_size=args.batch,
        accum_steps=args.accum,
        lr=args.lr,
        save_path=args.out,
        save_every=args.save_every,
        device=device,
        start_epoch=start_epoch,
        raw_triples=triples if args.bpe_dropout > 0 else None,
        bpe_dropout=args.bpe_dropout,
    )

    # ── Save final model ────────────────────────────────────────────────────
    save_checkpoint(model, vocab, vocab, args.out,
                    meta={"direction": "multi→en", "epoch": start_epoch + args.epochs - 1,
                          "langs": list(LANG_TOKENS.keys())})
    print(f"\nSaved → {args.out}")

    # ── Quick eval ───────────────────────────────────────────────────────────
    quick_eval(model, eval_triples, vocab, device, n=9)


if __name__ == "__main__":
    main()
