"""
scripts/filter_corpus.py — Corpus quality filtering for NMT training data.

Removes noisy, misaligned, or low-quality sentence pairs from a TSV corpus
using a combination of:

  1. Length-ratio filter — rejects pairs where src/tgt token count ratio is
     extreme (> 3× or < 1/3×). Highly mismatched lengths usually indicate
     misalignment or padding errors.

  2. Character n-gram language overlap — rejects pairs that share too many
     character 3-grams with their source, which catches untranslated copies,
     language ID errors, or code-mixed inputs that aren't real translations.

  3. Deduplication — removes exact duplicate source sentences.

  4. Model perplexity scoring (optional, --checkpoint) — if a trained NMT
     checkpoint is provided, scores every pair by teacher-forced loss and
     removes the top-p% highest-loss pairs (likely misalignments) and the
     bottom-p% lowest-loss pairs (trivial / repeated patterns that can
     cause overfitting).  This is the simplified OPUS-MT / Junczys-Dowmunt
     et al. (2018) "dual cross-entropy" approach using only the forward model.

Usage:
    # Heuristic-only filtering of a corpus:
    .venv/bin/python3 scripts/filter_corpus.py \\
        --input data/corpora/es_en_real.tsv \\
        --output data/corpora/es_en_filtered.tsv

    # With model-perplexity scoring (uses a trained checkpoint):
    .venv/bin/python3 scripts/filter_corpus.py \\
        --input data/corpora/es_en_real.tsv \\
        --output data/corpora/es_en_filtered.tsv \\
        --checkpoint data/models/nmt_es_en_ep125.pt \\
        --remove_high_loss 0.10 \\
        --remove_low_loss 0.03

    # Multilingual: filter each language corpus separately
    for lang in es fr de; do
        .venv/bin/python3 scripts/filter_corpus.py \\
            --input data/corpora/${lang}_en_real.tsv \\
            --output data/corpora/${lang}_en_filtered.tsv
    done
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Heuristic filters ─────────────────────────────────────────────────────────

def _char_trigrams(text: str) -> set[str]:
    t = text.replace(" ", "_")
    return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else set()


def heuristic_ok(
    src: str,
    tgt: str,
    max_len_ratio: float = 3.0,
    max_overlap: float = 0.5,
    min_tokens: int = 2,
    max_tokens: int = 100,
) -> bool:
    """Return True if the pair passes all heuristic checks."""
    src_toks = src.split()
    tgt_toks = tgt.split()

    # Too short or too long
    if len(src_toks) < min_tokens or len(tgt_toks) < min_tokens:
        return False
    if len(src_toks) > max_tokens or len(tgt_toks) > max_tokens:
        return False

    # Extreme length ratio
    ratio = len(src_toks) / max(1, len(tgt_toks))
    if ratio > max_len_ratio or ratio < 1.0 / max_len_ratio:
        return False

    # High character n-gram overlap between source and target (untranslated pair)
    src_3g = _char_trigrams(src)
    tgt_3g = _char_trigrams(tgt)
    if src_3g and tgt_3g:
        overlap = len(src_3g & tgt_3g) / min(len(src_3g), len(tgt_3g))
        if overlap > max_overlap:
            return False

    return True


# ── Model perplexity scoring ──────────────────────────────────────────────────

def score_pairs_by_loss(
    pairs: list[tuple[str, str]],
    checkpoint: str,
    batch_size: int = 64,
) -> list[float]:
    """Return per-pair teacher-forced loss under the given checkpoint.

    Lower loss = pair is easy/well-aligned for this model.
    Higher loss = pair is hard/noisy/misaligned.
    """
    from speech_engine.bpe import PAD_ID
    from speech_engine.nmt import collate_pairs, load_checkpoint, nmt_loss

    print(f"  Loading checkpoint: {checkpoint}")
    model, src_vocab, tgt_vocab, _ = load_checkpoint(checkpoint)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Scoring on {device}")
    model = model.to(device)
    model.eval()

    # Encode all pairs
    encoded = []
    for src, tgt in pairs:
        src_ids = src_vocab.encode(src, add_bos=True, add_eos=True)
        tgt_ids = tgt_vocab.encode(tgt, add_bos=True, add_eos=True)
        if src_ids and tgt_ids:
            encoded.append((src_ids, tgt_ids))
        else:
            encoded.append(([1, 2], [1, 2]))  # placeholder

    losses: list[float] = []
    with torch.no_grad():
        for i in range(0, len(encoded), batch_size):
            batch = encoded[i : i + batch_size]
            src_t, _, tgt_in, tgt_out, _ = collate_pairs(batch, pad_id=PAD_ID)
            src_t   = src_t.to(device)
            tgt_in  = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            logits = model(src_t, tgt_in)

            # Per-sample loss: compute loss for each example individually
            B = src_t.size(0)
            for b in range(B):
                mask = tgt_out[b] != PAD_ID
                if mask.sum() == 0:
                    losses.append(float("inf"))
                    continue
                import torch.nn.functional as F
                V = logits.size(-1)
                lp = F.log_softmax(logits[b], dim=-1)  # (T, V)
                nll = -lp[torch.arange(tgt_out.size(1)), tgt_out[b].clamp(min=0)]
                losses.append(nll[mask].mean().item())

            if (i // batch_size) % 50 == 0:
                print(f"  Scored {min(i + batch_size, len(encoded)):,}/{len(encoded):,} pairs", end="\r")

    print()
    return losses


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",            required=True, help="Input TSV corpus (src\\ttgt per line)")
    p.add_argument("--output",           required=True, help="Output filtered TSV")
    p.add_argument("--max_len_ratio",    type=float, default=3.0,  help="Max token length ratio src/tgt")
    p.add_argument("--max_overlap",      type=float, default=0.5,  help="Max character 3-gram overlap fraction")
    p.add_argument("--min_tokens",       type=int,   default=2,    help="Min tokens per side")
    p.add_argument("--max_tokens",       type=int,   default=100,  help="Max tokens per side")
    p.add_argument("--checkpoint",       default=None, help="NMT checkpoint for perplexity scoring")
    p.add_argument("--remove_high_loss", type=float, default=0.10, help="Remove top-p fraction by loss (noisy)")
    p.add_argument("--remove_low_loss",  type=float, default=0.03, help="Remove bottom-p fraction by loss (trivial)")
    p.add_argument("--batch_size",       type=int,   default=64)
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    # Load
    print(f"Loading {in_path}…")
    all_pairs: list[tuple[str, str]] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            all_pairs.append((parts[0].strip().lower(), parts[1].strip().lower()))
    print(f"  {len(all_pairs):,} pairs loaded")

    # Heuristic filtering
    print(f"\nApplying heuristic filters (ratio≤{args.max_len_ratio}, overlap≤{args.max_overlap}, {args.min_tokens}–{args.max_tokens} tokens)…")
    heuristic_ok_pairs = [
        (s, t) for s, t in all_pairs
        if heuristic_ok(s, t,
                        max_len_ratio=args.max_len_ratio,
                        max_overlap=args.max_overlap,
                        min_tokens=args.min_tokens,
                        max_tokens=args.max_tokens)
    ]
    print(f"  {len(heuristic_ok_pairs):,} pairs pass heuristics ({len(all_pairs) - len(heuristic_ok_pairs):,} removed)")

    # Deduplication on source
    print(f"\nDeduplicating on source…")
    seen_src: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for s, t in heuristic_ok_pairs:
        if s not in seen_src:
            seen_src.add(s)
            deduped.append((s, t))
    print(f"  {len(deduped):,} unique pairs ({len(heuristic_ok_pairs) - len(deduped):,} duplicates removed)")

    filtered = deduped

    # Optional perplexity scoring
    if args.checkpoint:
        print(f"\nScoring {len(filtered):,} pairs by model perplexity…")
        losses = score_pairs_by_loss(filtered, args.checkpoint, batch_size=args.batch_size)

        n = len(filtered)
        n_remove_high = int(n * args.remove_high_loss)
        n_remove_low  = int(n * args.remove_low_loss)

        # Sort by loss; remove top (noisy) and bottom (trivial) fractions
        indexed = sorted(enumerate(losses), key=lambda x: x[1])
        keep_indices = set(
            i for rank, (i, _) in enumerate(indexed)
            if n_remove_low <= rank < n - n_remove_high
        )
        filtered = [filtered[i] for i in range(n) if i in keep_indices]
        print(f"  Removed {n_remove_high:,} high-loss (noisy) and {n_remove_low:,} low-loss (trivial) pairs")
        print(f"  {len(filtered):,} pairs remain after perplexity filtering")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Filtered corpus — {len(filtered):,} pairs from {in_path.name}\n")
        for s, t in filtered:
            f.write(f"{s}\t{t}\n")
    print(f"\nSaved → {out_path}  ({len(filtered):,} pairs)")

    reduction = (1 - len(filtered) / max(1, len(all_pairs))) * 100
    print(f"Total reduction: {reduction:.1f}% of original corpus removed")


if __name__ == "__main__":
    main()
