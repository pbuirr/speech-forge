"""
scripts/translation/backtranslate.py — Generate synthetic parallel data via back-translation.

Back-translation (Sennrich et al. 2016) is one of the most effective ways
to improve NMT quality with no new human-labeled data:

  1. Take a large monolingual corpus in the TARGET language (e.g. English)
  2. Translate it into the SOURCE language using a REVERSE model (EN→ES)
  3. Treat the resulting (synthetic_source, real_target) pairs as training data

Why it works: the model sees many more target-side sentences in natural
language, improving fluency. The synthetic source is noisy (the reverse model
isn't perfect) but that noise acts as a form of regularization. Google saw
+2-4 BLEU from back-translation on WMT tasks.

Usage:
    # Translate English text into synthetic Spanish (requires EN→ES model)
    python scripts/translation/backtranslate.py \\
        --src en --tgt es \\
        --input data/monolingual_en.txt \\
        --output data/synthetic_en_es.tsv \\
        --checkpoint data/models/nmt_en_es.pt \\
        --limit 100000

    # To use the synthetic data in training, merge with the real corpus:
    #   cat data/corpora/es_en_real.tsv data/synthetic_en_es.tsv > data/combined.tsv
    # Then train with: python scripts/translation/train_nmt.py --corpus data/combined.tsv

Output format (TSV):
    # SYNTHETIC — back-translated from EN using checkpoint nmt_en_es.pt
    synthetic_spanish_sentence<TAB>original_english_sentence

The SYNTHETIC comment marks synthetic pairs so you can always distinguish
them from real human-translated data.

Reference:
    Sennrich, Haddow, Birch (2016). "Improving Neural Machine Translation
    Models with Monolingual Data." ACL 2016.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_engine.nmt import load_checkpoint


# ── Monolingual corpus loader ─────────────────────────────────────────────────

def load_monolingual(path: Path, limit: int | None = None) -> list[str]:
    """
    Load one sentence per line from a plain-text monolingual file.
    Skips blank lines and lines starting with #.
    """
    sentences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sentences.append(line)
        if limit and len(sentences) >= limit:
            break
    return sentences


# ── Batch translation ─────────────────────────────────────────────────────────

def translate_sentences(
    model,
    src_vocab,
    tgt_vocab,
    sentences: list[str],
    device: torch.device,
    beam: int = 4,
    max_len: int = 100,
    batch_report_every: int = 1000,
) -> list[str]:
    """Translate a list of sentences one at a time. Returns decoded outputs."""
    results: list[str] = []
    model.eval()
    t0 = time.time()

    with torch.no_grad():
        for i, sent in enumerate(sentences):
            src_ids = src_vocab.encode(sent.lower())
            if not src_ids:
                results.append("")
                continue
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            out_ids = model.translate(src_t, beam_size=beam, max_len=max_len)
            results.append(tgt_vocab.decode(out_ids).strip())

            if (i + 1) % batch_report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(sentences) - i - 1) / max(rate, 1e-9)
                print(f"  {i+1:,}/{len(sentences):,}  "
                      f"({rate:.1f} sent/s, ~{remaining/60:.1f}min remaining)")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if not args.no_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"\n[error] Checkpoint not found: {ckpt_path}")
        print(
            f"  Back-translation from {args.src}→{args.tgt} requires a trained REVERSE model.\n"
            f"  To train it, flip the corpus:\n"
            f"    python scripts/translation/train_nmt.py \\\n"
            f"      --corpus data/corpora/{args.tgt}_{args.src}_real.tsv \\\n"
            f"      --out data/models/nmt_{args.src}_{args.tgt}.pt\n"
            f"\n  Note: the --corpus flag in phase7_train.py may need updating\n"
            f"  to support non-ES source languages. See scripts/translation/train_nmt.py."
        )
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    model, src_vocab, tgt_vocab, meta = load_checkpoint(ckpt_path)
    model = model.to(device)
    print(f"  {model.param_count():,} parameters  |  direction: {meta.get('direction','?')}")

    # ── Load monolingual input ─────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n[error] Input file not found: {input_path}")
        print(
            f"  Provide a plain-text file with one {args.src} sentence per line.\n"
            f"  Any large monolingual {args.src} corpus works: Wikipedia, news, subtitles.\n"
            f"  The English side of your OPUS corpus is also a valid source:\n"
            f"    cut -f2 data/corpora/es_en_real.tsv > data/monolingual_en.txt"
        )
        sys.exit(1)

    print(f"Loading monolingual corpus: {input_path} (limit={args.limit or 'none'})")
    sentences = load_monolingual(input_path, limit=args.limit)
    print(f"  {len(sentences):,} sentences to translate")

    # ── Translate ──────────────────────────────────────────────────────────────
    print(f"\nTranslating {args.src}→{args.tgt} with beam={args.beam}…")
    t0 = time.time()
    translations = translate_sentences(
        model, src_vocab, tgt_vocab, sentences, device,
        beam=args.beam, max_len=args.max_len,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({len(sentences)/elapsed:.1f} sent/s)")

    # ── Write output ───────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_empty   = 0
    with output_path.open("w", encoding="utf-8") as f:
        # Header marks synthetic data
        f.write(f"# SYNTHETIC — back-translated from {args.src} using {ckpt_path.name}\n")
        f.write(f"# Source: {input_path}  |  {len(sentences):,} sentences\n")
        f.write(f"# Format: synthetic_{args.tgt}<TAB>original_{args.src}\n")
        for src_sent, tgt_sent in zip(sentences, translations):
            if not tgt_sent.strip():
                n_empty += 1
                continue
            # Output as (synthetic_tgt, real_src) TSV
            # This can be merged with the real corpus for ES→EN training:
            # the synthetic tgt becomes the "source" and real src becomes "target"
            # for the back-translation augmented training direction.
            f.write(f"{tgt_sent}\t{src_sent}\n")
            n_written += 1

    print(f"\nWrote {n_written:,} pairs → {output_path}")
    if n_empty:
        print(f"  ({n_empty} empty translations skipped)")

    print(f"\nTo merge with real training data:")
    print(f"  cat data/corpora/{args.tgt}_{args.src}_real.tsv {output_path} > data/combined_{args.tgt}_{args.src}.tsv")
    print(f"  python scripts/translation/train_nmt.py  # (update REAL_CORPUS_PATH to point to combined file)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic parallel data via back-translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src",        default="en", help="Language of the input (monolingual) file")
    p.add_argument("--tgt",        default="es", help="Language to translate INTO (the reverse model's output)")
    p.add_argument("--input",      default="data/monolingual_en.txt",
                   help="Plain-text monolingual corpus (one sentence per line)")
    p.add_argument("--output",     default="data/synthetic_en_es.tsv",
                   help="Output TSV: synthetic_tgt<TAB>real_src")
    p.add_argument("--checkpoint", default="data/models/nmt_en_es.pt",
                   help="Reverse model checkpoint (e.g. EN→ES for back-translating into ES)")
    p.add_argument("--limit",      type=int, default=100_000,
                   help="Max sentences to translate (default 100k)")
    p.add_argument("--beam",       type=int, default=4)
    p.add_argument("--max-len",    type=int, default=100, dest="max_len")
    p.add_argument("--no-mps",     action="store_true", dest="no_mps")
    return p.parse_args()


if __name__ == "__main__":
    main()
