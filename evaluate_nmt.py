"""
evaluate_nmt.py — Standalone BLEU evaluation for LinguaForge NMT models.

Loads a trained checkpoint, translates the held-out test set (created once
at training time with a fixed seed), computes corpus BLEU, saves a detailed
JSON report, and optionally renders an attention heatmap.

Usage:
    # Basic evaluation (uses auto-detected test split)
    python evaluate_nmt.py --src es --tgt en --checkpoint data/models/nmt_es_en.pt

    # With attention heatmap for a specific sentence
    python evaluate_nmt.py --src es --tgt en --checkpoint data/models/nmt_es_en.pt \\
        --attention "el gato curioso exploró la casa vieja"

    # Evaluate on a custom test file (TSV: source\\treference)
    python evaluate_nmt.py --src es --tgt en --checkpoint data/models/nmt_es_en.pt \\
        --test data/splits/es_en_test_seed42.tsv

    # Evaluate with named-entity protection enabled
    python evaluate_nmt.py --src es --tgt en --checkpoint data/models/nmt_es_en.pt \\
        --protect-entities

Report saved to: results/eval_<src>_<tgt>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from speech_engine.bleu import bleu_detail, sentence_bleu
from speech_engine.nmt import load_checkpoint
from speech_engine.ner import EntityProtector


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BEAM       = 4
DEFAULT_MAX_LEN    = 100
DEFAULT_N_SAMPLES  = 40      # sample translations saved to JSON
DEFAULT_SPLIT_SEED = 42


# ── Test-set loader ────────────────────────────────────────────────────────────

def load_test_set(path: Path) -> list[tuple[str, str]]:
    """Load a TSV test file: source\\treference. Skips comment lines."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def find_test_split(src: str, tgt: str, seed: int = DEFAULT_SPLIT_SEED) -> Path | None:
    """Find the test split file created during training."""
    candidates = [
        ROOT / "data" / "splits" / f"{src}_{tgt}_test_seed{seed}.tsv",
        ROOT / "data" / "splits" / f"{tgt}_{src}_test_seed{seed}.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── Translation ────────────────────────────────────────────────────────────────

def translate_batch(
    model,
    src_vocab,
    tgt_vocab,
    sentences: list[str],
    device: torch.device,
    beam: int = DEFAULT_BEAM,
    max_len: int = DEFAULT_MAX_LEN,
    protect_entities: bool = False,
    src_lang: str = "?",
) -> list[str]:
    """Translate a list of source sentences. Returns decoded target strings."""
    results = []
    model.eval()
    ep = EntityProtector(src_lang=src_lang) if protect_entities else None

    with torch.no_grad():
        for sent in sentences:
            # Optional NER masking
            masked, entities = (ep.mask(sent) if ep else (sent, []))

            src_ids = src_vocab.encode(masked.lower())
            if not src_ids:
                results.append("")
                continue
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            out_ids = model.translate(src_t, beam_size=beam, max_len=max_len)
            pred = tgt_vocab.decode(out_ids).strip()

            # Restore named entities
            if ep and entities:
                pred = ep.restore(pred, entities)

            results.append(pred)
    return results


# ── Attention visualization ────────────────────────────────────────────────────

def visualize_attention(
    model,
    src_vocab,
    tgt_vocab,
    sentence: str,
    device: torch.device,
    out_path: Path,
    beam: int = DEFAULT_BEAM,
) -> None:
    """
    Render a cross-attention heatmap and save as PNG.

    X-axis: source BPE tokens
    Y-axis: generated target BPE tokens
    Color:  attention weight (brighter = more attention)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  [warn] matplotlib not installed — skipping attention plot")
        return

    model.eval()
    src_ids = src_vocab.encode(sentence.lower())
    if not src_ids:
        print("  [warn] Could not encode sentence for attention visualization")
        return

    src_t = torch.tensor([src_ids], dtype=torch.long, device=device)

    result = model.translate_with_attention(src_t, beam_size=beam)
    if not result["out_ids"]:
        print("  [warn] Empty translation — skipping attention plot")
        return

    attn_matrix = result["attention"]    # list[list[float]]: (T_out, T_src)
    out_ids     = result["out_ids"]

    # Decode BPE tokens for axis labels
    src_tokens = [src_vocab._id_to_token.get(i, "?") for i in src_ids]
    tgt_tokens = [tgt_vocab._id_to_token.get(i, "?") for i in out_ids]

    # Convert to 2D for plotting
    import numpy as np
    matrix = np.array(attn_matrix)      # (T_out, T_src)

    fig_w = max(8, len(src_tokens) * 0.5)
    fig_h = max(5, len(tgt_tokens) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Attention weight")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tgt_tokens, fontsize=9)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated tokens")
    ax.set_title(f'Cross-attention: "{sentence}"')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Attention heatmap → {out_path}")


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
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
        print(  "  Train first: python scripts/phase7_train.py")
        sys.exit(1)

    print(f"\nLoading checkpoint: {ckpt_path}")
    model, src_vocab, tgt_vocab, meta = load_checkpoint(ckpt_path)
    model = model.to(device)
    model.eval()
    print(f"  {model.param_count():,} parameters")
    if meta:
        print(f"  Meta: {meta}")

    # ── Attention visualization (optional, before eval loop) ──────────────────
    if args.attention:
        attn_path = ROOT / "results" / f"attention_{args.src}_{args.tgt}.png"
        print(f"\nRendering attention heatmap for: \"{args.attention}\"")
        visualize_attention(
            model, src_vocab, tgt_vocab,
            args.attention, device, attn_path, beam=args.beam,
        )

    # ── Load test set ──────────────────────────────────────────────────────────
    test_path = Path(args.test) if args.test else find_test_split(args.src, args.tgt)

    if test_path is None or not test_path.exists():
        print(
            f"\n[error] No test split found for {args.src}→{args.tgt}.\n"
            f"  Expected: data/splits/{args.src}_{args.tgt}_test_seed{DEFAULT_SPLIT_SEED}.tsv\n"
            f"  Create it by running: python scripts/phase7_train.py\n"
            f"  (The split is created automatically on first training run.)"
        )
        sys.exit(1)

    test_pairs = load_test_set(test_path)
    if not test_pairs:
        print(f"[error] Test file is empty: {test_path}")
        sys.exit(1)

    # Optionally sub-sample for faster evaluation (--max-test)
    if args.max_test and len(test_pairs) > args.max_test:
        import random
        rng = random.Random(DEFAULT_SPLIT_SEED)
        test_pairs = rng.sample(test_pairs, args.max_test)
        print(f"  Subsampled to {len(test_pairs):,} pairs (--max-test={args.max_test})")

    sources    = [s for s, _ in test_pairs]
    references = [r for _, r in test_pairs]

    print(f"\nTest set: {len(test_pairs):,} pairs from {test_path.name}")
    print(f"Translating with beam={args.beam}, max_len={args.max_len}…")

    t0 = time.time()
    hypotheses = translate_batch(
        model, src_vocab, tgt_vocab, sources, device,
        beam=args.beam, max_len=args.max_len,
        protect_entities=args.protect_entities,
        src_lang=args.src,
    )
    elapsed = time.time() - t0
    print(f"  Translated {len(hypotheses):,} sentences in {elapsed:.1f}s "
          f"({elapsed/len(hypotheses)*1000:.0f}ms/sent)")

    # ── BLEU ──────────────────────────────────────────────────────────────────
    detail = bleu_detail(hypotheses, references)
    bleu   = detail["bleu"]

    print(f"\n{'='*50}")
    print(f"  {args.src.upper()} → {args.tgt.upper()} BLEU: {bleu:.2f}")
    print(f"  Precision: {' / '.join(f'{p:.1f}' for p in detail['precisions'])}")
    print(f"  BP: {detail['bp']:.4f}  |  Hyp tokens: {detail['hyp_len']:,}  |  Ref tokens: {detail['ref_len']:,}")
    print(f"  Test examples: {detail['n_sentences']:,}")
    print(f"{'='*50}")

    # ── Sample translations ────────────────────────────────────────────────────
    import random
    rng = random.Random(DEFAULT_SPLIT_SEED)
    sample_idx = rng.sample(range(len(test_pairs)), min(DEFAULT_N_SAMPLES, len(test_pairs)))
    samples = []
    for idx in sorted(sample_idx):
        s_score = sentence_bleu(hypotheses[idx], references[idx])
        samples.append({
            "source":     sources[idx],
            "reference":  references[idx],
            "hypothesis": hypotheses[idx],
            "sentence_bleu": round(s_score * 100, 2),
        })

    # Print a few examples
    print("\nSample translations (source → reference | prediction):")
    for ex in samples[:8]:
        print(f"  SRC: {ex['source']}")
        print(f"  REF: {ex['reference']}")
        print(f"  HYP: {ex['hypothesis']}  [sent-BLEU: {ex['sentence_bleu']:.1f}]")
        print()

    # ── Save JSON report ───────────────────────────────────────────────────────
    report = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "src_lang":        args.src,
        "tgt_lang":        args.tgt,
        "checkpoint":      str(ckpt_path),
        "test_split_file": str(test_path),
        "split_seed":      DEFAULT_SPLIT_SEED,
        "n_train":         meta.get("n_train", "unknown"),
        "n_test":          len(test_pairs),
        "beam_size":       args.beam,
        "protect_entities": args.protect_entities,
        "bleu":            bleu,
        "bleu_detail":     detail,
        "samples":         samples,
        "meta":            meta,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / f"eval_{args.src}_{args.tgt}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report → {report_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a LinguaForge NMT checkpoint with BLEU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src",        default="es",  help="Source language ISO code")
    p.add_argument("--tgt",        default="en",  help="Target language ISO code")
    p.add_argument("--checkpoint", default=str(ROOT / "data" / "models" / "nmt_es_en.pt"),
                   help="Path to .pt checkpoint")
    p.add_argument("--test",       default=None,
                   help="TSV test file (auto-detected from data/splits/ if omitted)")
    p.add_argument("--beam",       type=int,   default=DEFAULT_BEAM)
    p.add_argument("--max-len",    type=int,   default=DEFAULT_MAX_LEN, dest="max_len")
    p.add_argument("--max-test",   type=int,   default=None, dest="max_test",
                   help="Sub-sample test set to this many pairs (faster evaluation)")
    p.add_argument("--attention",  default=None,
                   help="Render attention heatmap for this sentence")
    p.add_argument("--protect-entities", action="store_true", dest="protect_entities",
                   help="Enable named-entity passthrough during translation")
    p.add_argument("--no-mps",     action="store_true", dest="no_mps")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
