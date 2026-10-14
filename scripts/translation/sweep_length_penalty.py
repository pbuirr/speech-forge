"""Sweep the beam-search length penalty α (Wu et al., 2016) on a dev sample.

Usage:
    python scripts/translation/sweep_length_penalty.py
    python scripts/translation/sweep_length_penalty.py --n 100 --checkpoint data/models/nmt_multilingual_16k_ep100.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_engine.bleu import bleu_detail
from speech_engine.bpe import BPEVocab
from speech_engine.nmt import load_checkpoint

LANG_TOKENS = {"es": "<ES>", "fr": "<FR>", "de": "<DE>"}
CORPORA = {
    lang: next(
        (
            ROOT / "data" / "corpora" / f"{lang}_en_{s}.tsv"
            for s in ("filtered", "real")
            if (ROOT / "data" / "corpora" / f"{lang}_en_{s}.tsv").exists()
        ),
        ROOT / "data" / "corpora" / f"{lang}_en_real.tsv",
    )
    for lang in LANG_TOKENS
}
SEED = 42


def load_pairs(lang: str, n: int) -> list[tuple[str, str]]:
    path = CORPORA[lang]
    if not path.exists():
        return []
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip().lower(), parts[1].strip().lower()))
    rng = random.Random(SEED + 1)
    rng.shuffle(pairs)
    return pairs[:n]


def encode_src(lang: str, src: str, vocab: BPEVocab) -> list[int]:
    lang_id = vocab._token_to_id[LANG_TOKENS[lang]]
    return [lang_id] + vocab.encode(src, add_bos=True, add_eos=True)


def sweep(
    model, vocab, device, lang: str, pairs: list[tuple[str, str]], alphas: list[float]
) -> dict[float, float]:
    sources = [s for s, _ in pairs]
    references = [r for _, r in pairs]
    results: dict[float, float] = {}

    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            hyps = []
            for src in sources:
                src_ids = encode_src(lang, src, vocab)
                src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
                out = model.translate_beam(src_t, beam_size=4, length_penalty=alpha)
                hyps.append(vocab.decode(out).strip())
            bleu = bleu_detail(hyps, references)["bleu"]
            results[alpha] = bleu
            print(f"  α={alpha:.1f}  BLEU={bleu:.2f}", flush=True)

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", default=str(ROOT / "data" / "models" / "nmt_multilingual_16k_ep100.pt")
    )
    p.add_argument("--n", type=int, default=150, help="Sentences per language")
    p.add_argument("--langs", nargs="+", default=["es", "fr", "de"])
    p.add_argument("--no_mps", action="store_true")
    args = p.parse_args()

    alphas = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]

    if not args.no_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading {args.checkpoint}…")
    model, vocab, _, meta = load_checkpoint(args.checkpoint)
    model = model.to(device)

    all_scores: dict[float, list[float]] = {a: [] for a in alphas}

    for lang in args.langs:
        pairs = load_pairs(lang, args.n)
        if not pairs:
            print(f"  [{lang.upper()}] no data, skipping")
            continue
        print(f"\n[{lang.upper()}→EN]  {len(pairs)} sentences")
        lang_scores = sweep(model, vocab, device, lang, pairs, alphas)
        for a, b in lang_scores.items():
            all_scores[a].append(b)

    print("\n── Combined average BLEU by α ────────────────────────────────")
    best_alpha, best_bleu = 0.6, 0.0
    for alpha in alphas:
        scores = all_scores[alpha]
        avg = sum(scores) / len(scores) if scores else 0.0
        marker = " ◄ best" if avg > best_bleu else ""
        if avg > best_bleu:
            best_bleu, best_alpha = avg, alpha
        print(f"  α={alpha:.1f}  avg BLEU={avg:.2f}{marker}")

    print(f"\nRecommended length_penalty = {best_alpha}")
    print(f"Add to translate_beam() call: length_penalty={best_alpha}")


if __name__ == "__main__":
    main()
