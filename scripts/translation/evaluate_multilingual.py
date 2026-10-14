"""
scripts/translation/evaluate_multilingual.py — BLEU evaluation for the Phase 9 multilingual checkpoint.

Evaluates ES→EN, FR→EN, DE→EN independently and combined, using the same
language-token + encode_pair() scheme as phase8_multilingual_train.py.

Also audits filtered corpora for data-fusion rows (source/target fused with no
tab separator, or single-column rows that slipped through).

Usage:
    .venv/bin/python3 scripts/translation/evaluate_multilingual.py
    .venv/bin/python3 scripts/translation/evaluate_multilingual.py --checkpoint data/models/nmt_multilingual_16k_ep100.pt
    .venv/bin/python3 scripts/translation/evaluate_multilingual.py --n 200 --beam 4
    .venv/bin/python3 scripts/translation/evaluate_multilingual.py --audit_only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_engine.bleu import bleu_detail, sentence_bleu
from speech_engine.bpe import BPEVocab
from speech_engine.nmt import Seq2SeqNMT, load_checkpoint

LANG_TOKENS = {"es": "<ES>", "fr": "<FR>", "de": "<DE>"}

CORPORA: dict[str, Path] = {
    lang: next(
        (
            ROOT / "data" / "corpora" / f"{lang}_en_{suffix}.tsv"
            for suffix in ("filtered", "real")
            if (ROOT / "data" / "corpora" / f"{lang}_en_{suffix}.tsv").exists()
        ),
        ROOT / "data" / "corpora" / f"{lang}_en_real.tsv",
    )
    for lang in LANG_TOKENS
}

EVAL_SEED = 99


# ── Data-fusion audit ─────────────────────────────────────────────────────────


def audit_corpus(path: Path) -> dict:
    """Scan a corpus TSV for malformed rows and return a summary."""
    if not path.exists():
        return {"path": str(path), "error": "file not found"}

    total = 0
    no_tab = 0
    no_tab_examples: list[str] = []
    empty_src = 0
    empty_tgt = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        total += 1
        parts = line.split("\t")
        if len(parts) < 2:
            no_tab += 1
            if len(no_tab_examples) < 3:
                no_tab_examples.append(line[:120])
            continue
        src, tgt = parts[0].strip(), parts[1].strip()
        if not src:
            empty_src += 1
        if not tgt:
            empty_tgt += 1

    bad = no_tab + empty_src + empty_tgt
    return {
        "path": str(path),
        "total_rows": total,
        "no_tab_rows": no_tab,
        "empty_src_rows": empty_src,
        "empty_tgt_rows": empty_tgt,
        "total_bad": bad,
        "bad_pct": round(100 * bad / max(1, total), 3),
        "no_tab_examples": no_tab_examples,
    }


# ── Encoding ──────────────────────────────────────────────────────────────────


def encode_src(lang: str, src: str, vocab: BPEVocab) -> list[int]:
    lang_tok = LANG_TOKENS[lang]
    lang_id = vocab._token_to_id[lang_tok]
    return [lang_id] + vocab.encode(src, add_bos=True, add_eos=True)


# ── Corpus loading ────────────────────────────────────────────────────────────


def load_test_pairs(
    lang: str,
    n: int,
    seed: int = EVAL_SEED,
) -> list[tuple[str, str]]:
    path = CORPORA[lang]
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {lang.upper()}")
        return []

    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip().lower(), parts[1].strip().lower()))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:n]


# ── Per-language BLEU eval ────────────────────────────────────────────────────


def eval_language(
    lang: str,
    model: Seq2SeqNMT,
    vocab: BPEVocab,
    device: torch.device,
    n: int,
    beam: int,
    n_samples: int,
) -> dict:
    pairs = load_test_pairs(lang, n)
    if not pairs:
        return {"lang": lang, "error": "no data", "hypotheses": [], "references": []}

    sources = [s for s, _ in pairs]
    references = [r for _, r in pairs]

    print(f"  [{lang.upper()}→EN] translating {len(pairs):,} sentences…", flush=True)
    t0 = time.time()

    hypotheses: list[str] = []
    model.eval()
    with torch.no_grad():
        for i, src_str in enumerate(sources):
            src_ids = encode_src(lang, src_str, vocab)
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            out_ids = model.translate_beam(src_t, beam_size=beam)
            hypotheses.append(vocab.decode(out_ids).strip())
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(pairs)}…", flush=True)

    elapsed = time.time() - t0
    detail = bleu_detail(hypotheses, references)

    # Sample translations
    rng = random.Random(EVAL_SEED + ord(lang[0]))
    idx = sorted(rng.sample(range(len(pairs)), min(n_samples, len(pairs))))
    samples = [
        {
            "source": sources[i],
            "reference": references[i],
            "hypothesis": hypotheses[i],
            "sentence_bleu": round(sentence_bleu(hypotheses[i], references[i]) * 100, 2),
        }
        for i in idx
    ]

    print(f"    BLEU: {detail['bleu']:.2f}  ({elapsed:.0f}s)", flush=True)
    return {
        "lang": lang,
        "n_pairs": len(pairs),
        "elapsed_s": round(elapsed, 1),
        "hypotheses": hypotheses,  # kept for combined BLEU, not written to JSON
        "references": references,
        **detail,
        "samples": samples,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", default=str(ROOT / "data" / "models" / "nmt_multilingual_16k_ep100.pt")
    )
    p.add_argument("--n", type=int, default=500, help="Test pairs per language")
    p.add_argument("--beam", type=int, default=4, help="Beam size")
    p.add_argument(
        "--n_samples", type=int, default=20, help="Sample translations per language in report"
    )
    p.add_argument("--out", default=str(ROOT / "results" / "eval_multilingual.json"))
    p.add_argument("--no_mps", action="store_true")
    p.add_argument(
        "--audit_only", action="store_true", help="Only run corpus audit, skip translation"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Corpus audit ─────────────────────────────────────────────────────────
    print("\n── Corpus audit ──────────────────────────────────────────────────────")
    audit_results = {}
    for lang, path in CORPORA.items():
        result = audit_corpus(path)
        audit_results[lang] = result
        status = (
            "✓ clean"
            if result.get("total_bad", 0) == 0
            else f"⚠ {result['total_bad']:,} bad rows ({result['bad_pct']}%)"
        )
        print(f"  {lang.upper()}: {path.name}  →  {result.get('total_rows', '?'):,} rows  {status}")
        if result.get("no_tab_rows", 0):
            print(f"    no-tab examples:")
            for ex in result.get("no_tab_examples", []):
                print(f"      {ex}")

    if args.audit_only:
        print(json.dumps({"audit": audit_results}, indent=2, ensure_ascii=False))
        return

    # ── Device ───────────────────────────────────────────────────────────────
    if not args.no_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\nUsing MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\nUsing CPU")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, vocab, _, meta = load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  vocab: {vocab.vocab_size:,} tokens  |  meta: {meta}")

    # Verify language tokens exist
    for lang, tok in LANG_TOKENS.items():
        tid = vocab._token_to_id.get(tok)
        if tid is None:
            print(f"  ERROR: {tok} missing from vocab — cannot evaluate {lang.upper()}")

    # ── Per-language evaluation ───────────────────────────────────────────────
    print(f"\n── BLEU evaluation  (n={args.n} per lang, beam={args.beam}) ──────────")
    lang_results: dict[str, dict] = {}
    all_hyps: list[str] = []
    all_refs: list[str] = []

    for lang in LANG_TOKENS:
        result = eval_language(lang, model, vocab, device, args.n, args.beam, args.n_samples)
        lang_results[lang] = result
        if "samples" in result:
            all_hyps.extend(h["hypothesis"] for h in result["samples"])
            all_refs.extend(h["reference"] for h in result["samples"])

    # Combined BLEU over all hypotheses/references we collected
    # (uses the full hyp/ref lists, not just samples)
    combined_pairs: list[tuple[str, str]] = []
    for lang in LANG_TOKENS:
        pairs = load_test_pairs(lang, args.n)
        combined_pairs.extend(pairs)

    # Combined BLEU over all languages
    all_hyps = [h for r in lang_results.values() for h in r.get("hypotheses", [])]
    all_refs = [r_ for r in lang_results.values() for r_ in r.get("references", [])]
    combined = bleu_detail(all_hyps, all_refs) if all_hyps else {"bleu": 0.0}

    print("\n── Summary ───────────────────────────────────────────────────────────")
    for lang, r in lang_results.items():
        print(
            f"  {lang.upper()}→EN:  BLEU {r.get('bleu', 0):5.2f}  ({r.get('n_pairs', 0):,} sentences)"
        )
    print(f"  Combined:     BLEU {combined['bleu']:5.2f}  ({len(all_hyps):,} sentences)")

    # ── Save report ───────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip raw hypotheses/references before saving — samples cover the qualitative view
    for r in lang_results.values():
        r.pop("hypotheses", None)
        r.pop("references", None)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "beam_size": args.beam,
        "n_per_lang": args.n,
        "combined_bleu": round(combined["bleu"], 4),
        "languages": lang_results,
        "audit": audit_results,
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
