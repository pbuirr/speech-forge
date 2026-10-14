"""Evaluate an NMT checkpoint using the same sentence splitting and preprocessing as the web app."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from speech_engine.bleu import bleu_detail
from speech_engine.nmt import load_checkpoint
from speech_engine.preprocessing import (
    normalize_es_input,
    postprocess_en_output,
    detect_translation_failure,
)
from scripts.translation.train_nmt import load_corpus
from scripts.translation.train_local_translation import canonical


def character_f2(hypotheses, references):
    precisions, recalls = [], []
    for order in range(1, 7):
        matched = hyp_total = ref_total = 0
        for hypothesis, reference in zip(hypotheses, references):
            h, r = "".join(hypothesis.lower().split()), "".join(reference.lower().split())
            hc = Counter(h[i : i + order] for i in range(max(0, len(h) - order + 1)))
            rc = Counter(r[i : i + order] for i in range(max(0, len(r) - order + 1)))
            matched += sum((hc & rc).values())
            hyp_total += sum(hc.values())
            ref_total += sum(rc.values())
        precisions.append(matched / max(1, hyp_total))
        recalls.append(matched / max(1, ref_total))
    p, r = sum(precisions) / 6, sum(recalls) / 6
    return 100 * 5 * p * r / max(1e-12, 4 * p + r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=983)
    parser.add_argument(
        "--exclude-prior-report",
        "--exclude-prior-reports",
        dest="exclude_prior_reports",
        type=Path,
        nargs="+",
        help="Reserve a fresh audit sample by excluding previously evaluated sources",
    )
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-eos-boost", action="store_true")
    parser.add_argument("--no-repeat-guard", action="store_true")
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--allow-repeated-fragments", action="store_true")
    parser.add_argument("--allow-repeated-digits", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--exclude-splits",
        type=Path,
        help="Exclude exact normalized overlap with original experiment",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, sv, tv, metadata = load_checkpoint(args.checkpoint)
    args.allow_repeated_digits = (
        args.allow_repeated_digits
        or metadata.get("decoding_policy") == "preserve_numeric_fragments"
    )
    device = torch.device(args.device)
    model.to(device)
    pairs = load_corpus(args.pairs)
    excluded = 0
    if args.exclude_splits:
        training = [
            p
            for name in ["train", "dev", "test"]
            for p in load_corpus(args.exclude_splits / f"{name}.tsv")
        ]
        seen_source = {canonical(s) for s, t in training}
        seen_target = {canonical(t) for s, t in training}
        clean = [
            (s, t)
            for s, t in pairs
            if canonical(s) not in seen_source and canonical(t) not in seen_target
        ]
        excluded = len(pairs) - len(clean)
        pairs = clean
    prior_excluded = 0
    if args.exclude_prior_reports:
        seen = set()
        for path in args.exclude_prior_reports:
            prior = json.loads(path.read_text(encoding="utf-8"))
            seen.update(canonical(row["source"]) for row in prior["rows"])
        clean = [pair for pair in pairs if canonical(pair[0]) not in seen]
        prior_excluded = len(pairs) - len(clean)
        pairs = clean
    pairs = random.Random(args.sample_seed).sample(pairs, min(args.limit, len(pairs)))
    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for index, (source, reference) in enumerate(pairs):
            prep = normalize_es_input(source) if metadata["direction"] == "es→en" else None
            normalized = prep.normalized if prep else source
            ids = sv.encode(normalized.lower(), add_bos=True, add_eos=True)
            if len(ids) > model._config["max_src_len"]:
                rows.append(
                    {
                        "source": source,
                        "reference": reference,
                        "raw": "",
                        "application": "",
                        "skipped": "source_token_limit",
                        "latency_s": 0,
                    }
                )
                continue
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            if args.greedy:
                out = model.translate_greedy(tensor)
                entropy = None
            else:
                result = model.translate_with_attention(
                    tensor,
                    beam_size=args.beam,
                    eos_boost=not args.no_eos_boost,
                    repeat_guard=not args.no_repeat_guard,
                    no_repeat_ngram=args.no_repeat_ngram,
                    repeat_exempt_ids={
                        i for i, t in tv._id_to_token.items() if not t.endswith("</w>") and i > 3
                    }
                    if args.allow_repeated_fragments
                    else {i for i, t in tv._id_to_token.items() if t.removesuffix("</w>").isdigit()}
                    if args.allow_repeated_digits
                    else None,
                )
                out = result["out_ids"]
                entropy = np.mean(result["entropy"]).item() if result["entropy"] else None
            if device.type == "cuda":
                torch.cuda.synchronize()
            raw = tv.decode(out).strip()
            output = postprocess_en_output(raw) if metadata["direction"].endswith("→en") else raw
            warnings = prep.coverage_failures(output) if prep else []
            flagged = detect_translation_failure(
                normalized,
                output,
                avg_entropy=entropy,
                src_tokens=[sv._id_to_token.get(i, "?") for i in ids],
            ) or bool(warnings)
            rows.append(
                {
                    "source": source,
                    "reference": reference,
                    "raw": raw,
                    "application": output,
                    "latency_s": time.perf_counter() - began,
                    "review_flag": flagged,
                    "attention_entropy_uncalibrated": entropy,
                }
            )
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{len(pairs)} ({time.perf_counter() - start:.1f}s)", flush=True)
    refs = [row["reference"] for row in rows]
    metrics = {}
    for key in ["raw", "application"]:
        predictions = [row[key] for row in rows]
        metrics[key] = {
            **bleu_detail(predictions, refs),
            "character_f2_6_no_space": character_f2(predictions, refs),
            "exact_match_lowercase": sum(
                h.strip().lower() == r.strip().lower() for h, r in zip(predictions, refs)
            )
            / max(1, len(rows)),
        }
    latencies = [row["latency_s"] for row in rows if not row.get("skipped")]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "metadata": metadata,
        "device": str(device),
        "decoder": "greedy" if args.greedy else f"beam_{args.beam}",
        "decoding_options": {
            "eos_boost": not args.no_eos_boost,
            "repeat_guard": not args.no_repeat_guard,
            "no_repeat_ngram": args.no_repeat_ngram,
            "allow_repeated_fragments": args.allow_repeated_fragments,
            "allow_repeated_digits": args.allow_repeated_digits,
        },
        "sample_seed": args.sample_seed,
        "prior_evaluation_excluded": prior_excluded,
        "examples": len(rows),
        "excluded_overlap": excluded,
        "metrics": metrics,
        "copy_source_baseline": bleu_detail([row["source"] for row in rows], refs),
        "review_flag_rate": sum(row.get("review_flag", False) for row in rows) / max(1, len(rows)),
        "latency_median_s": float(np.median(latencies)) if latencies else None,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
        "elapsed_s": time.perf_counter() - start,
        "limits": "single-reference lowercase whitespace BLEU; custom character F2; no semantic correctness or calibrated confidence claim; lexical fallback not included",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("rows", "metadata")},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
