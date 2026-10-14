"""Compare output length and vocabulary statistics between saved translation runs."""

import argparse
import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.translation.train_nmt import load_corpus
from scripts.translation.evaluate_local_translation import character_f2
from speech_engine.bleu import bleu_detail
from speech_engine.nmt import load_checkpoint


def diagnose(rows, known, vocab):
    words = [word for row in rows for word in row["source"].split()]
    subwords = [vocab.encode(row["source"]) for row in rows]
    return {
        "examples": len(rows),
        "mean_source_words": len(words) / max(1, len(rows)),
        "unseen_source_wordform_rate": sum(word not in known for word in words)
        / max(1, len(words)),
        "subwords_per_word": sum(map(len, subwords)) / max(1, len(words)),
        "unknown_subword_rate": sum(ids.count(3) for ids in subwords)
        / max(1, sum(map(len, subwords))),
        "raw_bleu": bleu_detail([row["raw"] for row in rows], [row["reference"] for row in rows])[
            "bleu"
        ],
        "raw_character_f2": character_f2(
            [row["raw"] for row in rows], [row["reference"] for row in rows]
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    _, vocab, _, _ = load_checkpoint(args.checkpoint)
    known = {word for source, target in load_corpus(args.training) for word in source.split()}
    report = {
        "limits": "Whitespace word-form novelty is not linguistic unknown-word accuracy. Length and novelty correlate with domain; descriptive strata do not identify causal effects.",
        "training": str(args.training),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(args.training.read_bytes()).hexdigest(),
        "datasets": {},
    }
    for path in args.reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        summary = diagnose(rows, known, vocab)
        summary["length_bins"] = {}
        for low, high in [(1, 10), (11, 20), (21, 55), (56, 1000)]:
            selected = [row for row in rows if low <= len(row["source"].split()) <= high]
            if selected:
                summary["length_bins"][f"{low}-{high}"] = diagnose(selected, known, vocab)
        report["datasets"][path.stem] = summary
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
