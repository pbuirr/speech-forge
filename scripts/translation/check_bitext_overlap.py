"""Audit evaluation bitext for overlap with a training corpus before comparing models.

Read-only with respect to its inputs. Standard library only.

For each test set it reports:
  * exact source / exact target overlap, using the project's `canonical` form
    (NFC, casefold, letters and digits only), the same rule the current
    evaluators and split checks use;
  * near-duplicates: test sentences of >= N words where at least --near-threshold
    of their word N-grams also occur in the training corpus (source vs source,
    target vs target);
and writes, per test set, a "clean" TSV that excludes every flagged pair, so
models can be compared on identical, uncontaminated sentences.

Example (PowerShell, from the repository root):

  python -X utf8 scripts/translation/check_bitext_overlap.py `
    --training C:/Users/user/speech-forge/data/corpora/es_en_filtered.tsv `
    --test news=data/training/es_en_news/test.tsv `
    --test newscommentary=data/corpora/es_en_newscommentary_independent.tsv `
    --test tatoeba=data/training/es_en_local/test.tsv `
    --output-dir results/legacy_nmt_overlap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from collections import defaultdict
from pathlib import Path


def canonical(text: str) -> str:
    # Identical to scripts/train_local_translation.canonical
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold() if c.isalnum())


def words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFC", text).casefold()
    cleaned = "".join(c if c.isalnum() else " " for c in folded)
    return cleaned.split()


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def read_pairs(path: Path):
    """Yield (line_number, source, target) like scripts/phase7_train.load_corpus.

    Lines starting with '#' are skipped, matching the project's loader.
    """
    with path.open(encoding="utf-8", errors="strict") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                yield number, parts[0].strip(), parts[1].strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--training",
        type=Path,
        nargs="+",
        required=True,
        help="One or more training TSVs (source<TAB>target)",
    )
    parser.add_argument(
        "--test",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Evaluation TSV to audit; repeat for several sets",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ngram", type=int, default=8, help="Word n-gram size for near-duplicate detection"
    )
    parser.add_argument(
        "--near-threshold",
        type=float,
        default=0.5,
        help="Flag a test sentence when this share of its n-grams occur in training",
    )
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()
    if args.ngram < 3 or not 0 < args.near_threshold <= 1:
        parser.error("--ngram must be >= 3 and --near-threshold within (0, 1]")

    tests = {}
    for spec in args.test:
        name, _, path = spec.partition("=")
        if not name or not path:
            parser.error(f"--test expects NAME=PATH, got {spec!r}")
        path = Path(path)
        rows = list(read_pairs(path))
        if not rows:
            parser.error(f"No pairs in {path}")
        tests[name] = {"path": path, "rows": rows}

    # Index the (small) test side, then stream the (large) training corpus once.
    exact_source = defaultdict(set)  # canonical source -> {(test, index)}
    exact_target = defaultdict(set)
    gram_index = defaultdict(set)  # (side, ngram) -> {(test, index)}
    gram_totals = {}  # (test, index, side) -> number of distinct n-grams
    for name, test in tests.items():
        for index, (_, source, target) in enumerate(test["rows"]):
            exact_source[canonical(source)].add((name, index))
            exact_target[canonical(target)].add((name, index))
            for side, text in (("source", source), ("target", target)):
                grams = ngrams(words(text), args.ngram)
                if grams:
                    gram_totals[(name, index, side)] = len(grams)
                    for gram in grams:
                        gram_index[(side, gram)].add((name, index))

    hits_exact_source, hits_exact_target = set(), set()
    gram_hits = defaultdict(set)  # (test, index, side) -> matched n-grams
    training_report = []
    started = time.perf_counter()
    for path in args.training:
        count = 0
        for _, source, target in read_pairs(path):
            count += 1
            hits_exact_source.update(exact_source.get(canonical(source), ()))
            hits_exact_target.update(exact_target.get(canonical(target), ()))
            for side, text in (("source", source), ("target", target)):
                for gram in ngrams(words(text), args.ngram):
                    for key in gram_index.get((side, gram), ()):
                        gram_hits[(key[0], key[1], side)].add(gram)
            if count % 200000 == 0:
                print(
                    f"{path.name}: {count:,} pairs scanned ({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
        training_report.append(
            {"path": str(path.resolve()), "pairs": count, "sha256": sha256(path)}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "training": training_report,
        "ngram": args.ngram,
        "near_threshold": args.near_threshold,
        "canonical_rule": "NFC + casefold + letters/digits only (same as train_local_translation.canonical)",
        "tests": {},
        "limits": "Detects exact and n-gram near-duplicates only. Paraphrases, translations of the same "
        "article via other routes, and sentences shorter than the n-gram size (exact check only) "
        "can still be contaminated.",
    }
    for name, test in tests.items():
        flagged = {}
        for index in range(len(test["rows"])):
            reasons = []
            if (name, index) in hits_exact_source:
                reasons.append("exact_source")
            if (name, index) in hits_exact_target:
                reasons.append("exact_target")
            for side in ("source", "target"):
                total = gram_totals.get((name, index, side))
                if total:
                    share = len(gram_hits.get((name, index, side), ())) / total
                    if share >= args.near_threshold:
                        reasons.append(f"near_{side}_{share:.2f}")
            if reasons:
                flagged[index] = reasons
        clean_rows = [row for index, row in enumerate(test["rows"]) if index not in flagged]
        clean_path = args.output_dir / f"{name}_clean.tsv"
        clean_path.write_text("".join(f"{s}\t{t}\n" for _, s, t in clean_rows), encoding="utf-8")
        short = sum(1 for _, s, _ in test["rows"] if len(words(s)) < args.ngram)
        report["tests"][name] = {
            "path": str(test["path"].resolve()),
            "sha256": sha256(test["path"]),
            "pairs": len(test["rows"]),
            "exact_source": sum("exact_source" in r for r in flagged.values()),
            "exact_target": sum("exact_target" in r for r in flagged.values()),
            "near_duplicate": sum(any(x.startswith("near_") for x in r) for r in flagged.values()),
            "flagged_total": len(flagged),
            "flagged_share": len(flagged) / len(test["rows"]),
            "sources_shorter_than_ngram_exact_check_only": short,
            "clean_pairs": len(clean_rows),
            "clean_path": str(clean_path.resolve()),
            "clean_sha256": sha256(clean_path),
            "examples": [
                {"line": test["rows"][i][0], "source": test["rows"][i][1], "reasons": flagged[i]}
                for i in sorted(flagged)[: args.examples]
            ],
        }
    (args.output_dir / "overlap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in t.items()
                    if k
                    in (
                        "pairs",
                        "exact_source",
                        "exact_target",
                        "near_duplicate",
                        "flagged_share",
                        "clean_pairs",
                    )
                }
                for name, t in report["tests"].items()
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Report: {args.output_dir / 'overlap_report.json'}")


if __name__ == "__main__":
    main()
