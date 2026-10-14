"""Reproduce exercise correctness on authored multilingual reference cases.

This measures software behavior, not student learning or general linguistic
accuracy. An optional trusted baseline Python module enables before/after runs.
"""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speech_engine.flashcard import FlashcardDeck
from speech_engine import exercise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-module", type=Path, help="Trusted Python exercise implementation to compare"
    )
    parser.add_argument(
        "--grammar-corpus",
        type=Path,
        help="Optional CoNLL-U original texts for rule execution/offset checks",
    )
    parser.add_argument("--grammar-lang", default="en")
    parser.add_argument("--output", type=Path, default=ROOT / "results/learning_exercises.json")
    args = parser.parse_args()
    implementations = [("current", exercise)]
    if args.baseline_module:
        spec = importlib.util.spec_from_file_location(
            "_trusted_exercise_baseline", args.baseline_module
        )
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        implementations.insert(0, ("baseline", baseline))
    fixture = ROOT / "tests/fixtures/learning_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    report = {
        "fixture": "tests/fixtures/learning_cases.json",
        "cases": len(cases),
        "fixture_sha256": sha256(fixture.read_bytes()).hexdigest(),
        "languages": sorted({c["lang"] for c in cases}),
        "implementations": {},
        "limits": "Authored regression cases, not representative corpora or a study of learning outcomes. Chinese/Japanese multi-character targets have explicit authored spans; this does not establish word segmentation quality. No model inference or training occurs.",
    }
    for name, module in implementations:
        correct = 0
        failures = []
        for case in cases:
            analysis = None
            if case.get("annotated_span"):
                start = case["text"].index(case["front"])
                analysis = {
                    "span": [start, start + len(case["front"])],
                    "span_unit": "unicode_codepoints",
                }
            card = FlashcardDeck().add(
                case["front"],
                "authored reference",
                case["lang"],
                example=case["text"],
                analysis=analysis,
            )
            generated = module.generate_fill_in_blank(card)
            hit = generated.prompt == case["expected_prompt"]
            correct += hit
            if not hit:
                failures.append(
                    {
                        "language": case["lang"],
                        "text": case["text"],
                        "expected_prompt": case["expected_prompt"],
                        "actual_prompt": generated.prompt,
                    }
                )
        report["implementations"][name] = {
            "correct_cloze_prompts": correct,
            "total": len(cases),
            "failures": failures,
        }
    if args.baseline_module:
        report["baseline_module_sha256"] = sha256(args.baseline_module.read_bytes()).hexdigest()
    if args.grammar_corpus:
        from speech_engine.grammar import GrammarChecker

        checker = GrammarChecker()
        completed = Counter()
        failed = Counter()
        sentences = 0
        for line in args.grammar_corpus.read_text(encoding="utf-8").splitlines():
            if not line.startswith("# text = "):
                continue
            sentences += 1
            result = checker.check(line[9:], args.grammar_lang)
            completed.update(result.checked_rules)
            failed.update((f["rule_id"], f["error_type"]) for f in result.failed_rules)
        report["grammar_health"] = {
            "language": args.grammar_lang,
            "sentences": sentences,
            "corpus_sha256": sha256(args.grammar_corpus.read_bytes()).hexdigest(),
            "completed_checks": sum(completed.values()),
            "failures": [
                {"rule_id": rule, "error_type": kind, "count": count}
                for (rule, kind), count in sorted(failed.items())
            ],
            "limits": "Checks rule execution and valid offsets, not linguistic correctness of findings.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {k: v for k, v in result.items() if k != "failures"}
                for name, result in report["implementations"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
