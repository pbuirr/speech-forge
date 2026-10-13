"""Measure lexical coverage and exact analysis agreement on a held-out corpus."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from speech_engine.linguistic_evaluation import evaluate_lexicon, evaluate_context, evaluate_raw_text
from speech_engine.linguistic_resources import CorpusLexicon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", action="store_true", help="Also evaluate the production contextual analyzer versus heuristics")
    parser.add_argument("--raw-text", action="store_true", help="Evaluate tokenization and POS from raw sentence text")
    args = parser.parse_args()
    lexicon = CorpusLexicon.load(args.resource)
    report = evaluate_lexicon(lexicon, args.test)
    if args.context:
        report["contextual_comparison"] = evaluate_context(lexicon, args.test)
    if args.raw_text:
        report["raw_text"] = evaluate_raw_text(lexicon, args.test)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
