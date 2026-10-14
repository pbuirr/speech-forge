"""Fit the POS confidence calibration on development data."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.linguistic_resources import CorpusLexicon
from speech_engine.pos_uncertainty import POSCalibration
from speech_engine.uncertainty_evaluation import (
    collect_pos_scores,
    reliability_metrics,
    sentence_fingerprints,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument(
        "--training-corpus",
        type=Path,
        required=True,
        help="Original training corpus for exact-sentence overlap checks",
    )
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lexicon = CorpusLexicon.load(args.resource)
    from hashlib import sha256
    from speech_engine.morphology import MorphologyEngine

    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    if sha256(args.training_corpus.read_bytes()).hexdigest() != lexicon.source["sha256"]:
        raise ValueError("Training corpus does not match the loaded resource")
    training_fingerprints = sentence_fingerprints(args.training_corpus)
    development, fingerprints, dev_source = collect_pos_scores(
        lexicon, args.development, training_fingerprints
    )
    # Check before fitting/writing: evaluation must be a separate source.
    if sha256(args.test.read_bytes()).hexdigest() in {
        dev_source["sha256"],
        lexicon.source["sha256"],
    }:
        raise ValueError("Training, development and test sources must be distinct")
    calibration = POSCalibration.fit(
        [(r["score"], r["known"], r["correct"]) for r in development],
        lexicon.lang,
        lexicon.source["sha256"],
        dev_source["sha256"],
        engine.tagger.model_fingerprint(lexicon.lang),
    )
    test, _, test_source = collect_pos_scores(
        lexicon, args.test, fingerprints | training_fingerprints
    )
    report = {
        "language": lexicon.lang,
        "training_source": lexicon.source,
        "model_sha256": calibration.model_sha256,
        "development": dev_source,
        "test": test_source,
        "development_tokens": len(development),
        "test_tokens": len(test),
        "test_raw": reliability_metrics(test),
        "test_calibrated": reliability_metrics(test, calibration),
        "by_form": {
            group: {
                "raw": reliability_metrics([r for r in test if r["known"] == known]),
                "calibrated": reliability_metrics(
                    [r for r in test if r["known"] == known], calibration
                ),
            }
            for group, known in [("known", True), ("unknown", False)]
        },
        "limits": "Gold sentence/word boundaries. POS only, including missing-lemma words; foreign-tagged and returned-POS mismatches are excluded explicitly. Development excludes exact training word sequences; test excludes exact training/development word sequences. Near-duplicate overlap is not detected. Scores are model-conditionals; empirical accuracy estimates may fail under domain shift or tokenization errors. No lemma, feature, semantic, speech or translation confidence is calibrated.",
    }
    calibration.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: {
                    k: v for k, v in report[key].items() if k not in ("bins", "selective_accuracy")
                }
                for key in ("test_raw", "test_calibrated")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
