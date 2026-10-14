"""Compare learned spelling with the previous flat OOV lattice, without fitting."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.linguistic_resources import CorpusLexicon, read_conllu, normalized_form
from speech_engine.morphology import MorphologyEngine, POSTagger, CLOSED_CLASS
from speech_engine.pos_uncertainty import POSCalibration
from speech_engine.uncertainty_evaluation import (
    sentence_fingerprints,
    sentence_fingerprint,
    reliability_metrics,
)


def evaluate(resource, corpus, excluded, calibration, baseline=False):
    engine = MorphologyEngine()
    engine.add_lexicon(resource)
    original = POSTagger._lattice
    if baseline:
        # Reproduce the previous lattice exactly, including isolated-word hints.
        def flat(self, tokens, lang):
            lattice, transitions = original(self, tokens, lang)
            import math

            emissions = self._models[resource.lang][1]
            for token, scores in zip(tokens, lattice):
                if normalized_form(token) in emissions or not any(c.isalpha() for c in token):
                    continue
                hint = self._lookup_closed(token.lower(), CLOSED_CLASS.get(resource.lang, {}))
                hint = hint or self._apply_suffix_rules(token.lower(), resource.lang)
                choices = [hint] if len(tokens) == 1 and hint in scores else list(scores)
                scores.clear()
                scores.update(
                    {
                        tag: -math.log(len(emissions) + 1) + (math.log(4) if tag == hint else 0)
                        for tag in choices
                    }
                )
            return lattice, transitions

        POSTagger._lattice = flat
    sentences = []
    for word in read_conllu(Path(corpus).read_text(encoding="utf-8")):
        if word.token_id == 1:
            sentences.append([])
        sentences[-1].append(word)
    rows = []
    confusion = Counter()
    correct = Counter()
    totals = Counter()
    excluded_count = 0
    start = perf_counter()
    try:
        for sentence in sentences:
            if sentence_fingerprint(sentence) in excluded:
                excluded_count += 1
                continue
            analyses = engine.analyze_tokens([w.form for w in sentence], resource.lang).analyses
            for word, analysis in zip(sentence, analyses):
                if word.language and word.language != resource.lang:
                    continue
                known = bool(resource.lookup(word.form))
                group = "known" if known else "unknown"
                totals[group] += 1
                correct[group] += analysis.pos == word.pos
                if not known and analysis.pos != word.pos:
                    confusion[(word.pos, analysis.pos)] += 1
                evidence = analysis.pos_uncertainty
                if evidence and evidence["applies_to_returned_pos"]:
                    rows.append(
                        dict(
                            score=evidence["model_probability"],
                            known=known,
                            correct=analysis.pos == word.pos,
                        )
                    )
    finally:
        POSTagger._lattice = original
    elapsed = perf_counter() - start
    return {
        "sha256": sha256(Path(corpus).read_bytes()).hexdigest(),
        "excluded_sentences": excluded_count,
        "seconds_including_first_model_build": elapsed,
        "tokens_per_second": sum(totals.values()) / elapsed,
        "all_returned_predictions": {
            g: {"tokens": totals[g], "accuracy": correct[g] / totals[g]} for g in totals
        },
        "reliability_coverage": len(rows) / sum(totals.values()),
        "raw": reliability_metrics(rows),
        "calibrated": reliability_metrics(rows, calibration) if not baseline else None,
        "unknown_confusions": [
            {"gold": a, "predicted": b, "tokens": n} for (a, b), n in confusion.most_common(15)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    resource = CorpusLexicon.load(args.resource)
    if sha256(Path(args.training).read_bytes()).hexdigest() != resource.source["sha256"]:
        raise ValueError("Training corpus does not match resource")
    if sha256(Path(args.corpus).read_bytes()).hexdigest() in {
        resource.source["sha256"],
        sha256(Path(args.development).read_bytes()).hexdigest(),
    }:
        raise ValueError("Evaluation must be held out from training and calibration")
    excluded = sentence_fingerprints(args.training) | sentence_fingerprints(args.development)
    calibration = POSCalibration.load(args.calibration)
    engine = MorphologyEngine()
    engine.add_lexicon(resource)
    engine.tagger.add_calibration(calibration)
    report = {
        name: evaluate(resource, args.corpus, excluded, calibration, baseline)
        for name, baseline in [("baseline", True), ("learned", False)]
    }
    report["limits"] = (
        "Gold boundaries; no fitting on evaluation data. Baseline uses the previous flat unknown emissions. Timings include model construction and marginals, one run. Domain-shift calibration is applied unchanged from EWT development."
    )
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v["all_returned_predictions"] for k, v in report.items() if k != "limits"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
