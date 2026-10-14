"""Evaluate sentence splitting, roots and attachments on raw text, with exact source spans."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine import morphology
from speech_engine.dependency import read_dependencies, load_dependency_models, syntax_payload
from speech_engine.uncertainty_evaluation import sentence_fingerprint, sentence_fingerprints


def old_ranges(tokens, *args, **kwargs):
    result = []
    start = 0
    for end, token in enumerate(tokens, 1):
        if token in {".", "!", "?", "。", "！", "？"} or end == len(tokens):
            result.append((start, end))
            start = end
    return result


def rate(correct, predicted, gold):
    precision = correct / predicted if predicted else 0
    recall = correct / gold if gold else 0
    return {
        "correct": correct,
        "predicted": predicted,
        "reference": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["corpus", "training", "development", "resources", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--syntax-resources", help="Optional syntax directory for controlled model comparisons"
    )
    args = parser.parse_args()
    engine = morphology.MorphologyEngine()
    engine.load_resources(args.resources)
    load_dependency_models(engine, args.syntax_resources or Path(args.resources) / "syntax")
    from speech_engine.linguistic_resources import language_code

    model = engine.dependency_parsers.get(language_code(args.lang))
    if (
        model is None
        or sha256(Path(args.training).read_bytes()).hexdigest() != model.source["training_sha256"]
    ):
        raise ValueError("Evaluation requires a matching locally trained syntax resource")
    if (
        args.limit < 1
        or sha256(Path(args.corpus).read_bytes()).hexdigest() == model.source["training_sha256"]
    ):
        raise ValueError("Use a positive limit and held-out corpus")
    excluded = sentence_fingerprints(args.training)
    if Path(args.corpus).resolve() != Path(args.development).resolve():
        excluded |= sentence_fingerprints(args.development)
    sentences = list(read_dependencies(args.corpus))
    blocks = Path(args.corpus).read_text(encoding="utf-8").strip().split("\n\n")
    examples = []
    for block, sentence in zip(blocks, sentences):
        match = re.search(r"^# text = (.*)$", block, re.M)
        if match and sentence_fingerprint(sentence.words) not in excluded:
            examples.append((match[1], sentence))
    examples = examples[: args.limit]
    documents = []
    unsupported = 0
    for i in range(0, len(examples), 2):
        text = ""
        boundaries = set()
        gold_roots = set()
        gold_arcs = {}
        missing = 0
        for source, sentence in examples[i : i + 2]:
            if text:
                text += " "
            offset = len(text)
            text += source
            boundaries.add(len(text))
            spans = []
            cursor = 0
            for word in sentence.words:
                left = source.find(word.form, cursor)
                span = (offset + left, offset + left + len(word.form)) if left >= 0 else None
                if span:
                    cursor = left + len(word.form)
                spans.append(span)
                missing += span is None
            for j, word in enumerate(sentence.words):
                if word.pos == "PUNCT":
                    continue
                head = sentence.heads[j]
                if head == 0 and spans[j]:
                    gold_roots.add(spans[j])
                gold_arcs[(offset, j)] = (
                    spans[j],
                    spans[head - 1] if head else None,
                    sentence.labels[j],
                    head == 0,
                )
        unsupported += missing
        documents.append((text, boundaries, gold_roots, gold_arcs))
    report = {}
    original = morphology.sentence_ranges

    def without_abbreviations(tokens, spans=None, text=None, abbreviations=()):
        return original(tokens, spans, text)

    try:
        for name, function in [
            ("previous_independent_period_splits", old_ranges),
            ("shared_without_abbreviation_hints", without_abbreviations),
            ("shared_source_boundaries", original),
        ]:
            morphology.sentence_ranges = function
            bc = bp = bg = rc = rp = rg = ac = ap = ag = 0
            start = perf_counter()
            for text, bounds, roots, gold in documents:
                analysis = engine.analyze(text, args.lang)
                syntax = syntax_payload(analysis, engine.dependency_parsers)
                predicted_bounds = {
                    analysis.analyses[b - 1].span[1] for a, b in analysis.sentence_ranges
                }
                bc += len(predicted_bounds & bounds)
                bp += len(predicted_bounds)
                bg += len(bounds)
                predicted_roots = {
                    analysis.analyses[a["token_index"]].span
                    for a in syntax["arcs"]
                    if a["head_token_index"] is None
                }
                rc += len(predicted_roots & roots)
                rp += len(predicted_roots)
                rg += sum(value[3] for value in gold.values())
                reference = {value[:3] for value in gold.values() if value[0] is not None}
                guesses = set()
                for arc in syntax["arcs"]:
                    word = analysis.analyses[arc["token_index"]]
                    if not any(c.isalnum() for c in word.surface):
                        continue
                    head = (
                        analysis.analyses[arc["head_token_index"]].span
                        if arc["head_token_index"] is not None
                        else None
                    )
                    guesses.add((word.span, head, arc["relation"]))
                ac += len(reference & guesses)
                ap += len(guesses)
                ag += len(gold)
            report[name] = {
                "boundary": rate(bc, bp, bg),
                "root": rate(rc, rp, rg),
                "labeled_attachment": rate(ac, ap, ag),
                "seconds": perf_counter() - start,
            }
    finally:
        morphology.sentence_ranges = original
    report["data"] = {
        "corpus_sha256": sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "source_sentences": len(examples),
        "paired_documents": len(documents),
        "unaligned_reference_words": unsupported,
        "limits": "First N held-out text records, paired with one space; gold boundaries never enter inference. Exact train/dev overlap excluded. Exact surface alignment can fail for expanded multiword tokens; all reference lexical arcs/root counts remain in recall denominators. Root F1 is exact source span, attachment F1 requires dependent/head spans and relation. PUD annotations sometimes contain multiple linguistic sentences in one record. Natural paragraph distribution and speech segmentation are not established.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
