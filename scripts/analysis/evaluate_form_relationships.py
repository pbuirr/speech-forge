"""Evaluate corpus-derived form links on held-out gold word boundaries."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.linguistic_resources import read_conllu, normalized_form
from speech_engine.morphology import MorphologyEngine
from speech_engine.understanding import morphology_payload
from speech_engine.uncertainty_evaluation import sentence_fingerprint, sentence_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine = MorphologyEngine()
    engine.load_resources(args.resources)
    excluded = sentence_fingerprints(args.training) | sentence_fingerprints(args.development)
    raw = Path(args.corpus).read_bytes()
    if sha256(raw).hexdigest() in {
        sha256(Path(p).read_bytes()).hexdigest() for p in [args.training, args.development]
    }:
        raise ValueError("Use held-out evaluation data")
    sentences = []
    for word in read_conllu(raw.decode("utf-8")):
        if word.token_id == 1:
            sentences.append([])
        sentences[-1].append(word)
    counts = dict(
        sentences=0,
        excluded_overlap_sentences=0,
        links=0,
        correct_link_labels=0,
        contrasts=0,
        correct_contrast_labels=0,
        reference_link_opportunities=0,
    )
    start = perf_counter()
    for sentence in sentences:
        if sentence_fingerprint(sentence) in excluded:
            counts["excluded_overlap_sentences"] += 1
            continue
        counts["sentences"] += 1
        seen = set()
        for word in sentence:
            if word.lemma and word.pos not in {"PUNCT", "SYM", "X"}:
                key = normalized_form(word.lemma), word.pos
                counts["reference_link_opportunities"] += key in seen
                seen.add(key)
        payload = morphology_payload(
            engine.analyze_tokens([w.form for w in sentence], args.lang), engine
        )
        for link in payload["relationships"]:
            counts["links"] += 1
            a, b = [sentence[i] for i in link["token_indices"]]
            correct = (
                a.lemma is not None
                and b.lemma is not None
                and normalized_form(a.lemma)
                == normalized_form(b.lemma)
                == normalized_form(link["lemma"])
                and a.pos == b.pos == link["pos"]
            )
            counts["correct_link_labels"] += correct
            for contrast in link["contrasts"]:
                counts["contrasts"] += 1
                counts["correct_contrast_labels"] += bool(
                    correct
                    and dict(a.features).get(contrast["feature"]) == contrast["left"]
                    and dict(b.features).get(contrast["feature"]) == contrast["right"]
                )
    report = {
        "corpus_sha256": sha256(raw).hexdigest(),
        "language": args.lang,
        "counts": counts,
        "link_label_precision": counts["correct_link_labels"] / counts["links"]
        if counts["links"]
        else None,
        "contrast_label_precision": counts["correct_contrast_labels"] / counts["contrasts"]
        if counts["contrasts"]
        else None,
        "seconds": perf_counter() - start,
        "limits": "Gold words/sentences, exact train/dev sentence overlap excluded. Reference opportunities count repeated lemma/POS, not dependency edges or semantic identity. Label precision checks both endpoint annotations. Not a linguistic explanation comprehension or student learning study. Corpus annotation conventions and automatic lemma/feature annotation can affect results.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
