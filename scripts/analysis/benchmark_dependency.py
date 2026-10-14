"""Time the dependency parser on the same cached inputs, for latency comparisons."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.dependency import DependencyParser, read_dependencies
from speech_engine.morphology import MorphologyEngine
from speech_engine.uncertainty_evaluation import sentence_fingerprint, sentence_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "baseline",
        "candidate",
        "corpus",
        "training",
        "development",
        "resources",
        "output",
    ]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    models = [DependencyParser.load(args.baseline), DependencyParser.load(args.candidate)]
    checksum = sha256(Path(args.training).read_bytes()).hexdigest()
    if (
        any(m.source.get("training_sha256") != checksum for m in models)
        or models[0].language != models[1].language
    ):
        raise ValueError("Compare models trained on the same language and corpus")
    engine = MorphologyEngine()
    engine.load_resources(args.resources)
    excluded = sentence_fingerprints(args.training) | sentence_fingerprints(args.development)
    evidence = []
    for sentence in read_dependencies(args.corpus):
        if sentence_fingerprint(sentence.words) in excluded:
            continue
        evidence.append(
            [
                a.to_dict()
                for a in engine.analyze_tokens(
                    [w.form for w in sentence.words], models[0].language
                ).analyses
            ]
        )
        if len(evidence) == 400:
            break
    if not evidence:
        raise ValueError("No held-out sentences")
    durations = [[], []]
    for model in models:
        for tokens in evidence[:10]:
            model.parse(tokens)
    for repeat in range(4):
        for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
            start = perf_counter()
            for tokens in evidence:
                models[index].parse(tokens)
            durations[index].append(perf_counter() - start)
    report = {
        "sentences": len(evidence),
        "tokens": sum(map(len, evidence)),
        "corpus_sha256": sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "models": {
            name: {
                "sha256": sha256(Path(path).read_bytes()).hexdigest(),
                "bytes": Path(path).stat().st_size,
                "seconds": times,
                "median_seconds": statistics.median(times),
            }
            for name, path, times in zip(
                ["baseline", "candidate"], [args.baseline, args.candidate], durations
            )
        },
        "limits": "Parser-only warmed latency; identical cached production inputs, four alternating-order runs. Not full application latency or a hardware-independent speed claim.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
