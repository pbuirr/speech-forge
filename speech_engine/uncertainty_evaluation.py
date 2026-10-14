"""Held-out evaluation of POS confidence: calibration and selective accuracy."""

from hashlib import sha256
import json
from pathlib import Path

from speech_engine.linguistic_resources import read_conllu
from speech_engine.morphology import MorphologyEngine


def sentence_fingerprint(sentence):
    return sha256(json.dumps([w.form for w in sentence], ensure_ascii=False).encode()).hexdigest()


def sentence_fingerprints(path):
    fingerprints = set()
    words = []
    for word in read_conllu(Path(path).read_text(encoding="utf-8")):
        if word.token_id == 1 and words:
            fingerprints.add(sentence_fingerprint(words))
            words = []
        words.append(word)
    if words:
        fingerprints.add(sentence_fingerprint(words))
    return fingerprints


def collect_pos_scores(lexicon, path, excluded_sentences=None):
    raw = Path(path).read_bytes()
    checksum = sha256(raw).hexdigest()
    if checksum == lexicon.source["sha256"]:
        raise ValueError("Use held-out data, not the training source")
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    rows = []
    fingerprints = set()
    excluded = unscored = foreign = sentences = 0
    words = []

    def consume(sentence):
        nonlocal excluded, unscored, foreign, sentences
        sentences += 1
        fingerprint = sentence_fingerprint(sentence)
        fingerprints.add(fingerprint)
        if excluded_sentences and fingerprint in excluded_sentences:
            excluded += 1
            return
        analyses = engine.analyze_tokens([w.form for w in sentence], lexicon.lang).analyses
        for word, analysis in zip(sentence, analyses):
            if word.language and word.language != lexicon.lang:
                foreign += 1
                continue
            evidence = analysis.pos_uncertainty
            if evidence is None or not evidence["applies_to_returned_pos"]:
                unscored += 1
                continue
            rows.append(
                {
                    "score": evidence["model_probability"],
                    "known": evidence["known_form"],
                    "correct": analysis.pos == word.pos,
                }
            )

    for word in read_conllu(raw.decode("utf-8")):
        if word.token_id == 1 and words:
            consume(words)
            words = []
        words.append(word)
    if words:
        consume(words)
    return (
        rows,
        fingerprints,
        {
            "sha256": checksum,
            "sentences": sentences,
            "excluded_overlap_sentences": excluded,
            "unscored_tokens": unscored,
            "foreign_tokens": foreign,
        },
    )


def reliability_metrics(rows, calibration=None):
    scored = []
    for row in rows:
        if calibration:
            estimate = calibration.estimate(row["score"], row["known"])
            if estimate is None:
                continue
            score = estimate["accuracy_estimate"]
        else:
            score = row["score"]
        scored.append((score, row["correct"]))
    if not scored:
        return {"tokens": 0}
    count = len(scored)
    bins = []
    for index in range(10):
        members = [(p, correct) for p, correct in scored if min(9, int(p * 10)) == index]
        if members:
            bins.append(
                {
                    "bin": index,
                    "tokens": len(members),
                    "mean_score": sum(p for p, _ in members) / len(members),
                    "accuracy": sum(correct for _, correct in members) / len(members),
                }
            )
    selective = []
    for threshold in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
        kept = [correct for score, correct in scored if score >= threshold]
        selective.append(
            {
                "threshold": threshold,
                "tokens": len(kept),
                "coverage": len(kept) / count,
                "accuracy": sum(kept) / len(kept) if kept else None,
            }
        )
    return {
        "tokens": count,
        "accuracy": sum(correct for _, correct in scored) / count,
        "mean_score": sum(p for p, _ in scored) / count,
        "binary_brier": sum((p - correct) ** 2 for p, correct in scored) / count,
        "ece_10_bins": sum(b["tokens"] * abs(b["mean_score"] - b["accuracy"]) for b in bins)
        / count,
        "bins": bins,
        "selective_accuracy": selective,
    }
