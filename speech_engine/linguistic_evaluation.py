"""Held-out evaluation of corpus lookups, reporting coverage and accuracy separately."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import unicodedata

from speech_engine.linguistic_resources import CorpusLexicon, read_conllu


def evaluate_raw_text(lexicon: CorpusLexicon, test_path: str | Path) -> dict:
    """Score actual text input; reject unalignable annotations explicitly.

    Alignment requires exact concatenative word forms, allowing only whitespace
    and NFC normalization. Non-concatenative multiword expansions are excluded,
    never assigned fabricated offsets. Gold labels do not affect predictions.
    """
    from speech_engine.morphology import MorphologyEngine

    raw = Path(test_path).read_bytes()
    if sha256(raw).hexdigest() == lexicon.source["sha256"]:
        raise ValueError("Evaluation file is the training source; use a held-out split")
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    totals = dict(
        sentences=0,
        aligned_sentences=0,
        excluded_sentences=0,
        gold_tokens=0,
        predicted_tokens=0,
        matched_tokens=0,
        pos_correct=0,
        exact_tokenization_sentences=0,
    )
    failures = []
    for block in re.split(r"\r?\n\s*\r?\n", raw.decode("utf-8").lstrip("\ufeff")):
        words = list(read_conllu(block))
        if not words:
            continue
        totals["sentences"] += 1
        text_match = re.search(r"^# text = (.*)$", block, re.MULTILINE)
        sentence_text = text_match.group(1).rstrip("\r") if text_match else None
        compact = lambda value: "".join(
            c for c in unicodedata.normalize("NFC", value) if not c.isspace()
        )
        if sentence_text is None or compact(sentence_text) != "".join(
            compact(w.form) for w in words
        ):
            totals["excluded_sentences"] += 1
            continue

        # Positions in the whitespace-free NFC stream permit fair comparison
        # even when surface offsets differ under Unicode normalization.
        def spans(forms):
            position = 0
            result = {}
            for item in forms:
                form = compact(item.form if hasattr(item, "form") else item.surface)
                end = position + len(form)
                result[(position, end, form)] = item
                position = end
            return result

        gold = spans(words)
        predicted = spans(engine.analyze(sentence_text, lexicon.lang).analyses)
        shared = gold.keys() & predicted.keys()
        totals["aligned_sentences"] += 1
        totals["gold_tokens"] += len(gold)
        totals["predicted_tokens"] += len(predicted)
        totals["matched_tokens"] += len(shared)
        totals["pos_correct"] += sum(gold[k].pos == predicted[k].pos for k in shared)
        totals["exact_tokenization_sentences"] += gold.keys() == predicted.keys()
        if gold.keys() != predicted.keys() and len(failures) < 20:
            failures.append(
                {
                    "sentence_id": words[0].sentence_id,
                    "text": sentence_text,
                    "gold": [w.form for w in words],
                    "predicted": [a.surface for a in predicted.values()],
                }
            )
    ratio = lambda n, d: n / d if d else None
    return {
        **totals,
        "token_precision": ratio(totals["matched_tokens"], totals["predicted_tokens"]),
        "token_recall": ratio(totals["matched_tokens"], totals["gold_tokens"]),
        "token_f1": ratio(
            2 * totals["matched_tokens"], totals["gold_tokens"] + totals["predicted_tokens"]
        ),
        "pos_accuracy_all_gold_tokens": ratio(totals["pos_correct"], totals["gold_tokens"]),
        "sample_tokenization_failures": failures,
        "limits": "Gold sentence boundaries, actual raw sentence text. Exact token spans and UPOS; unmatched gold tokens count as errors. Missing text or non-concatenative annotations excluded explicitly. No sentence splitting evaluation.",
    }


def evaluate_lexicon(lexicon: CorpusLexicon, test_path: str | Path) -> dict:
    raw = Path(test_path).read_bytes()
    checksum = sha256(raw).hexdigest()
    if checksum == lexicon.source["sha256"]:
        raise ValueError("Evaluation file is the training source; use a held-out split")
    eligible = covered = decided = correct = candidate_correct = ambiguous = 0
    excluded = 0
    lemma_decided = lemma_correct = pos_decided = pos_correct = 0
    failures = []
    for word in read_conllu(raw.decode("utf-8")):
        if word.lemma is None or (word.language and word.language != lexicon.lang):
            excluded += 1
            continue
        eligible += 1
        candidates = lexicon.lookup(word.form)
        expected = (word.lemma, word.pos, dict(word.features))
        matches = [(c["lemma"], c["pos"], c["features"]) == expected for c in candidates]
        covered += bool(candidates)
        lemmas = {c["lemma"] for c in candidates}
        tags = {c["pos"] for c in candidates}
        lemma_decided += len(lemmas) == 1
        lemma_correct += len(lemmas) == 1 and word.lemma in lemmas
        pos_decided += len(tags) == 1
        pos_correct += len(tags) == 1 and word.pos in tags
        candidate_correct += any(matches)
        ambiguous += len(candidates) > 1
        decided += len(candidates) == 1
        hit = len(candidates) == 1 and matches[0]
        correct += hit
        if not hit and len(failures) < 20:
            failures.append(
                {
                    "form": word.form,
                    "sentence_id": word.sentence_id,
                    "token_id": word.token_id,
                    "reason": "unknown"
                    if not candidates
                    else "ambiguous"
                    if len(candidates) > 1
                    else "annotation_mismatch",
                }
            )

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "language": lexicon.lang,
        "training_source": lexicon.source,
        "test_file": Path(test_path).name,
        "test_sha256": checksum,
        "eligible_tokens": eligible,
        "excluded_tokens": excluded,
        "covered_tokens": covered,
        "decided_tokens": decided,
        "ambiguous_tokens": ambiguous,
        "correct_decisions": correct,
        "coverage": ratio(covered, eligible),
        "decision_coverage": ratio(decided, eligible),
        "accuracy_on_decisions": ratio(correct, decided),
        "accuracy_all_tokens": ratio(correct, eligible),
        "candidate_recall": ratio(candidate_correct, eligible),
        "lemma_decision_coverage": ratio(lemma_decided, eligible),
        "lemma_accuracy_on_decisions": ratio(lemma_correct, lemma_decided),
        "lemma_accuracy_all_tokens": ratio(lemma_correct, eligible),
        "pos_decision_coverage": ratio(pos_decided, eligible),
        "pos_accuracy_on_decisions": ratio(pos_correct, pos_decided),
        "sample_failures": failures,
        "limits": "Exact lemma+UPOS+feature agreement against held-out annotations. No contextual disambiguation or dependency parsing. Distinct checksums do not rule out overlapping sentences; curate splits independently.",
    }


def evaluate_context(lexicon: CorpusLexicon, test_path: str | Path) -> dict:
    """Compare the production analyzer with the heuristic baseline on gold tokens.

    Sentence/token boundaries come from annotations: this does not evaluate raw
    text tokenization. Gold labels never train or influence the decoder.
    """
    from speech_engine.morphology import MorphologyEngine

    raw = Path(test_path).read_bytes()
    if sha256(raw).hexdigest() == lexicon.source["sha256"]:
        raise ValueError("Evaluation file is the training source; use a held-out split")
    baseline = MorphologyEngine()
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    totals = {
        "tokens": 0,
        "baseline_pos_correct": 0,
        "context_pos_correct": 0,
        "lemma_decisions": 0,
        "baseline_lemma_correct_on_same_tokens": 0,
        "context_lemma_correct": 0,
        "context_predictions": 0,
    }

    def evaluate_sentence(sentence):
        tokens = [word.form for word in sentence]
        before = baseline.analyze_tokens(tokens, lexicon.lang).analyses
        after = engine.analyze_tokens(tokens, lexicon.lang).analyses
        for word, old, new in zip(sentence, before, after):
            if word.lemma is None or (word.language and word.language != lexicon.lang):
                continue
            totals["tokens"] += 1
            old_pos = {"PREP": "ADP", "CONJ": "CCONJ"}.get(old.pos, old.pos)
            totals["baseline_pos_correct"] += old_pos == word.pos
            totals["context_pos_correct"] += new.pos == word.pos
            totals["context_predictions"] += new.method == "corpus_context"
            if new.to_dict()["root_status"] in ("predicted", "corpus_attested"):
                totals["lemma_decisions"] += 1
                totals["baseline_lemma_correct_on_same_tokens"] += old.root == word.lemma
                totals["context_lemma_correct"] += new.root == word.lemma

    sentence = []
    for word in read_conllu(raw.decode("utf-8")):
        if word.token_id == 1 and sentence:
            evaluate_sentence(sentence)
            sentence = []
        sentence.append(word)
    if sentence:
        evaluate_sentence(sentence)

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        **totals,
        "baseline_pos_accuracy": ratio(totals["baseline_pos_correct"], totals["tokens"]),
        "context_pos_accuracy": ratio(totals["context_pos_correct"], totals["tokens"]),
        "lemma_decision_coverage": ratio(totals["lemma_decisions"], totals["tokens"]),
        "baseline_lemma_accuracy_on_same_tokens": ratio(
            totals["baseline_lemma_correct_on_same_tokens"], totals["lemma_decisions"]
        ),
        "context_lemma_accuracy_on_decisions": ratio(
            totals["context_lemma_correct"], totals["lemma_decisions"]
        ),
        "limits": "Gold sentence/word boundaries. POS predictions use only training counts; lemma selection uses only candidate POS and field consensus. Features or senses sharing a POS may remain unresolved. This is not general semantic understanding.",
    }
