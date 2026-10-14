from unittest.mock import Mock

import pytest

from speech_engine.flashcard import FlashcardDeck
from speech_engine.grammar import GrammarChecker
from speech_engine.language_detector import LanguageDetector
from speech_engine.morphology import MorphologyEngine
from speech_engine.understanding import LinguisticPipeline
from test_morphology_integration import http_json


def pipeline():
    return LinguisticPipeline(MorphologyEngine(), GrammarChecker(), LanguageDetector())


def test_pipeline_aligns_rule_findings_without_changing_lexical_evidence():
    p = pipeline()
    p.morphology.analyze = Mock(wraps=p.morphology.analyze)
    text = "He bought a apple."
    result = p.analyze(text, "en-US")
    payload = result.to_dict(p.morphology, p.grammar)
    p.morphology.analyze.assert_called_once_with(text, "en-us")
    assert payload["language_selection"] == "explicit"
    assert payload["text"] == result.morphology.text == result.grammar.text
    issue_index = next(
        i for i, e in enumerate(payload["grammar"]["errors"]) if e["rule_id"] == "EN_ARTICLE_AN"
    )
    issue = payload["grammar"]["errors"][issue_index]
    assert issue["status"] == "heuristic"
    assert issue["token_indices"]
    for i in issue["token_indices"]:
        token = payload["morphology"]["tokens"][i]
        assert issue_index in token["grammar_issue_indices"]
        assert text[slice(*token["span"])] == token["surface"]
    cards = FlashcardDeck().extract_from_analysis(result.morphology)
    assert cards and all(c.example == text for c in cards)
    assert all(c.analysis["method"] == "english_affix_rules" for c in cards)
    assert p.morphology.analyze.call_count == 1


def test_unavailable_grammar_cannot_display_perfect_score():
    p = pipeline()
    payload = p.analyze("私は学生です。", "ja").to_dict(p.morphology, p.grammar)
    assert payload["grammar"]["limited"]
    assert payload["grammar"]["score"] is None
    assert payload["grammar"]["language_specific_rules"] == 0
    assert payload["morphology"]["capabilities"]["morphology"] == "unavailable"


def test_explicit_language_is_not_overridden_by_detector():
    p = pipeline()
    assert p.analyze("This is English text.", "es-MX").lang == "es-mx"
    detected = p.analyze("This is a sentence in the English language.", "auto")
    assert detected.lang == detected.detection.language
    assert detected.language_selection == "detected"
    with pytest.raises(ValueError):
        p.analyze("text", "not a tag")
    with pytest.raises(ValueError):
        p.analyze(12, "en")


def test_upstream_tokens_do_not_invent_original_card_context():
    analysis = MorphologyEngine().analyze_tokens(["cat"], "en")
    with pytest.raises(ValueError, match="original text"):
        FlashcardDeck().extract_from_analysis(analysis)


def test_integrated_http_result_matches_existing_endpoints():
    text = "He bought a apple."
    data = {"text": text, "lang": "en"}
    unified = http_json("/api/analyze", data)
    individual = http_json("/api/morphology", data)
    for combined, old in zip(unified["morphology"]["tokens"], individual["tokens"]):
        assert {k: v for k, v in combined.items() if k != "grammar_issue_indices"} == old
    assert unified["grammar"]["errors"]
    assert http_json("/api/grammar", {"text": "猫。", "lang": "ja"})["score"] is None
