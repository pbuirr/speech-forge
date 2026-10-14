import pytest
from speech_engine.sentence_boundaries import validate_ranges
from speech_engine.morphology import MorphologyEngine


@pytest.mark.parametrize(
    "text,expected",
    [
        ("It costs 3.14. Really?", ["It costs 3.14.", "Really?"]),
        ('He said "Go!" Then left.', ['He said "Go!"', "Then left."]),
        ('Go. "Really?"', ["Go.", '"Really?"']),
        ("Wait... Go!!", ["Wait...", "Go!!"]),
        ("你好。再见！", ["你好。", "再见！"]),
        ("यह ठीक है। फिर मिलेंगे।", ["यह ठीक है।", "फिर मिलेंगे।"]),
        ("مرحبا؟ نعم.", ["مرحبا؟", "نعم."]),
    ],
)
def test_shared_sentence_spans(text, expected):
    engine = MorphologyEngine()
    result = engine.analyze(text, "zz")
    assert [
        text[result.analyses[a].span[0] : result.analyses[b - 1].span[1]]
        for a, b in result.sentence_ranges
    ] == expected
    assert len(result.tokens) == sum(b - a for a, b in result.sentence_ranges)


def test_corpus_abbreviation_hints_and_supplied_boundaries(tmp_path):
    from test_linguistic_resources import row, compile_resource

    engine = MorphologyEngine()
    engine.add_lexicon(compile_resource(tmp_path, row(1, "Dr.", "Dr.", "PROPN") + "\n"))
    result = engine.analyze("Dr. Smith left. Bye.", "en")
    assert len(result.sentence_ranges) == 2
    supplied = engine.analyze_tokens(["Dr.", "Smith", "left", ".", "Bye", "."], "en")
    assert supplied.sentence_ranges == [(0, 6)]
    assert supplied.text is None
    with pytest.raises(ValueError):
        validate_ranges([(0, 2), (3, 4)], 4)
    with pytest.raises(ValueError):
        validate_ranges([(0, 2)], 3)


def test_pos_and_syntax_use_same_ranges():
    from speech_engine.dependency import DependencyParser, syntax_payload
    from unittest.mock import Mock

    engine = MorphologyEngine()
    engine.tagger.fit([[("costs", "VERB"), ("3", "NUM"), (".", "PUNCT"), ("14", "NUM")]], "en")
    original = engine.tagger.tag_with_uncertainty
    engine.tagger.tag_with_uncertainty = Mock(wraps=original)
    analysis = engine.analyze("It costs 3.14. Again!", "en")
    model = DependencyParser("en")
    model.labels = ["dep", "root"]
    syntax = syntax_payload(analysis, {"en": model})
    assert engine.tagger.tag_with_uncertainty.call_args.args[2] == analysis.sentence_ranges
    assert sum(a["head_token_index"] is None for a in syntax["arcs"]) == len(
        analysis.sentence_ranges
    )
    assert all(
        analysis.sentence_ranges[a["sentence_index"]][0]
        <= a["token_index"]
        < analysis.sentence_ranges[a["sentence_index"]][1]
        for a in syntax["arcs"]
    )


def test_sentence_evidence_survives_http_and_card_persistence(tmp_path, monkeypatch):
    from speech_engine.understanding import LinguisticPipeline
    from speech_engine.grammar import GrammarChecker
    from speech_engine.language_detector import LanguageDetector
    from speech_engine.flashcard import FlashcardDeck
    from webapp import server
    from test_morphology_integration import http_json

    engine = MorphologyEngine()
    checker = GrammarChecker()
    text = "Cats sleep. Dogs bark!"
    result = LinguisticPipeline(engine, checker, LanguageDetector()).analyze(text, "en")
    payload = result.to_dict(engine, checker)
    assert [s["text"] for s in payload["sentences"]] == ["Cats sleep.", "Dogs bark!"]
    assert all(s["status"] == "heuristic" for s in payload["sentences"])
    deck = FlashcardDeck()
    cards = deck.extract_from_understanding(result, engine, checker)
    target = tmp_path / "deck.json"
    deck.save(target)
    restored = FlashcardDeck.load(target)
    assert [c.analysis for c in restored.all()] == [c.analysis for c in cards]
    assert all(c.analysis["context"]["sentences"] for c in cards)
    monkeypatch.setattr(server, "morph", engine)
    assert (
        http_json("/api/analyze", {"text": text, "lang": "en"})["sentences"] == payload["sentences"]
    )
