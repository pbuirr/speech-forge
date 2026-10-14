import json
from io import BytesIO

import pytest

from speech_engine.exercise import generate_fill_in_blank
from speech_engine.flashcard import FlashcardDeck
from speech_engine.morphology import BPESegmenter, MorphologyEngine, POS
from webapp import server


def test_bpe_does_not_override_english_analysis():
    engine = MorphologyEngine()
    expected = engine.analyze_word("cats")
    engine.add_segmenter("en-US", BPESegmenter())
    result = engine.analyze_word("cats", "en-GB")
    assert result.root == expected.root == "cat"
    assert result.morphemes == expected.morphemes
    assert result.subwords == list("cats")
    assert result.confidence is None
    assert result.to_dict()["root_status"] == "heuristic"


def test_trained_bpe_is_not_morphology(tmp_path):
    segmenter = BPESegmenter(vocab_size=20)
    segmenter.fit(["casas casa casas casa"])
    path = tmp_path / "bpe.json"
    segmenter.save(path)
    restored = BPESegmenter()
    restored.load(path)
    assert restored.segment("casas") == segmenter.segment("casas")
    engine = MorphologyEngine()
    engine.add_segmenter("es-MX", restored)
    result = engine.analyze_word("Casas", "es-ES")
    assert result.subwords == restored.segment("casas")
    assert result.root == "Casas"
    assert result.morphemes == []
    assert result.features.to_dict() == {}
    assert result.method == "unavailable"
    assert result.to_dict()["root_status"] == "surface"


@pytest.mark.parametrize("lang", ["es", "fr", "de", "ru", "ar", "zh", "ja", "zz"])
def test_no_analyzer_does_not_invent_morphemes(lang):
    result = MorphologyEngine().analyze_word("word", lang)
    assert result.morphemes == []
    assert result.root == "word"
    assert result.confidence is None


@pytest.mark.parametrize("lang,text", [("ja", "私は学生です。"), ("zh", "我是学生。")])
def test_regional_tags_keep_tokenization(lang, text):
    engine = MorphologyEngine()
    assert engine.analyze(text, lang).tokens == engine.analyze(text, lang.upper() + "-XX").tokens


def test_regional_tags_keep_pos_and_capabilities():
    engine = MorphologyEngine()
    assert engine.analyze_word("the", "EN-US").pos == POS.DET
    assert engine.analyze_word("hablando", "es-MX").pos == POS.VERB
    assert engine.capabilities("en-US") == engine.capabilities("en")
    assert engine.capabilities("es")["morphology"] == "unavailable"


@pytest.mark.parametrize("word", ["", " ", "\t\n"])
def test_empty_word_has_clear_error(word):
    with pytest.raises(ValueError, match="non-whitespace"):
        MorphologyEngine().analyze_word(word)


def test_empty_sentence_and_structural_tokens():
    engine = MorphologyEngine()
    assert engine.analyze("").analyses == []
    results = engine.analyze("42 !").analyses
    assert [a.pos for a in results] == [POS.NUM, POS.PUNCT]
    assert all(a.method == "structural" for a in results)


def test_flashcards_keep_distinct_forms_and_label_predictions(tmp_path):
    engine = MorphologyEngine()
    engine.add_segmenter("en", BPESegmenter())
    deck = FlashcardDeck("test")
    cards = deck.extract_from_text("The cat cats cat.", "en", engine)
    assert [c.front for c in cards] == ["cat", "cats"]
    assert all("predicted" in c.back and "heuristic stem" in c.back for c in cards)
    assert all("needs-review" in c.tags for c in cards)
    path = tmp_path / "deck.json"
    deck.save(path)
    restored = FlashcardDeck.load(path)
    assert [c.to_dict() for c in restored.all()] == [c.to_dict() for c in cards]
    reviewed = restored.review_button(cards[0].id, "good")
    assert reviewed.srs.interval > 0
    assert generate_fill_in_blank(cards[0]).grade("cat").correct


def test_unsupported_morphology_card_uses_surface():
    engine = MorphologyEngine()
    engine.add_segmenter("es", BPESegmenter())
    cards = FlashcardDeck().extract_from_text("casas", "es", engine)
    assert cards[0].back == "casas (predicted noun)"
    assert "stem" not in cards[0].back


def http_json(path, payload=None, expected_status=200):
    """Run the real HTTP parser/router without requiring network access."""
    body = json.dumps(payload).encode() if payload is not None else b""
    method = "POST" if payload is not None else "GET"
    raw = (
        f"{method} {path} HTTP/1.0\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode() + body

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return BytesIO(raw)

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    server.Handler(connection, ("127.0.0.1", 0), None)
    headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    assert headers.startswith(f"HTTP/1.0 {expected_status}".encode())
    return json.loads(response)


def test_http_morphology_and_language_capabilities():
    for lang in ["en-US", "es-MX"]:
        result = http_json("/api/morphology", {"text": "cats", "lang": lang})
        assert result["limited"] == (lang == "es-MX")
        assert result["tokens"][0]["confidence"] is None
        assert "heuristics" in result["analysis_note"]
        if result["limited"]:
            assert result["tokens"][0]["morphemes"] == []
    languages = {x["code"]: x for x in http_json("/api/languages")["languages"]}
    assert languages["en"]["caps"]["morphology"] is True
    assert languages["es"]["caps"]["morphology"] is False
