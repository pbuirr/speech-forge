from speech_engine.flashcard import FlashcardDeck
from speech_engine.morphology import MorphologyEngine
from speech_engine.grammar import GrammarChecker
from speech_engine.language_detector import LanguageDetector
from speech_engine.understanding import LinguisticPipeline
from speech_engine.exercise import generate_fill_in_blank


def test_distinct_uses_and_occurrences_survive_learning_roundtrip(tmp_path):
    engine = MorphologyEngine()
    analysis = engine.analyze('record record record', 'en')
    # Explicit authored analyses test transport, not model accuracy.
    for token, pos in zip(analysis.analyses, ['NOUN', 'VERB', 'NOUN']):
        token.pos = pos
        token.pos_method = 'bigram_hmm'
    deck = FlashcardDeck()
    cards = deck.extract_from_analysis(analysis)
    assert len(cards) == 2
    assert [o['token_index'] for o in cards[0].analysis['occurrences']] == [0, 2]
    assert cards[1].analysis['occurrences'][0]['span'] == [7, 13]
    assert generate_fill_in_blank(cards[1]).prompt == 'record ____ record'
    path = tmp_path / 'deck.json'
    deck.save(path)
    restored = FlashcardDeck.load(path)
    assert [c.to_dict() for c in restored.all()] == [c.to_dict() for c in cards]


def test_shared_rule_findings_are_retained_without_rerunning_models():
    engine = MorphologyEngine(); checker = GrammarChecker()
    result = LinguisticPipeline(engine, checker, LanguageDetector()).analyze('This is is useful.', 'en')
    # Extraction consumes existing results; it must not perform inference again.
    engine.analyze = lambda *a, **k: (_ for _ in ()).throw(AssertionError('rerun'))
    checker.check = engine.analyze
    cards = FlashcardDeck().extract_from_understanding(result, engine, checker)
    assert cards
    for card in cards:
        context = card.analysis['context']
        assert context['language'] == 'en'
        assert len(context['text_sha256']) == 64
        assert context['span_unit'] == 'unicode_codepoints'
        for issue in context['grammar_findings']:
            assert issue['status'] == 'heuristic'
            assert any(o['token_index'] in issue['token_indices'] for o in card.analysis['occurrences'])


def test_multilingual_occurrence_offsets_preserve_source():
    engine = MorphologyEngine(); checker = GrammarChecker()
    for lang, text in [('tr', 'evler evler'), ('ar', 'كتاب كتاب'), ('es', 'casas casas')]:
        result = LinguisticPipeline(engine, checker, LanguageDetector()).analyze(text, lang)
        cards = FlashcardDeck().extract_from_understanding(result, engine, checker)
        assert cards
        for card in cards:
            for occurrence in card.analysis['occurrences']:
                left, right = occurrence['span']
                assert text[left:right] == occurrence['surface']
            assert card.analysis['context']['language'] == lang


def test_extract_endpoint_persists_shared_context(monkeypatch):
    from webapp import server
    from test_morphology_integration import http_json
    deck = FlashcardDeck()
    monkeypatch.setattr(server, '_load_deck', lambda: deck)
    monkeypatch.setattr(deck, 'save', lambda path: None)
    response = http_json('/api/extract', {'text': 'Cats chase cats.', 'lang': 'en'})
    assert deck.all()
    assert all(c.analysis['context']['language_selection'] == 'explicit' for c in deck.all())
