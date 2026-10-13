import json
from pathlib import Path
import random
from unittest.mock import Mock

import pytest

from speech_engine.exercise import generate_fill_in_blank, generate_multiple_choice, generate_word_order
from speech_engine.flashcard import FlashcardDeck
from speech_engine.morphology import MorphologyEngine
from speech_engine.text_matching import answer_key
from test_morphology_integration import http_json
from webapp import server

CASES = json.loads((Path(__file__).parent / 'fixtures/learning_cases.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['lang'] + ':' + case['front'])
def test_multilingual_cloze_preserves_original_context(case):
    span = None
    if case.get('annotated_span'):
        left = case['text'].index(case['front'])
        span = {'span': [left, left + len(case['front'])], 'span_unit': 'unicode_codepoints'}
    card = FlashcardDeck().add(case['front'], 'authored reference', case['lang'], example=case['text'], analysis=span)
    exercise = generate_fill_in_blank(card)
    assert exercise.prompt == case['expected_prompt']
    assert exercise.span is not None
    assert card.example[slice(*exercise.span)] == exercise.answer
    assert exercise.grade(case['front']).correct


@pytest.mark.parametrize('lang,expected,given,correct', [
    ('fr', 'café', 'cafe\u0301', True), ('fr', 'café', 'cafe', False),
    ('tr', 'İstanbul', 'istanbul', True), ('tr', 'I', 'ı', True),
    ('tr', 'I', 'i', False), ('az', 'İ', 'i', True),
    ('de', 'Straße', 'STRASSE', False), ('fa', 'می\u200cروم', 'میروم', False),
    ('ja', 'カ', 'ｶ', False), ('en', 'Cat', 'cat', True),
])
def test_answer_matching_accepts_encoding_equivalence_not_linguistic_changes(lang, expected, given, correct):
    card = FlashcardDeck().add(expected, 'reference', lang)
    assert generate_fill_in_blank(card).grade(given).correct == correct


def test_card_deduplication_preserves_turkish_letter_distinctions_and_unicode_equivalence():
    engine = MorphologyEngine()
    assert [c.front for c in FlashcardDeck().extract_from_text('I ı İ i', 'tr', engine)] == ['I', 'İ']
    assert [c.front for c in FlashcardDeck().extract_from_text('café cafe\u0301', 'fr', engine)] == ['café']


def test_pipeline_card_span_survives_persistence_and_no_model_is_rerun(tmp_path):
    engine = MorphologyEngine()
    deck = FlashcardDeck()
    card = next(c for c in deck.extract_from_text('🐈 The cat rests.', 'en', engine) if c.front == 'cat')
    path = tmp_path / 'deck.json'
    deck.save(path)
    card = FlashcardDeck.load(path).get(card.id)
    engine.analyze = Mock(side_effect=AssertionError('Exercise must not rerun inference'))
    exercise = generate_fill_in_blank(card, engine)
    assert exercise.prompt == '🐈 The ____ rests.'
    assert exercise.source == 'analysis_span'
    engine.analyze.assert_not_called()


def test_span_selects_recorded_occurrence_and_invalid_metadata_falls_back():
    deck = FlashcardDeck()
    card = deck.add('cat', 'reference', example='cat and cat.',
                    analysis={'span': [8, 11], 'span_unit': 'unicode_codepoints'})
    assert generate_fill_in_blank(card).prompt == 'cat and ____.'
    card.analysis['span'] = [0, 4]
    assert generate_fill_in_blank(card).prompt == '____ and cat.'
    card.example = 'cathedral'
    card.analysis['span'] = [0, 3]
    assert generate_fill_in_blank(card).source == 'definition'
    card.analysis['span'] = [False, 3]
    assert generate_fill_in_blank(card).source == 'definition'


def test_mcq_distractors_are_distinct_same_language_and_same_gloss_is_not_wrong():
    deck = FlashcardDeck()
    target = deck.add('café', 'coffee', 'fr', pos='NOUN')
    deck.add('cafe\u0301', 'coffee beverage', 'fr', pos='NOUN')
    deck.add('tasse', 'cup', 'fr-FR', pos='NOUN')
    deck.add('TASSE', 'drinking vessel', 'fr', pos='NOUN')
    deck.add('chat', 'cat', 'fr', pos='NOUN')
    deck.add('coffee', 'coffee', 'en', pos='NOUN')
    deck.add('synonyme', 'coffee', 'fr', pos='NOUN')
    exercise = generate_multiple_choice(target, deck, rng=random.Random(2))
    assert set(exercise.choices) == {'café', 'tasse', 'chat'}
    assert exercise.grade(exercise.answer_index).correct
    with pytest.raises(ValueError):
        generate_multiple_choice(target, deck, n_distractors=-1)
    assert generate_multiple_choice(FlashcardDeck().add('only', 'only'), FlashcardDeck()) is None


def test_word_order_is_bounded_and_only_grades_reference_reconstruction():
    deck = FlashcardDeck()
    assert generate_word_order(deck.add('go', 'move', example='go go go')) is None
    rng = Mock()
    rng.shuffle = lambda words: None
    exercise = generate_word_order(deck.add('cat', 'animal', example='The cat sleeps.'), rng)
    assert exercise.scrambled != ['The', 'cat', 'sleeps']
    assert sorted(exercise.scrambled) == ['The', 'cat', 'sleeps']
    assert exercise.grade('the cat sleeps').correct
    assert not exercise.grade('sleeps the cat').correct


def test_http_exercise_reuses_persisted_unicode_spans(monkeypatch):
    deck = FlashcardDeck()
    card = deck.add('学习', 'study', 'zh', example='我爱学习。',
                    analysis={'span': [2, 4], 'span_unit': 'unicode_codepoints'})
    monkeypatch.setattr(server, '_load_deck', lambda: deck)
    payload = http_json('/api/exercise', {'card_id': card.id, 'kind': 'fill'})
    assert payload['prompt'] == '我爱____。'
    assert payload['source'] == 'analysis_span'
    assert payload['span'] == [2, 4]
    unavailable = http_json('/api/exercise', {'card_id': card.id, 'kind': 'mcq'}, expected_status=400)
    assert 'same-language distractors' in unavailable['error']
