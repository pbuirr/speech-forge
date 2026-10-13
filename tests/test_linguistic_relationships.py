import pytest
from speech_engine.linguistic_relationships import form_relationships
from speech_engine.morphology import MorphologyEngine
from speech_engine.understanding import morphology_payload, LinguisticPipeline
from speech_engine.grammar import GrammarChecker
from speech_engine.language_detector import LanguageDetector
from speech_engine.flashcard import FlashcardDeck
from test_linguistic_resources import row, compile_resource


@pytest.mark.parametrize('lang,first,second,lemma,feature,a,b', [
    ('en', 'cat', 'cats', 'cat', 'Number', 'Sing', 'Plur'),
    ('tr', 'ev', 'evler', 'ev', 'Number', 'Sing', 'Plur'),
    ('ru', 'дом', 'дома', 'дом', 'Case', 'Nom', 'Gen'),
    ('ar', 'كتاب', 'كتب', 'كتاب', 'Number', 'Sing', 'Plur'),
])
def test_corpus_form_comparison_reaches_pipeline_and_cards(tmp_path, lang, first, second, lemma, feature, a, b):
    engine = MorphologyEngine()
    engine.add_lexicon(compile_resource(tmp_path, row(1, first, lemma, feats=feature+'='+a) + row(2, second, lemma, feats=feature+'='+b), lang))
    checker = GrammarChecker()
    result = LinguisticPipeline(engine, checker, LanguageDetector()).analyze(first+' '+second, lang)
    links = result.to_dict(engine, checker)['morphology']['relationships']
    assert len(links) == 1
    assert links[0]['token_indices'] == [0, 1]
    assert links[0]['contrasts'][0]['feature'] == feature
    assert links[0]['confidence'] is None
    cards = FlashcardDeck().extract_from_understanding(result, engine, checker)
    assert all(card.analysis['context']['form_relationships'] == links for card in cards)


def test_missing_features_and_heuristic_stems_do_not_establish_relationships():
    engine = MorphologyEngine()
    assert morphology_payload(engine.analyze('cat cats'), engine)['relationships'] == []
    token = dict(surface='x', root='x', pos='NOUN', root_status='corpus_attested', pos_status='corpus_attested', candidates=[{}], ud_features={'Number': 'Sing'})
    other = {**token, 'ud_features': {}}
    assert form_relationships([token, other])[0]['contrasts'] == []
    assert form_relationships([token, {**other, 'root_status': 'surface'}]) == []
    assert form_relationships([token, {**other, 'pos': 'VERB'}]) == []


def test_ambiguity_and_linear_size_are_preserved():
    token = dict(surface='x', root='x', pos='NOUN', root_status='predicted', pos_status='predicted', candidates=[{}, {}], ud_features={'Case': 'Acc,Nom'}, method='corpus_context')
    other = {**token, 'ud_features': {'Case': 'Nom'}}
    links = form_relationships([token, other] * 100)
    assert len(links) == 199
    assert links[0]['status'] == 'predicted'
    assert links[0]['contrasts'][0]['overlapping_values']
    assert len(token['candidates']) == 2
