from itertools import product
import json
import math

import pytest

from speech_engine.flashcard import FlashcardDeck
from speech_engine.morphology import MorphologyEngine, POSTagger
from speech_engine.pos_uncertainty import forward_backward, POSCalibration, pos_review_recommended
from speech_engine.uncertainty_evaluation import collect_pos_scores, reliability_metrics
from test_linguistic_resources import compile_resource, row
from test_morphology_integration import http_json
from webapp import server


def test_forward_backward_matches_exhaustive_paths_including_end_transition():
    lattice = [{'NOUN': math.log(.8), 'VERB': math.log(.2)},
               {'NOUN': math.log(.3), 'VERB': math.log(.7)},
               {'NOUN': math.log(.4), 'VERB': math.log(.6)}]
    transitions = {(a, b): math.log(.8 if a == b else .2)
                   for a in ['BOS', 'NOUN', 'VERB'] for b in ['NOUN', 'VERB', 'EOS']}
    transitions['VERB', 'EOS'] = math.log(.9)
    paths = []
    for path in product(['NOUN', 'VERB'], repeat=3):
        edges = list(zip(['BOS', *path], [*path, 'EOS']))
        mass = math.exp(sum(lattice[i][tag] for i, tag in enumerate(path)) + sum(transitions[e] for e in edges))
        paths.append((path, mass))
    total = sum(mass for _, mass in paths)
    actual = forward_backward(lattice, transitions)
    for i, distribution in enumerate(actual):
        assert sum(distribution.values()) == pytest.approx(1)
        for tag, probability in distribution.items():
            assert probability == pytest.approx(sum(mass for path, mass in paths if path[i] == tag) / total)


def test_long_lattice_is_stable_and_tagger_predictions_are_preserved():
    tagger = POSTagger()
    tagger.fit([[('They', 'PRON'), ('run', 'VERB'), ('.', 'PUNCT')], [('a', 'DET'), ('run', 'NOUN')]])
    tokens = ['They', 'run', '.', 'a', 'run']
    predicted = tagger.tag_with_uncertainty(tokens)
    assert [(token, pos) for token, pos, _ in predicted] == tagger.tag(tokens)
    assert predicted[1][2]['distribution'] != predicted[-1][2]['distribution']
    lattice = [{'NOUN': -1000., 'VERB': -1001.}] * 300
    transitions = {(a, b): -2. for a in ['BOS', 'NOUN', 'VERB'] for b in ['NOUN', 'VERB', 'EOS']}
    assert all(sum(p.values()) == pytest.approx(1) for p in forward_backward(lattice, transitions))
    assert forward_backward([], {}) == []
    assert MorphologyEngine().analyze_word('cat').pos_uncertainty is None


def test_calibration_pools_violations_preserves_strata_and_persists(tmp_path):
    rows = [(0.1, True, i < 9) for i in range(10)] + [(0.5, True, i < 1) for i in range(10)]
    rows += [(0.9, True, True)] * 10 + [(1., False, False)] * 5
    calibration = POSCalibration.fit(rows, 'en', 'a' * 64, 'b' * 64, 'c' * 64)
    assert calibration.estimate(.1, True)['accuracy_estimate'] == .5
    assert calibration.estimate(.5, True)['development_observations'] == 20
    assert calibration.estimate(1., False)['accuracy_estimate'] == 0
    path = tmp_path / 'en.json'
    calibration.save(path)
    assert POSCalibration.load(path).estimate(.5, True) == calibration.estimate(.5, True)
    data = json.loads(path.read_text())
    data['scoring_version'] = 'future'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        POSCalibration.load(path)
    for score in [float('nan'), float('inf'), -1, 2, True]:
        with pytest.raises(ValueError):
            calibration.estimate(score, True)


def test_calibration_cannot_be_applied_to_another_model_or_survive_retraining(tmp_path):
    lexicon = compile_resource(tmp_path, row(1, 'run', 'run', 'VERB'))
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    calibration = POSCalibration.fit([(1., True, False)] * 10, 'en', lexicon.source['sha256'], 'b' * 64, engine.tagger.model_fingerprint('en'))
    engine.tagger.add_calibration(calibration)
    analysis = engine.analyze_word('run')
    assert analysis.confidence is None
    assert analysis.to_dict()['pos_review_recommended']
    assert 'pos-uncertain' in FlashcardDeck().extract_from_text('run', 'en', engine)[0].tags
    engine.tagger.fit([[('run', 'NOUN')]])
    assert 'calibration' not in engine.analyze_word('run').pos_uncertainty
    with pytest.raises(ValueError, match='training model'):
        engine.tagger.add_calibration(calibration)
    assert pos_review_recommended({'applies_to_returned_pos': False, 'calibration': {'accuracy_estimate': 0}}) is None
    # A modified compiled model must also reject the old calibration even when
    # its declared corpus checksum is unchanged.
    lexicon.transitions[0][2] += 1
    engine.add_lexicon(lexicon)
    with pytest.raises(ValueError, match='training model'):
        engine.tagger.add_calibration(calibration)


def test_overlap_and_calibration_metrics_do_not_change_predictions(tmp_path):
    lexicon = compile_resource(tmp_path, row(1, 'run', 'run', 'VERB'))
    test = tmp_path / 'dev.conllu'
    test.write_text(row(1, 'run', 'run', 'NOUN') + '\n' + row(1, 'cat', 'cat', 'NOUN'))
    rows, fingerprints, info = collect_pos_scores(lexicon, test)
    assert rows[0]['score'] == 1 and not rows[0]['correct']
    excluded, _, excluded_info = collect_pos_scores(lexicon, test, fingerprints)
    assert not excluded and excluded_info['excluded_overlap_sentences'] == 2
    calibration = POSCalibration.fit([(r['score'], r['known'], r['correct']) for r in rows], 'en', lexicon.source['sha256'], info['sha256'], 'c' * 64)
    metrics = reliability_metrics(rows, calibration)
    assert metrics['binary_brier'] == 0
    assert metrics['accuracy'] == reliability_metrics(rows)['accuracy']
    with pytest.raises(ValueError):
        collect_pos_scores(lexicon, tmp_path / 'train.conllu')


def test_atomic_calibration_loading_and_http_card_evidence(tmp_path, monkeypatch):
    lexicon = compile_resource(tmp_path, row(1, 'run', 'run', 'VERB'))
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    calibration = POSCalibration.fit([(1., True, False)] * 10, 'en', lexicon.source['sha256'], 'b' * 64, engine.tagger.model_fingerprint('en'))
    directory = tmp_path / 'calibration'
    calibration.save(directory / 'en.json')
    assert engine.tagger.load_calibrations(directory) == ['en']
    wrong = POSCalibration.fit([(1., True, True)], 'fr', 'c' * 64, 'd' * 64, 'e' * 64)
    wrong.save(directory / 'fr.json')
    with pytest.raises(ValueError, match='mismatched'):
        engine.tagger.load_calibrations(directory)
    assert engine.analyze_word('run').pos_uncertainty['calibration']['accuracy_estimate'] == 0
    assert engine.analyze_word('chat', 'fr').pos_uncertainty is None
    monkeypatch.setattr(server, 'morph', engine)
    monkeypatch.setattr(server, 'DECK_PATH', tmp_path / 'deck.json')
    analysis = http_json('/api/analyze', {'text': 'run', 'lang': 'en'})['morphology']['tokens'][0]
    assert analysis['confidence'] is None
    assert analysis['pos_uncertainty']['status'] == 'development_calibrated'
    card = http_json('/api/add_card', {'front': 'run', 'back': 'reference', 'analysis': analysis})['card']
    assert card['analysis']['pos_uncertainty'] == analysis['pos_uncertainty']
    assert 'pos-uncertain' in card['tags']
