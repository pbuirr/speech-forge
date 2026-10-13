import math
import itertools
from collections import defaultdict

import numpy as np
import pytest

from speech_engine.decode import prefix_beam_search
from speech_engine.lm import NgramLM
from speech_engine.tokenizer import CharTokenizer


@pytest.mark.parametrize('order', [3, 5])
def test_weighted_beam_matches_exhaustive_ctc_paths(order):
    tokenizer = CharTokenizer().fit(['aba', 'bab'])
    model = NgramLM(min_count=1, order=order)
    model.train([list('aba')] * 3 + [list('bab')])
    model.finalize()
    ids = [tokenizer.blank_id, tokenizer.encode('a')[0], tokenizer.encode('b')[0]]
    probabilities = np.array([[.2, .5, .3], [.4, .2, .4], [.3, .6, .1], [.2, .3, .5]])
    logits = np.full((4, tokenizer.vocab_size), -np.inf)
    logits[:, ids] = np.log(probabilities)
    totals = defaultdict(float)
    for path in itertools.product(range(3), repeat=4):
        collapsed = []
        previous = None
        for symbol in path:
            if symbol != previous and symbol != 0:
                collapsed.append('ab'[symbol - 1])
            previous = symbol
        totals[''.join(collapsed)] += math.prod(probabilities[t, symbol] for t, symbol in enumerate(path))
    def score(text):
        context = ['<s>'] * model.context_size
        bonus = 0.0
        for char in text:
            bonus += model.score(char, context[-model.context_size:])
            context.append(char)
        return math.log(totals[text]) + .2 * bonus + .1 * len(text)
    expected = max(totals, key=score)
    assert prefix_beam_search(logits, tokenizer, beam_width=128, lm=model, lm_weight=.2, token_reward=.1) == expected


def test_longer_context_changes_ambiguous_ctc_spelling_without_dictionary_constraints():
    sentences = [list('abcdz')] * 5 + [list('xycdf')] * 9
    tokenizer = CharTokenizer().fit(['abcdz', 'xycdf'])
    models = [NgramLM(min_count=1, order=order) for order in (3, 5)]
    for model in models:
        model.train(sentences)
        model.finalize()
    logits = np.full((5, tokenizer.vocab_size), -np.inf)
    for frame, char in enumerate('abcd'):
        logits[frame, tokenizer.encode(char)[0]] = 0
    logits[-1, tokenizer.encode('z')[0]] = math.log(.46)
    logits[-1, tokenizer.encode('f')[0]] = math.log(.54)
    decoded = [prefix_beam_search(logits, tokenizer, lm=model, lm_weight=.2) for model in models]
    assert decoded == ['abcdf', 'abcdz']
    assert math.isfinite(models[1].score('q', list('unseen')))


def test_default_trigram_scores_are_unchanged_and_unicode_context_is_supported():
    model = NgramLM(min_count=1)
    model.train([list('iyi değil'), list('iyi değil')])
    assert model.context_size == 2
    for token in 'iyi değilx':
        for context in [[], list('de'), list('öğrenci')]:
            assert model.score(token, context) == model._score_trigram(token, context)
    extended = NgramLM(min_count=2, order=5)
    extended.train([list('iyi değil'), list('iyi değil')])
    assert extended.score('ğ', list('de')) > extended.score('x', list('de'))
    with pytest.raises(ValueError):
        NgramLM(order=8)
