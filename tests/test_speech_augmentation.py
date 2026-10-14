import random
import torch
import pytest
from speech_engine.augment import mask_features


def test_masks_preserve_padding_cache_and_lengths():
    mel = torch.arange(2 * 80 * 100, dtype=torch.float32).reshape(2, 80, 100)
    original = mel.clone()
    lengths = torch.tensor([100, 37])
    masked = mask_features(mel, lengths, random.Random(19))
    assert torch.equal(mel, original)
    assert torch.equal(masked[1, :, 37:], original[1, :, 37:])
    assert torch.equal(masked, mask_features(mel, lengths, random.Random(19)))
    assert not torch.equal(masked, original)
    assert torch.isfinite(masked).all()
    assert torch.equal(mask_features(mel, lengths, random.Random(1), 0, 0), mel)
    with pytest.raises(ValueError):
        mask_features(mel, torch.tensor([101, 37]), random.Random(1))


def test_error_breakdown_does_not_invent_semantic_recall():
    from scripts.speech.analyze_speech_errors import summarize

    rows = [{"reference": "john did not buy two", "hypothesis": "jon did buy three"}]
    result = summarize(rows, english=True)
    assert result["word_edits"]["deletions"] == 1
    assert result["critical_lexical_forms"]["negation_forms"]["deletions"] == 1
    assert result["critical_lexical_forms"]["number_forms"]["substitutions"] == 1
    assert result["critical_lexical_forms"]["number_forms"]["unaligned_hypothesis_occurrences"] == 1
    assert result["critical_lexical_forms"]["name_spelling_probes"]["substitutions"] == 1
    assert summarize(rows)["critical_lexical_forms"] == {}
    unicode_rows = [{"reference": "東京 です", "hypothesis": "東京です"}]
    assert summarize(unicode_rows)["space_alignment"]["deletions"] == 1


def test_ctc_prefix_search_matches_exhaustive_path_sum():
    # Independent enumeration catches repeated-label/blank path accounting bugs.
    import itertools
    import numpy as np
    from collections import defaultdict
    from speech_engine.tokenizer import CharTokenizer
    from speech_engine.decode import prefix_beam_search

    tok = CharTokenizer()
    ids = [tok.blank_id, tok._char_to_id["a"], tok._char_to_id[" "]]
    probabilities = np.random.default_rng(93).dirichlet([2, 2, 2], size=4)
    logits = np.full((4, tok.vocab_size), -np.inf)
    logits[:, ids] = np.log(probabilities)
    totals = defaultdict(float)
    for path in itertools.product(range(3), repeat=4):
        collapsed = []
        previous = None
        for index in path:
            token = ids[index]
            if token != previous and token != tok.blank_id:
                collapsed.append(token)
            previous = token
        totals[tok.decode(collapsed)] += float(
            np.prod([probabilities[t, i] for t, i in enumerate(path)])
        )
    assert prefix_beam_search(logits, tok, beam_width=128) == max(totals, key=totals.get)
