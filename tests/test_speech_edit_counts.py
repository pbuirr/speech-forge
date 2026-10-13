import itertools
from speech_engine.evaluate import edit_counts, levenshtein


def test_edit_counts_match_distance_and_preserve_reference_and_hypothesis_lengths():
    sequences=[list(s) for n in range(4) for s in itertools.product('aβ',repeat=n)]
    for ref in sequences:
        for hyp in sequences:
            counts=edit_counts(ref,hyp)
            assert sum(counts[k] for k in ['substitutions','deletions','insertions'])==levenshtein(ref,hyp)
            assert counts['matches']+counts['substitutions']+counts['deletions']==len(ref)
            assert counts['matches']+counts['substitutions']+counts['insertions']==len(hyp)
