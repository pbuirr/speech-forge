"""Locally learned spelling likelihoods; suffixes are not morpheme claims."""
import math
from collections import Counter, defaultdict
from speech_engine.linguistic_resources import normalized_form


class UnknownWordModel:
    """Smoothed case-variant evidence, then rare-form suffix/case backoff.

    Each spelling contributes one unit, shared across its observed tags. This
    prevents frequent words from dominating the distribution of unseen types.
    Posterior/prior ratios supply emission evidence to the contextual HMM.
    Case variants predict POS only: they do not establish lexical identity,
    lemmas, meanings, or named entities, and never exclude competing tags.
    """
    def __init__(self, emissions, maximum_suffix=5, smoothing=5.0, language=None):
        self.maximum_suffix = maximum_suffix
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)
        self.prior = Counter()
        self.language = language
        self.case_variants = defaultdict(Counter)
        for form, tags in emissions.items():
            self.case_variants[self.case_key(form)].update(tags)
            total = sum(tags.values())
            if total > 10 or not any(c.isalpha() for c in form):
                continue
            shape = self.shape(form)
            folded = self.case_key(form)
            weights = {tag: count / total for tag, count in tags.items()}
            self.prior.update(weights)
            self.counts[(shape, '')].update(weights)
            for size in range(1, min(maximum_suffix, len(folded)) + 1):
                self.counts[(shape, folded[-size:])].update(weights)

    @staticmethod
    def shape(form):
        return 'upper' if form.isupper() else 'title' if form.istitle() else 'lower' if form.islower() else 'mixed'

    def case_key(self, form):
        form = normalized_form(form)
        if self.language in {'tr', 'az'}:
            form = form.replace('I', 'ı').replace('İ', 'i')
        return form.lower()

    def scores(self, form, tags):
        total = sum(self.prior.values()) + len(tags)
        prior = {tag: (self.prior[tag] + 1) / total for tag in tags}
        posterior = dict(prior)
        folded = self.case_key(form)
        variants = self.case_variants.get(folded)
        if variants:
            denominator = sum(variants.values()) + self.smoothing
            return {tag: math.log(((variants[tag] + self.smoothing * prior[tag]) / denominator) / prior[tag]) for tag in tags}
        shape = self.shape(form)
        for size in range(0, min(self.maximum_suffix, len(folded)) + 1):
            counts = self.counts.get((shape, folded[-size:] if size else ''))
            if counts:
                denominator = sum(counts.values()) + self.smoothing
                posterior = {tag: (counts[tag] + self.smoothing * posterior[tag]) / denominator for tag in tags}
        return {tag: math.log(posterior[tag] / prior[tag]) for tag in tags}
