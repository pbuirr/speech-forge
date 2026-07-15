"""
speech_engine/lm.py — Stupid-backoff n-gram language model.

Used to score translation candidates in context: given the words already
output, pick the word that produces the most natural English sequence
rather than just the highest-IDF IBM1 translation.

The classic WSD example: "le monde" → "the world" (not "the everyone").
IBM1 can't tell them apart since "monde" maps to both. But P("world" | "the")
is much higher than P("everyone" | "the") in any English corpus, so the LM
picks correctly.

Algorithm: stupid backoff (Brants et al., 2007 — "Large Language Models
in Machine Translation"). Simpler than Kneser-Ney; nearly identical
accuracy in practice, and used by Google at web scale.

  score(w | c2, c1):
    if trigram (c2,c1,w) seen:   count(c2,c1,w) / count(c2,c1)
    elif bigram (c1,w) seen:     α × count(c1,w) / count(c1)
    elif unigram w seen:         α² × count(w) / N
    else:                        α² / N    (OOV floor)

α = 0.4 (Brants et al. default).

Reference implementations: KenLM (Heafield, 2011) uses modified KN smoothing
for better OOV handling, but stupid backoff is sufficient for 5-10 word
translation glosses where context windows are short.
"""

from __future__ import annotations

import math
from typing import Iterable


class NgramLM:
    """
    Trigram language model with stupid-backoff smoothing.

    Train on the target side of the parallel corpus (English sentences) so
    translate_literal() can pick contextually appropriate words from IBM1's
    top-k candidate pool rather than always defaulting to the highest-IDF word.
    """

    def __init__(self, alpha: float = 0.4, min_count: int = 2) -> None:
        self.alpha = alpha
        self.min_count = min_count          # prune n-grams seen < min_count times
        self._log_alpha  = math.log(alpha)
        self._log_alpha2 = 2 * self._log_alpha

        # Unigram: word → count
        self._uni: dict[str, int] = {}
        self._total: int = 0

        # Bigram: context_word → {next_word: count}
        self._bi: dict[str, dict[str, int]] = {}
        # Bigram denominator: context_word → total count of all bigrams starting with it
        self._bi_ctx: dict[str, int] = {}

        # Trigram: (c2, c1) → {next_word: count}
        self._tri: dict[tuple[str, str], dict[str, int]] = {}
        # Trigram denominator: (c2, c1) → total count
        self._tri_ctx: dict[tuple[str, str], int] = {}

        self._finalized = False

    def train(self, sentences: Iterable[list[str]]) -> None:
        """Add training sentences (already tokenized word lists)."""
        for tokens in sentences:
            if not tokens:
                continue
            padded = ["<s>", "<s>"] + tokens

            for i in range(2, len(padded)):
                w2, w1, w0 = padded[i], padded[i - 1], padded[i - 2]

                # Unigram (exclude <s> padding)
                if w2 != "<s>":
                    self._uni[w2] = self._uni.get(w2, 0) + 1
                    self._total += 1

                # Bigram: w1 → w2
                if w1 not in self._bi:
                    self._bi[w1] = {}
                inner_bi = self._bi[w1]
                inner_bi[w2] = inner_bi.get(w2, 0) + 1

                # Trigram: (w0, w1) → w2
                key = (w0, w1)
                if key not in self._tri:
                    self._tri[key] = {}
                inner_tri = self._tri[key]
                inner_tri[w2] = inner_tri.get(w2, 0) + 1

        self._finalized = False

    def finalize(self) -> None:
        """
        Prune low-count n-grams and precompute context denominators.
        Must be called after all training sentences have been added.
        """
        # Prune and compute bigram denominators
        self._bi_ctx = {}
        for ctx in list(self._bi.keys()):
            inner = {w: c for w, c in self._bi[ctx].items() if c >= self.min_count}
            if inner:
                self._bi[ctx] = inner
                self._bi_ctx[ctx] = sum(inner.values())
            else:
                del self._bi[ctx]

        # Prune and compute trigram denominators
        self._tri_ctx = {}
        for ctx in list(self._tri.keys()):
            inner = {w: c for w, c in self._tri[ctx].items() if c >= self.min_count}
            if inner:
                self._tri[ctx] = inner
                self._tri_ctx[ctx] = sum(inner.values())
            else:
                del self._tri[ctx]

        self._finalized = True

    def score(self, word: str, context: tuple[str, ...] | list[str]) -> float:
        """
        Stupid-backoff log-score: log P(word | last two context words).

        Returns a comparable log-probability. Higher = more likely given context.
        Monotone: a word with bigram evidence always scores higher than pure unigram.
        """
        if not self._finalized:
            self.finalize()

        c1 = context[-1] if context else "<s>"        # immediately preceding word
        c2 = context[-2] if len(context) >= 2 else "<s>"  # two words back

        # Try trigram: P(word | c2, c1)
        tri_inner = self._tri.get((c2, c1))
        if tri_inner:
            tri_cnt = tri_inner.get(word, 0)
            ctx_cnt = self._tri_ctx.get((c2, c1), 0)
            if tri_cnt > 0 and ctx_cnt > 0:
                return math.log(tri_cnt / ctx_cnt)

        # Try bigram backoff: α × P(word | c1)
        bi_inner = self._bi.get(c1)
        if bi_inner:
            bi_cnt = bi_inner.get(word, 0)
            ctx_cnt = self._bi_ctx.get(c1, 0)
            if bi_cnt > 0 and ctx_cnt > 0:
                return self._log_alpha + math.log(bi_cnt / ctx_cnt)

        # Unigram backoff: α² × P(word)
        uni_cnt = self._uni.get(word, 0)
        total = max(self._total, 1)
        if uni_cnt > 0:
            return self._log_alpha2 + math.log(uni_cnt / total)

        # OOV floor
        return self._log_alpha2 + math.log(1 / (total + 1))
