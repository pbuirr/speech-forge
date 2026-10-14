"""Stupid-backoff n-gram language model (Brants et al., 2007).

    score(w | context) = count(context, w) / count(context)    if seen
                       = α · score(w | shorter context)         otherwise

with α = 0.4. Scores are unnormalised and only used for ranking. Works on any
token sequence: words for translation reranking, characters for CTC shallow
fusion. Order 3 by default, up to 5.
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

    def __init__(self, alpha: float = 0.4, min_count: int = 2, order: int = 3) -> None:
        if order not in (3, 4, 5):
            raise ValueError("Backoff order must be three, four or five")
        self.order = order
        self.context_size = order - 1
        self._higher = {n: {} for n in range(4, order + 1)}
        self._higher_ctx = {n: {} for n in range(4, order + 1)}
        self.alpha = alpha
        self.min_count = min_count  # prune n-grams seen < min_count times
        self._log_alpha = math.log(alpha)
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
            extended = ["<s>"] * self.context_size + tokens
            for n, counts in self._higher.items():
                for i in range(self.context_size, len(extended)):
                    context = tuple(extended[i - n + 1 : i])
                    inner = counts.setdefault(context, {})
                    token = extended[i]
                    inner[token] = inner.get(token, 0) + 1

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

        for n, counts in self._higher.items():
            denominators = {}
            for context in list(counts):
                inner = {w: c for w, c in counts[context].items() if c >= self.min_count}
                if inner:
                    counts[context] = inner
                    denominators[context] = sum(inner.values())
                else:
                    del counts[context]
            self._higher_ctx[n] = denominators
        self._finalized = True

    def score(self, word: str, context: tuple[str, ...] | list[str]) -> float:
        """Unnormalized backoff ranking score, never a calibrated probability."""
        if not self._finalized:
            self.finalize()
        offset = 0.0
        padded = ["<s>"] * max(0, self.context_size - len(context)) + list(context)
        for n in range(self.order, 3, -1):
            key = tuple(padded[-(n - 1) :])
            count = self._higher[n].get(key, {}).get(word, 0)
            if count:
                return offset + math.log(count / self._higher_ctx[n][key])
            offset += self._log_alpha
        return offset + self._score_trigram(word, context)

    def _score_trigram(self, word: str, context: tuple[str, ...] | list[str]) -> float:
        """
        Stupid-backoff log-score: log P(word | last two context words).

        Returns an unnormalized log-score for candidate ranking.
        """
        if not self._finalized:
            self.finalize()

        c1 = context[-1] if context else "<s>"  # immediately preceding word
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
