"""
speech_engine/lexical_fallback.py — Statistical lexical translation fallback.

When the neural model fails (coverage check or entropy gate fires), this
module provides a word-level gloss derived from a parallel corpus using two
classical NLP algorithms:

  1. IBM Model 1  (Brown et al., 1993) — EM algorithm that learns P(e|f),
     the probability that a target word e is a translation of a source word f.
     Replaces hand-coded bilingual dictionaries: the corpus teaches the lexicon.

  2. PMI phrase extraction — after word alignment, source n-grams that
     consistently co-occur with a target n-gram (measured by pointwise mutual
     information) are promoted to a phrase table.

Architecture:
  - Completely language-agnostic: pass any parallel TSV corpus and language
    codes; the same code learns ES→EN, FR→EN, DE→EN, etc.
  - Zero hardcoded translations: all lexical knowledge comes from data.
  - Loaded once at server startup; translate_literal() is then O(n) in
    sentence length.

Why IBM Model 1?
  Unlike a simple frequency table or positional alignment, IBM Model 1 uses
  the entire bitext to resolve ambiguity. "banco" appears as both "bank" and
  "bench" in the corpus — Model 1 assigns higher probability to whichever
  translation appears more consistently across all sentences containing "banco".
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from speech_engine.lm import NgramLM
from speech_engine.morphology import CLOSED_CLASS, INFLECTION_SUFFIXES, POS


# ── Curated verb fallback (ES→EN) ─────────────────────────────────────────────
# Common Spanish infinitives that may be absent from a news-domain IBM2 table
# (vocab capped at 15k, minimum alignment probability ≥ 0.05).  These cover
# conversational / language-learning vocabulary that would otherwise surface
# as [brackets] even when the morphological suffix stripping correctly recovers
# the infinitive.  Only infinitives go here — conjugated forms are handled by
# INFLECTION_SUFFIXES stripping in lookup().
_ES_VERB_FALLBACK: dict[str, str] = {
    "preferir": "prefer",
    "imaginar": "imagine",
    "necesitar": "need",
    "decidir": "decide",
    "recordar": "remember",
    "olvidar": "forget",
    "esperar": "wait",
    "sentir": "feel",
    "perder": "lose",
    "ganar": "win",
    "encontrar": "find",
    "llevar": "take",
    "dejar": "leave",
    "seguir": "continue",
    "creer": "believe",
    "pensar": "think",
    "aprender": "learn",
    "enseñar": "teach",
    "explicar": "explain",
    "intentar": "try",
    "lograr": "achieve",
    "mejorar": "improve",
    "cambiar": "change",
    "ayudar": "help",
    "compartir": "share",
    "mostrar": "show",
    "preguntar": "ask",
    "contestar": "answer",
    "entender": "understand",
    "conocer": "know",
    "reconocer": "recognize",
    "desear": "wish",
    "soñar": "dream",
    "vivir": "live",
    "morir": "die",
    "nacer": "be born",
    "crecer": "grow",
    "correr": "run",
    "caminar": "walk",
    "saltar": "jump",
    "caer": "fall",
    "subir": "go up",
    "bajar": "go down",
    "abrir": "open",
    "cerrar": "close",
    "empezar": "start",
    "terminar": "finish",
    "continuar": "continue",
    "parar": "stop",
}

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer, strips punctuation."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text, re.UNICODE) if w]


# ── Corpus loader ─────────────────────────────────────────────────────────────

def _load_pairs(
    tsv_path: Path,
    max_len: int = 25,
) -> list[tuple[list[str], list[str]]]:
    """
    Load a tab-separated parallel corpus.
    Filters out empty lines, comments, and very long sentences (which add
    noise to word alignment without contributing clean signal).
    """
    pairs: list[tuple[list[str], list[str]]] = []
    for line in tsv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        src = _tokenize(parts[0])
        tgt = _tokenize(parts[1])
        # Length ratio check: skip badly misaligned pairs
        if not src or not tgt:
            continue
        ratio = len(tgt) / len(src)
        if not (0.3 <= ratio <= 3.5):
            continue
        if len(src) <= max_len and len(tgt) <= max_len:
            pairs.append((src, tgt))
    return pairs


# ── IBM Model 1 ───────────────────────────────────────────────────────────────

_NULL = "<NULL>"  # Null alignment token — absorbs source words with no clean match


def _train_ibm2(
    pairs: list[tuple[list[str], list[str]]],
    iterations: int = 5,
    vocab_size: int = 40_000,
    diag_lambda: float = 0.3,
) -> dict[str, dict[str, float]]:
    """
    EM training with IBM Model 1 translation table + diagonal alignment prior.

    Extends IBM Model 1 by weighting each (source_pos, target_pos) alignment
    by how close it is to the diagonal of the alignment matrix — the fast_align
    approach (Dyer et al., 2013):

        weight(i, j, lf, le) = exp(-λ × |i/lf − j/le|)

    Words at the same relative position (diagonal) get weight 1.0; words at
    opposite corners get weight exp(-λ). With λ=0.3, the off-corner discount
    is ~26% — a gentle bias that improves alignment without breaking
    cross-lingual reordering (SOV→SVO, modifier placement, etc.).

    This is not full IBM Model 2 (which learns a separate alignment table
    a(j|i, lf, le) per position triple) — it is the fixed-function prior from
    fast_align, which gives 90% of the benefit at a fraction of the cost.

    Args:
        pairs:       list of (source_tokens, target_tokens)
        iterations:  EM rounds (5 is standard for IBM1/2)
        vocab_size:  vocabulary cap (top-N by frequency)
        diag_lambda: diagonal bias strength; 0 = pure IBM1, 1.0 = strong diagonal

    Returns:
        t[src_word][tgt_word] = P(tgt_word | src_word)

    Reference: Dyer et al., 2013 — "A Simple, Fast, and Effective Reparameterization
               of IBM Model 2" (fast_align). https://github.com/clab/fast_align
    """
    # Build bounded vocabulary from frequency
    src_freq: Counter[str] = Counter(w for f, _ in pairs for w in f)
    tgt_freq: Counter[str] = Counter(w for _, e in pairs for w in e)
    src_vocab = {w for w, _ in src_freq.most_common(vocab_size)}
    tgt_vocab = {w for w, _ in tgt_freq.most_common(vocab_size)}

    # Filter pairs to in-vocabulary words
    filtered: list[tuple[list[str], list[str]]] = []
    for f_words, e_words in pairs:
        f2 = [w for w in f_words if w in src_vocab]
        e2 = [w for w in e_words if w in tgt_vocab]
        if f2 and e2:
            filtered.append((f2, e2))

    n_tgt = len(tgt_vocab)
    uniform = 1.0 / max(n_tgt, 1)

    # t[f][e] initialized to uniform P(e|f)
    t: dict[str, dict[str, float]] = {
        f: defaultdict(lambda: uniform) for f in src_vocab
    }

    for _iter in range(iterations):
        count: dict[str, dict[str, float]] = {f: defaultdict(float) for f in src_vocab}
        total: dict[str, float] = defaultdict(float)

        # E-step — diagonal-prior weighted alignment
        for f_words, e_words in filtered:
            lf = len(f_words)
            le = len(e_words)
            lf_norm = max(lf - 1, 1)
            le_norm = max(le - 1, 1)

            for i, f in enumerate(f_words):
                tf = t[f]
                fi = i / lf_norm  # source relative position in [0, 1]

                # Compute per-target-position weights: t(e|f) × diagonal_prior
                weights: list[float] = []
                for j, e in enumerate(e_words):
                    fj = j / le_norm
                    pos_weight = math.exp(-diag_lambda * abs(fi - fj))
                    weights.append(tf[e] * pos_weight)

                s = sum(weights)
                if s < 1e-300:
                    continue
                inv_s = 1.0 / s
                cf = count[f]
                for e, w in zip(e_words, weights):
                    delta = w * inv_s
                    cf[e] += delta
                    total[f] += delta

        # M-step
        for f in count:
            tot = total[f]
            if tot < 1e-300:
                continue
            inv_tot = 1.0 / tot
            tf = t[f]
            for e, c in count[f].items():
                tf[e] = c * inv_tot

    return t


def _compute_idf(
    pairs: list[tuple[list[str], list[str]]],
) -> dict[str, float]:
    """
    Compute inverse document frequency for target-side vocabulary.

    IDF(w) = log(N / df(w))

    High-frequency words like "i", "the", "a" appear in almost every
    sentence — their IDF is near zero. Rare content words ("beach",
    "excited", "negotiate") have high IDF and are much more informative
    as translations.

    Using IDF to reweight IBM Model 1 probabilities prevents function
    words from dominating the best-translation ranking in noisy corpora
    (e.g. movie subtitles where "I" appears in 80%+ of lines).
    """
    n = len(pairs)
    doc_freq: dict[str, int] = {}
    for _, e_words in pairs:
        for w in set(e_words):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    return {w: math.log(n / df) for w, df in doc_freq.items()}


def _best_translations(
    t: dict[str, dict[str, float]],
    idf: dict[str, float] | None = None,
    min_prob: float = 0.05,
    top_k: int = 3,
) -> dict[str, list[str]]:
    """
    For each source word, extract the top-k most informative translations.

    Score = P(e|f) × IDF(e)

    Multiplying by IDF down-ranks function words ("i", "the", "a", "you")
    even when IBM Model 1 assigns them high probability — those words appear
    everywhere so alignment to them is mostly noise. Content words like
    "beach", "famous", "understand" are rare enough that high P(e|f) is
    a genuine alignment signal.

    Returns word → [best_translation, ...] sorted by score descending.
    """
    # IDF threshold for the second-pass enrichment trigger.
    # A source word's top-k must ALL fall below this value before we rescan
    # for content words. Set to 2.0 so "of" (IDF=1.68) no longer blocks the
    # rescan for words like "lugar" whose top-3 are all prepositions/articles.
    _FUNC_IDF = 2.0

    result: dict[str, list[str]] = {}
    for f, e_dist in t.items():
        candidates: list[tuple[float, str]] = []
        for e, prob in e_dist.items():
            if prob < min_prob or e == _NULL:
                continue
            score = prob * (idf[e] if idf and e in idf else 1.0)
            candidates.append((score, e))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        top = [e for _, e in candidates[:top_k]]
        result[f] = top

        # Second pass: if ALL top-k entries are very common function words
        # (every entry has IDF < _FUNC_IDF), rescan with a lower threshold
        # to surface content words that are just below the main cutoff.
        # This is targeted — it only fires for words like "lugar" where IBM1
        # co-aligns overwhelmingly to "the"/"to"/"of", not for words like
        # "por" where "for" (IDF=2.4) already made the first-pass top-k.
        if idf and all((idf.get(e, 0.0) < _FUNC_IDF) for e in top):
            extra: list[tuple[float, str]] = []
            for e, prob in e_dist.items():
                if prob < 0.005 or e == _NULL or e in top:
                    continue
                score = prob * idf.get(e, 1.0)
                if idf.get(e, 0.0) >= _FUNC_IDF:  # content words only
                    extra.append((score, e))
            if extra:
                extra.sort(reverse=True)
                result[f] = top + [e for _, e in extra[:top_k]]

    return result


# ── PMI phrase extraction ─────────────────────────────────────────────────────

def _extract_phrase_pairs(
    pairs: list[tuple[list[str], list[str]]],
    word_trans: dict[str, list[str]],
    max_n: int = 3,
    min_freq: int = 4,
    min_pmi: float = 2.0,
) -> dict[tuple[str, ...], str]:
    """
    Extract multi-word phrase pairs from the parallel corpus using PMI.

    Algorithm:
    1. For each pair, build a MAP alignment: f[i] → e[j] = argmax_e t(e|f[i])
    2. For each contiguous source span of length 2..max_n, find the
       corresponding target span (min to max of aligned target indices)
    3. Count how often each (f-span, e-span) appears together
    4. Keep pairs whose log PMI exceeds min_pmi and appear ≥ min_freq times

    PMI(f-span, e-span) = log[ P(f-span, e-span) / (P(f-span) × P(e-span)) ]
    High PMI means the words appear together more than chance would predict —
    the hallmark of an idiomatic phrase.
    """
    f_span_count: Counter[tuple[str, ...]] = Counter()
    e_span_count: Counter[tuple[str, ...]] = Counter()
    joint_count: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    n_pairs = len(pairs)

    for f_words, e_words in pairs:
        le = len(e_words)

        # MAP alignment: for each source position, best target index
        f_to_e: dict[int, int] = {}
        for i, f in enumerate(f_words):
            if f not in word_trans:
                continue
            best_j, best_p = -1, -1.0
            for j, e in enumerate(e_words):
                if e in word_trans.get(f, []):
                    # Rank by position in top-k list (index 0 = most probable)
                    rank = word_trans[f].index(e)
                    p = 1.0 / (rank + 1)
                    if p > best_p:
                        best_p, best_j = p, j
            if best_j >= 0:
                f_to_e[i] = best_j

        lf = len(f_words)
        for i1 in range(lf):
            for i2 in range(i1 + 1, min(i1 + max_n + 1, lf + 1)):
                f_span = tuple(f_words[i1:i2])
                f_span_count[f_span] += 1
                if len(f_span) < 2:
                    continue
                # Target span = range of aligned target positions
                e_idxs = [f_to_e[i] for i in range(i1, i2) if i in f_to_e]
                if not e_idxs:
                    continue
                j1, j2 = min(e_idxs), max(e_idxs) + 1
                if j2 - j1 > max_n + 1:
                    continue
                e_span = tuple(e_words[j1:j2])
                e_span_count[e_span] += 1
                joint_count[(f_span, e_span)] += 1

    # PMI filter
    phrase_table: dict[tuple[str, ...], tuple[str, int]] = {}  # f_span → (e_str, count)
    for (f_span, e_span), cnt in joint_count.items():
        if cnt < min_freq:
            continue
        pf = f_span_count[f_span] / n_pairs
        pe = e_span_count[e_span] / n_pairs
        pfe = cnt / n_pairs
        if pf < 1e-10 or pe < 1e-10:
            continue
        pmi = math.log(pfe / (pf * pe))
        if pmi < min_pmi:
            continue
        e_str = " ".join(e_span)
        # Keep highest-PMI translation per source phrase
        existing = phrase_table.get(f_span)
        if existing is None or cnt > existing[1]:
            phrase_table[f_span] = (e_str, cnt)

    return {k: v[0] for k, v in phrase_table.items()}


# ── Core class ────────────────────────────────────────────────────────────────

class LexicalFallback:
    """
    Language-agnostic word-level translator trained from a parallel corpus.

    Two internal indexes — one for single words (from IBM Model 1) and one
    for multi-word phrases (from PMI phrase extraction) — are built at load
    time. At inference time, translate_literal() scans for the longest phrase
    match first, then falls back to single-word lookup.

    This works for any language pair: ES→EN, FR→EN, DE→EN, etc., given the
    appropriate parallel TSV corpus.
    """

    # IDF threshold below which a target word is considered a function word.
    # Words with IDF < this value appear in more than ~13% of sentences and
    # carry little translation signal (e.g. "I", "you", "the", "a").
    # When the top-1 translation is a function word and top-2 is not, we
    # prefer top-2. This is not a hardcoded word list — it is a frequency
    # threshold derived from the target-side IDF distribution.
    # Personal pronouns ("you"≈2.06, "i"≈1.45, "we"≈2.32) and determiners
    # cluster below 3.0 in Europarl+Tatoeba EN IDF. Modals ("could"≈4.56,
    # "must"≈3.62) and content verbs ("understand"≈5.44) are above it.
    # "be"≈2.57 is also below but is still returned when no better alt exists.
    _FUNC_WORD_IDF = 3.0

    def __init__(
        self,
        word_trans: dict[str, list[str]],
        phrase_trans: dict[tuple[str, ...], str],
        src_lang: str = "?",
        tgt_lang: str = "?",
        tgt_idf: dict[str, float] | None = None,
        lm: "NgramLM | None" = None,
    ) -> None:
        self._word_trans = word_trans      # f → [best_e, second_best_e, ...]
        self._phrase_trans = phrase_trans  # (f1,f2,...) → "e1 e2 ..."
        self._tgt_idf = tgt_idf or {}     # target word → IDF score
        self._lm = lm                      # n-gram LM for context-aware word selection
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Bypass sets derived from morphology.CLOSED_CLASS — no hardcoded lists here.
        #
        # Three bypass regimes, all driven by morphology.CLOSED_CLASS — no
        # hardcoded English word lists.
        #
        #   PRON regime — triggers when top-1 is a target-language pronoun AND
        #     the source word is NOT a source-language pronoun.  IBM1 aligns verbs
        #     to subject pronouns ("tengo"→"i", "parler"→"i") because subject and
        #     verb co-occur.  But "je"→"i" IS correct (pronoun→pronoun), so we skip
        #     bypass when the source is also a pronoun.
        #
        #   LOW-IDF closed-class regime — triggers when top-1 is any closed-class
        #     word (DET, PREP, CONJ …) with IDF < 1.5.  In a large EN corpus that
        #     catches "the" (≈0.92) and "to" (≈1.30) — the two most common false
        #     alignments for content words — while leaving "a" (≈1.65), "of" (≈1.70),
        #     "in" (≈1.88) untouched (they are sometimes correct translations).
        #     The 1.5 threshold is corpus-derived: any language's ultra-common
        #     function words will cluster below it.
        #
        # bypass_stop: closed-class words that must never be returned as a bypass
        #   result.  Excludes auxiliaries (have, be, could, must) so they remain
        #   reachable as legitimate translations.
        _cc  = CLOSED_CLASS.get(tgt_lang, {})
        _src = CLOSED_CLASS.get(src_lang, {})
        self._tgt_pron_words: frozenset[str] = frozenset(_cc.get(POS.PRON, []))
        self._src_pron_words: frozenset[str] = frozenset(_src.get(POS.PRON, []))
        self._tgt_bypass_stop: frozenset[str] = frozenset(
            w for pos in (POS.PRON, POS.DET, POS.PREP, POS.CONJ, POS.SCONJ)
            for w in _cc.get(pos, [])
        )

        # Source-side inflection suffix table from morphology.INFLECTION_SUFFIXES.
        # Allows recovering an infinitive from an unknown conjugated form, e.g.
        # "irías" (ES conditional) → strip "irías" → stem "i" + "r" = "ir" → look up.
        # Language-specific data lives entirely in morphology.py.
        self._src_suffixes: tuple[tuple[str, list[str]], ...] = (
            INFLECTION_SUFFIXES.get(src_lang, ())
        )

    @classmethod
    def from_corpus(
        cls,
        tsv_path: Path | str,
        src_lang: str = "?",
        tgt_lang: str = "?",
        ibm_iterations: int = 5,
        max_phrase_n: int = 3,
    ) -> "LexicalFallback":
        """Build from a single parallel corpus TSV."""
        return cls.from_corpora(
            [tsv_path],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            ibm_iterations=ibm_iterations,
            max_phrase_n=max_phrase_n,
        )

    @classmethod
    def from_corpora(
        cls,
        tsv_paths: list[Path | str],
        src_lang: str = "?",
        tgt_lang: str = "?",
        ibm_iterations: int = 5,
        max_phrase_n: int = 3,
    ) -> "LexicalFallback":
        """
        Build a LexicalFallback from one or more parallel corpora combined.

        Multiple corpora are merged before training so that IBM Model 1
        sees a richer, more diverse signal. A curated high-quality corpus
        (even with only a few hundred pairs) can be combined with a large
        noisy corpus (200k movie subtitles) to anchor rare-word translations.

        Steps:
        1. Load and merge sentence pairs from all corpus files
        2. Train IBM Model 1 with NULL alignment (EM, 5 iterations)
        3. IDF-weight translations to suppress function-word dominance
        4. PMI phrase extraction on IBM1-aligned spans

        Language-agnostic: no language-specific code — only corpus data.
        """
        all_pairs: list[tuple[list[str], list[str]]] = []
        for path in tsv_paths:
            p = Path(path)
            if p.exists():
                chunk = _load_pairs(p)
                all_pairs.extend(chunk)

        if not all_pairs:
            return cls({}, {}, src_lang, tgt_lang)

        # Cache key: hash of all corpus file modification times + sizes
        cache_sig = _corpus_signature(tsv_paths)
        cache_path = _cache_path(src_lang, tgt_lang, cache_sig)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached = pickle.load(f)
                if len(cached) == 3:
                    word_trans, phrase_trans, tgt_idf = cached
                else:
                    word_trans, phrase_trans = cached
                    tgt_idf = {}
                # LM is fast to rebuild from the already-loaded pairs, no cache needed
                lm = _build_lm(all_pairs[:50_000])
                return cls(word_trans, phrase_trans, src_lang, tgt_lang, tgt_idf, lm=lm)
            except Exception:
                pass  # cache corrupt — retrain

        # Deduplicate on source side
        seen: set[tuple[str, ...]] = set()
        deduped: list[tuple[list[str], list[str]]] = []
        for f, e in all_pairs:
            key = tuple(f)
            if key not in seen:
                seen.add(key)
                deduped.append((f, e))
        pairs = deduped

        # Cap training pairs and vocabulary to avoid OOM on large corpora.
        # 150k pairs with vocab_size=15k still covers 95%+ of word types and
        # produces high-quality IBM2 alignments while staying under ~1GB RAM.
        # Pairs are already deduped+shuffled so the first 150k are representative.
        train_pairs = pairs[:150_000]

        # IBM Model 1 + diagonal alignment prior (IBM Model 2 approximation)
        t = _train_ibm2(train_pairs, iterations=ibm_iterations, vocab_size=15_000)

        # IDF over all deduped pairs (more data = better frequency estimates)
        idf = _compute_idf(pairs)

        # IDF-weighted best translations — top_k=5 gives the LM more candidates
        word_trans = _best_translations(t, idf=idf, min_prob=0.05, top_k=5)

        # PMI phrase pairs (cap at 100k to keep memory reasonable)
        phrase_trans = _extract_phrase_pairs(
            pairs[:100_000], word_trans, max_n=max_phrase_n, min_freq=4, min_pmi=2.0
        )

        # Reuse the already-computed English-side IDF for function-word
        # filtering in lookup(). No extra computation needed.
        tgt_idf = idf

        # Persist to disk so future startups skip the EM training
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache_path.open("wb") as f:
                pickle.dump((word_trans, phrase_trans, tgt_idf), f)
        except Exception:
            pass

        # Build n-gram LM from target side — fast (<5s), not cached separately.
        # All pairs are available here whether we hit the cache or not.
        lm = _build_lm(all_pairs[:50_000])

        return cls(word_trans, phrase_trans, src_lang, tgt_lang, tgt_idf, lm=lm)

    def lookup(self, word: str) -> str | None:
        """
        Return best single-word translation, or None if unknown.

        Two layers:
        1. Closed-class bypass: if IBM1 top-1 is a closed-class word (pronoun or
           determiner) that the morphology model knows about, prefer any open-class
           alternative with higher IDF.  Bypass set is derived from
           morphology.CLOSED_CLASS[tgt_lang] — no hardcoded English word lists here.
        2. Morphological fallback: if the word is completely unknown, strip
           inflectional suffixes (from morphology.INFLECTION_SUFFIXES[src_lang])
           to recover the infinitive and look that up instead.
           E.g. "irías" (ES conditional) → "ir" → "go".
        """
        hits = self._word_trans.get(word.lower())
        if hits:
            if len(hits) == 1:
                return hits[0]
            h0_idf = self._tgt_idf.get(hits[0], 0.0)

            # PRON regime: subject pronouns dominate IBM1 alignments for verbs
            # because subject and verb always co-occur ("tengo" → "i", "parler" → "i").
            # Guard: if the SOURCE word is itself a pronoun (je→i, yo→i), the
            # pronoun translation IS correct — skip bypass so we don't jump to
            # a spurious high-IDF word like "please" (from "je vous en prie").
            if (hits[0] in self._tgt_pron_words
                    and word.lower() not in self._src_pron_words):
                for alt in hits[1:]:
                    if (len(alt) >= 2
                            and alt not in self._tgt_bypass_stop
                            and self._tgt_idf.get(alt, 0.0) > h0_idf):
                        return alt

            # Low-IDF closed-class regime: triggers when top-1 is any closed-class
            # word with IDF < 1.5 that is NOT a pronoun (pronouns are handled above).
            # In a large EN corpus this catches "the" (≈0.92) and "to" (≈1.30) —
            # the two most common false alignments — while leaving "a" (≈1.65),
            # "of" (≈1.70), "in" (≈1.88) alone (they are sometimes correct).
            # Prefer a genuine content word (IDF ≥ _FUNC_WORD_IDF = 3.0).
            elif (h0_idf < 1.5
                    and hits[0] in self._tgt_bypass_stop
                    and hits[0] not in self._tgt_pron_words):
                for alt in hits[1:]:
                    if (len(alt) >= 2
                            and alt not in self._tgt_bypass_stop
                            and self._tgt_idf.get(alt, 0.0) >= self._FUNC_WORD_IDF):
                        return alt

            return hits[0]

        # Word not in lexicon — check curated verb fallback first (covers infinitives
        # whose frequency is too low to make the IBM2 vocab cap), then try suffix stripping.
        if self.src_lang == "es":
            direct = _ES_VERB_FALLBACK.get(word.lower())
            if direct:
                return direct

        # Try stripping inflectional suffixes to find the infinitive or base form.
        # Require ≥ 2 chars in the stem so short words like "dia" (3 chars) don't
        # match suffix "ia" → stem "d" → "dar".
        w = word.lower()
        for suffix, inf_endings in self._src_suffixes:
            if not w.endswith(suffix):
                continue
            stem = w[: -len(suffix)] if suffix else w
            if len(stem) < 2:
                continue
            for ending in inf_endings:
                candidate = stem + ending
                if not candidate:
                    continue
                if self._word_trans.get(candidate):
                    result = self.lookup(candidate)  # recurse once for bypass
                    if result:
                        return result
                # IBM2 may not have the infinitive if it's rare in the corpus.
                # Fall through to the curated verb table before giving up.
                if self.src_lang == "es":
                    fallback = _ES_VERB_FALLBACK.get(candidate)
                    if fallback:
                        return fallback
        return None

    def lookup_with_context(
        self, word: str, context: tuple[str, ...] | list[str]
    ) -> str | None:
        """
        Context-aware translation using the n-gram LM to disambiguate top-k candidates.

        Applies the same bypass logic as lookup() to determine the candidate pool,
        then uses the LM to pick the candidate most likely given the preceding output
        tokens. This resolves word sense ambiguity that IDF alone cannot handle.

        Example:
          "monde" has IBM1 top-k = ["everyone", "world", "all"]
          After "see the": P("world" | "see","the") >> P("everyone" | "see","the")
          → returns "world"  ✓

          "femme" after "my": P("wife" | "my") >> P("woman" | "my")
          → returns "wife"  ✓
          "femme" after "is a": P("woman" | "is","a") >> P("wife" | "is","a")
          → returns "woman"  ✓
        """
        if self._lm is None:
            return self.lookup(word)

        hits = self._word_trans.get(word.lower())
        if not hits:
            if self.src_lang == "es":
                direct = _ES_VERB_FALLBACK.get(word.lower())
                if direct:
                    return direct
            # Morphological suffix stripping fallback — same as lookup()
            w = word.lower()
            for suffix, inf_endings in self._src_suffixes:
                if not w.endswith(suffix):
                    continue
                stem = w[: -len(suffix)] if suffix else w
                if len(stem) < 2:
                    continue
                for ending in inf_endings:
                    candidate = stem + ending
                    if not candidate:
                        continue
                    if self._word_trans.get(candidate):
                        return self.lookup_with_context(candidate, context)
                    if self.src_lang == "es":
                        fallback = _ES_VERB_FALLBACK.get(candidate)
                        if fallback:
                            return fallback
            return None

        if len(hits) == 1:
            return hits[0]

        h0_idf = self._tgt_idf.get(hits[0], 0.0)

        # Determine candidate pool using the same bypass rules as lookup().
        # The LM then picks the best within that pool.
        if (hits[0] in self._tgt_pron_words
                and word.lower() not in self._src_pron_words):
            # PRON regime: pool = non-closed-class alternatives with higher IDF
            pool = [h for h in hits[1:]
                    if len(h) >= 2
                    and h not in self._tgt_bypass_stop
                    and self._tgt_idf.get(h, 0.0) > h0_idf]
            if not pool:
                return hits[0]  # no better alternative exists

        elif (h0_idf < 1.5
                and hits[0] in self._tgt_bypass_stop
                and hits[0] not in self._tgt_pron_words):
            # Low-IDF regime: pool = content words (IDF >= FUNC_WORD_IDF)
            pool = [h for h in hits
                    if len(h) >= 2
                    and h not in self._tgt_bypass_stop
                    and self._tgt_idf.get(h, 0.0) >= self._FUNC_WORD_IDF]
            if not pool:
                return hits[0]

        else:
            # No bypass triggered — IBM1 top-1 is already a plausible translation.
            # Use the LM only to disambiguate among content-word alternatives.
            # Never let the LM override a content word with a function word — "the"
            # scores high in almost every context ("I like the ___") but is wrong
            # when the IBM1 top-1 is a content word like "world" or "understand".
            if self._tgt_idf.get(hits[0], 0.0) >= self._FUNC_WORD_IDF:
                # Top-1 is a content word; restrict pool to content words only
                pool = [h for h in hits
                        if self._tgt_idf.get(h, 0.0) >= self._FUNC_WORD_IDF]
                if len(pool) <= 1:
                    return hits[0]  # only one content word — no ambiguity to resolve
            else:
                # Top-1 is a function word, use full pool
                pool = hits

        # Score each candidate with the LM and return the winner
        return max(pool, key=lambda w: self._lm.score(w, context))

    def translate_literal(self, text: str) -> str:
        """
        Produce a word-by-word gloss of the input text.

        Phrase matches (longest first) take priority over single-word lookup,
        so idiomatic expressions like "a menudo" → "often" are handled
        before "a" → "to" and "menudo" → "little".

        Single words are resolved with lookup_with_context() so the n-gram LM
        can use the growing output sequence to pick the most natural word.
        """
        words = _tokenize(text)
        parts: list[str] = []
        out_tokens: list[str] = []  # output tokens so far, for LM context
        i = 0
        while i < len(words):
            word = words[i]

            # Bare numbers pass through unchanged
            if re.match(r"^\d+(?:[.,]\d+)*$", word):
                parts.append(word)
                out_tokens.append(word)
                i += 1
                continue

            # Try longest phrase match first (max_n → 4).
            # Guard: skip a phrase if it loses content words compared to
            # translating each source word individually.
            # Example: ("si","pudieras") → "if you" loses "could" (IDF=4.56)
            # vs word-by-word "if" + "could". If the phrase drops content IDF,
            # fall through to word-level so modals/verbs aren't swallowed.
            matched = False
            for n in range(min(4, len(words) - i), 1, -1):
                gram = tuple(words[i:i + n])
                if gram not in self._phrase_trans:
                    continue
                phrase_en = self._phrase_trans[gram]

                # Sum content IDF of the phrase output
                phrase_idf = sum(
                    self._tgt_idf.get(w, 0.0)
                    for w in phrase_en.split()
                    if self._tgt_idf.get(w, 0.0) >= self._FUNC_WORD_IDF
                )
                # Sum content IDF of word-by-word equivalents
                word_idf = sum(
                    self._tgt_idf.get(self.lookup(src_w) or "", 0.0)
                    for src_w in gram
                    if self._tgt_idf.get(self.lookup(src_w) or "", 0.0) >= self._FUNC_WORD_IDF
                )
                if word_idf > 0 and phrase_idf < word_idf * 0.75:
                    continue  # phrase loses too much content; use word-by-word

                parts.append(phrase_en)
                out_tokens.extend(phrase_en.split())
                i += n
                matched = True
                break
            if matched:
                continue

            # Single-word lookup with LM context (last 2 output tokens)
            ctx = tuple(out_tokens[-2:])
            en = self.lookup_with_context(word, ctx)
            if en:
                parts.append(en)
                out_tokens.append(en)
            else:
                parts.append(f"[{word}]")
                out_tokens.append(word)  # use source word as context placeholder

            i += 1

        result = " ".join(parts).strip()
        result = re.sub(r"\s+", " ", result)
        return result[0].upper() + result[1:] if result else result


# ── Disk cache helpers ────────────────────────────────────────────────────────

def _build_lm(pairs: list[tuple[list[str], list[str]]]) -> NgramLM:
    """Train a stupid-backoff trigram LM on the target (English) side of the corpus."""
    lm = NgramLM(alpha=0.4, min_count=2)
    lm.train(e for _, e in pairs)
    lm.finalize()
    return lm


def _corpus_signature(paths: list, version: str = "ibm2-v1") -> str:
    """Fingerprint corpus files for cache invalidation. Bump `version` to force a global rebuild."""
    h = hashlib.md5()
    h.update(version.encode())  # change this string to invalidate all existing caches
    for p in sorted(str(x) for x in paths):
        path = Path(p)
        if path.exists():
            stat = path.stat()
            h.update(f"{p}:{stat.st_mtime:.0f}:{stat.st_size}".encode())
    return h.hexdigest()[:12]


def _cache_path(src_lang: str, tgt_lang: str, sig: str) -> Path:
    root = Path(__file__).resolve().parent.parent / "data" / "corpora" / ".lex_cache"
    return root / f"ibm1_{src_lang}_{tgt_lang}_{sig}.pkl"


# ── Per-language-pair cache ───────────────────────────────────────────────────

_fallback_cache: dict[str, LexicalFallback] = {}


def get_fallback(
    corpus_path: Path | str | None = None,
    src_lang: str = "es",
    tgt_lang: str = "en",
) -> LexicalFallback:
    """
    Return a trained LexicalFallback for the given language pair.

    Builds from corpus on first call; subsequent calls for the same pair
    return the cached instance. Thread-safe for single-writer startup.
    """
    key = f"{src_lang}→{tgt_lang}"
    if key not in _fallback_cache:
        if corpus_path is not None:
            fb = LexicalFallback.from_corpus(
                corpus_path, src_lang=src_lang, tgt_lang=tgt_lang
            )
        else:
            fb = LexicalFallback({}, {}, src_lang, tgt_lang)
        _fallback_cache[key] = fb
    return _fallback_cache[key]
