"""
speech_engine/bleu.py — BLEU score, implemented from scratch.

BLEU (Bilingual Evaluation Understudy, Papineni et al. 2002) is the
standard automatic metric for machine translation quality. It measures
how much a machine translation overlaps with one or more human reference
translations, using modified n-gram precision with a brevity penalty.

Why implement it ourselves?
    This project builds everything from scratch. The BLEU formula is
    well-defined (20 lines of math) and understanding it tells you how
    to interpret the number. sacrebleu is also easy to install and
    preferred for paper-comparable scores — see the note at the bottom.

Formula:
    BLEU = BP × exp( Σ_n  w_n × log p_n )

    p_n  = modified n-gram precision for order n
         = Σ_sentences  clipped_matches(n) / candidate_ngrams(n)

    BP   = brevity penalty
         = 1.0                if len(cand) > len(ref)
         = exp(1 - r/c)       otherwise  (c = cand tokens, r = ref tokens)

    w_n  = 1/N for n = 1..N  (uniform, N=4 is standard)

Modified precision "clips" each hypothesis n-gram count by the maximum
number of times it appears in the reference — this prevents the model
gaming BLEU by repeating a common word many times.

Reference:
    Papineni, Roukos, Ward, Zhu (2002). "BLEU: a Method for Automatic
    Evaluation of Machine Translation." ACL 2002.

Note on sacreBLEU:
    For results that can be compared across papers, install sacrebleu:
        pip install sacrebleu
    and use:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    SacreBLEU standardizes tokenization so scores from different systems
    are directly comparable. Our implementation is consistent with the
    original paper but may differ slightly from sacreBLEU in edge cases.
"""

from __future__ import annotations

import math
from collections import Counter


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace. Simple but consistent."""
    return text.lower().split()


# ── N-gram helpers ────────────────────────────────────────────────────────────

def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Count all n-grams in a token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _clipped_matches(hyp: list[str], ref: list[str], n: int) -> int:
    """
    Clipped n-gram matches between hypothesis and reference.

    For each n-gram in hyp, its count is clipped to the number of times
    it appears in ref. This prevents repetition gaming.
    """
    hyp_counts = _ngram_counts(hyp, n)
    ref_counts = _ngram_counts(ref, n)
    return sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())


# ── Sentence-level BLEU ───────────────────────────────────────────────────────

def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU with +1 smoothing (Lin & Och 2004).

    Smoothing prevents log(0) for short sentences that have no n-gram
    matches beyond unigrams. Added to both numerator and denominator.

    Returns a float in [0.0, 1.0].
    """
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)

    if not hyp:
        return 0.0

    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        if len(hyp) < n:
            # Sentence shorter than n — treat as 0 matches / 1 candidate (smoothed)
            log_precision_sum += math.log(1 / (len(hyp) + 1))
            continue
        matches = _clipped_matches(hyp, ref, n) + 1  # +1 smoothing
        total   = len(hyp) - n + 1 + 1               # +1 smoothing
        log_precision_sum += math.log(matches / total)

    # Brevity penalty
    c, r = len(hyp), len(ref)
    bp = 1.0 if c >= r else math.exp(1.0 - r / max(c, 1))

    return bp * math.exp(log_precision_sum / max_n)


# ── Corpus-level BLEU ─────────────────────────────────────────────────────────

def corpus_bleu(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU score.

    Aggregates n-gram statistics across all sentence pairs before
    computing precision — this is the correct way to compute BLEU
    (per-sentence then average gives different and less stable results).

    Args:
        hypotheses: list of model output strings (one per sentence)
        references: list of reference strings (one per sentence)
        max_n:      highest n-gram order to include (default 4)

    Returns:
        BLEU score as a float in [0.0, 100.0]  (×100 so "BLEU 25" means 0.25)
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"hypotheses ({len(hypotheses)}) and references ({len(references)}) must match"
        )
    if not hypotheses:
        return 0.0

    # Corpus-level counts
    clipped_totals = [0] * (max_n + 1)   # clipped_totals[n] = Σ clipped matches
    candidate_totals = [0] * (max_n + 1) # candidate_totals[n] = Σ candidate ngrams
    total_hyp_len = 0
    total_ref_len = 0

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        ref = _tokenize(ref_str)

        total_hyp_len += len(hyp)
        total_ref_len += len(ref)

        for n in range(1, max_n + 1):
            if len(hyp) < n:
                continue
            clipped_totals[n]   += _clipped_matches(hyp, ref, n)
            candidate_totals[n] += len(hyp) - n + 1

    # Check for any zero precision — BLEU is 0 if any n-gram precision is 0
    for n in range(1, max_n + 1):
        if candidate_totals[n] == 0 or clipped_totals[n] == 0:
            return 0.0

    # Geometric mean of precisions
    log_avg = sum(
        math.log(clipped_totals[n] / candidate_totals[n])
        for n in range(1, max_n + 1)
    ) / max_n

    # Brevity penalty
    c, r = total_hyp_len, total_ref_len
    bp = 1.0 if c >= r else math.exp(1.0 - r / max(c, 1))

    return 100.0 * bp * math.exp(log_avg)


# ── Detailed result ───────────────────────────────────────────────────────────

def bleu_detail(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 4,
) -> dict:
    """
    Corpus BLEU with per-order breakdown.

    Returns a dict with keys:
        "bleu":        float — overall BLEU × 100
        "bp":          float — brevity penalty
        "precisions":  list[float] — p1, p2, p3, p4 × 100
        "hyp_len":     int — total hypothesis tokens
        "ref_len":     int — total reference tokens
        "n_sentences": int — number of sentence pairs
    """
    if not hypotheses or len(hypotheses) != len(references):
        return {"bleu": 0.0, "bp": 0.0, "precisions": [0.0]*max_n,
                "hyp_len": 0, "ref_len": 0, "n_sentences": 0}

    clipped_totals   = [0] * (max_n + 1)
    candidate_totals = [0] * (max_n + 1)
    total_hyp_len = 0
    total_ref_len = 0

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        ref = _tokenize(ref_str)
        total_hyp_len += len(hyp)
        total_ref_len += len(ref)
        for n in range(1, max_n + 1):
            if len(hyp) >= n:
                clipped_totals[n]   += _clipped_matches(hyp, ref, n)
                candidate_totals[n] += len(hyp) - n + 1

    precisions = []
    log_avg = 0.0
    any_zero = False
    for n in range(1, max_n + 1):
        if candidate_totals[n] == 0 or clipped_totals[n] == 0:
            precisions.append(0.0)
            any_zero = True
        else:
            p = clipped_totals[n] / candidate_totals[n]
            precisions.append(100.0 * p)
            log_avg += math.log(p)

    log_avg /= max_n
    c, r = total_hyp_len, total_ref_len
    bp = 1.0 if c >= r else math.exp(1.0 - r / max(c, 1))
    bleu = 0.0 if any_zero else 100.0 * bp * math.exp(log_avg)

    return {
        "bleu":        round(bleu, 4),
        "bp":          round(bp, 4),
        "precisions":  [round(p, 4) for p in precisions],
        "hyp_len":     total_hyp_len,
        "ref_len":     total_ref_len,
        "n_sentences": len(hypotheses),
    }
