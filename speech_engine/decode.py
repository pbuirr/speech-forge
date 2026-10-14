"""CTC decoding: greedy and prefix beam search.

Greedy decoding takes the best token per frame, then removes repeats and
blanks. Prefix beam search keeps the k best prefixes in log space and can add
a character n-gram language model score (shallow fusion) with a per-token
reward.
"""

from __future__ import annotations

import math

import torch
import numpy as np


# Greedy CTC decode


def greedy_decode(
    log_probs: torch.Tensor,
    tokenizer,
) -> list[str]:
    """Greedy CTC decoding: argmax per frame, then collapse.

    Args:
        log_probs: (T, B, vocab_size) model output, time-first
        tokenizer: CharTokenizer

    Returns:
        one decoded string per batch element
    """
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(1)  # (T, 1, V) — single example

    # argmax over vocab at each time step: best token per frame
    pred_ids = log_probs.argmax(dim=-1)  # (T, B)

    results = []
    B = pred_ids.shape[1]
    for b in range(B):
        ids = pred_ids[:, b].tolist()
        text = tokenizer.ctc_collapse(ids)
        results.append(text)

    return results


# CTC prefix beam search


def _logsumexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def prefix_beam_search(
    log_probs_1d: np.ndarray,
    tokenizer,
    beam_width: int = 10,
    lm=None,
    lm_weight: float = 0.0,
    token_reward: float = 0.0,
) -> str:
    """CTC prefix beam search for a single sequence.

    Args:
        log_probs_1d: (T, vocab_size) log-probabilities for one example
        tokenizer:    CharTokenizer
        beam_width:   number of active hypotheses to maintain

    Returns:
        str — the most probable decoded text

    Algorithm (Graves et al. 2012, "Sequence Transduction with RNNs"):
    ─────────────────────────────────────────────────────────────────
    We track, for each active prefix l:
        log_Pb(l)  — log-probability of all CTC paths that produce l
                     and end with a BLANK at the current time step
        log_Pn(l)  — log-probability of all CTC paths that produce l
                     and end with a NON-BLANK at the current time step

    At each time step t, for each beam (l, last_c, log_Pb, log_Pn):

    1. Output BLANK:
        → prefix unchanged, prob adds to log_Pb
        log_Pb(l)  += log(Pb + Pn) + log_p(blank, t)

    2. Output c == last_c  (same char as last non-blank in l):
        → "holding" the current char (not extending prefix):
        log_Pn(l)  += log_Pn + log_p(c, t)
        → "new occurrence" after blank (extending prefix with repeated char):
        log_Pn(l+c) += log_Pb + log_p(c, t)

    3. Output c != last_c  (different char):
        → always extends prefix:
        log_Pn(l+c) += log(Pb + Pn) + log_p(c, t)

    After each time step, prune beam to top beam_width by log(Pb + Pn).
    """
    if beam_width < 1 or lm_weight < 0:
        raise ValueError("Beam width must be positive and LM weight nonnegative")
    # Optional training-only character n-gram shallow fusion. Backoff scores
    # are ranking scores, not calibrated probabilities or verified corrections.
    lm_scores = {(): 0.0}

    def rank(key, probabilities):
        prefix = key[0]
        if lm is not None and lm_weight:
            if prefix not in lm_scores:
                parent = prefix[:-1]
                context_size = getattr(lm, "context_size", 2)
                context = [tokenizer._id_to_char.get(i, "<unk>") for i in parent[-context_size:]]
                context = ["<s>"] * (context_size - len(context)) + context
                lm_scores[prefix] = lm_scores[parent] + lm.score(
                    tokenizer._id_to_char.get(prefix[-1], "<unk>"), context
                )
            bonus = lm_weight * lm_scores[prefix]
        else:
            bonus = 0.0
        return _logsumexp(*probabilities) + bonus + token_reward * len(prefix)

    BLANK = tokenizer.blank_id
    # Skip tokens that should never appear in CTC output
    SKIP = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id}
    NEG_INF = float("-inf")

    V = log_probs_1d.shape[1]

    # beam: dict mapping (prefix_tuple, last_char_id_or_None) → (log_Pb, log_Pn)
    # Initial state: empty prefix, no last char, all probability is "blank"
    # (log_Pb=0 means Pb=1.0; log_Pn=-inf means Pn=0.0)
    beam: dict[tuple, tuple[float, float]] = {((), None): (0.0, NEG_INF)}

    for t in range(log_probs_1d.shape[0]):
        lp = log_probs_1d[t]  # (V,) log-probabilities at frame t
        new_beam: dict[tuple, tuple[float, float]] = {}

        def _update(key: tuple, add_log_pb: float, add_log_pn: float) -> None:
            old_pb, old_pn = new_beam.get(key, (NEG_INF, NEG_INF))
            new_beam[key] = (
                _logsumexp(old_pb, add_log_pb),
                _logsumexp(old_pn, add_log_pn),
            )

        for (prefix, last_c), (log_pb, log_pn) in beam.items():
            log_p_total = _logsumexp(log_pb, log_pn)

            # Case 1: output blank
            _update((prefix, last_c), log_p_total + lp[BLANK], NEG_INF)

            # Case 2 & 3: output non-blank character
            for c in range(V):
                if c == BLANK or c in SKIP:
                    continue
                log_pc = float(lp[c])
                if log_pc == NEG_INF:
                    continue

                if c == last_c:
                    # Same char as the last non-blank in prefix:
                    # Either "hold" (doesn't extend prefix, uses Pn path)
                    _update((prefix, last_c), NEG_INF, log_pn + log_pc)
                    # Or "new occurrence after blank" (extends prefix, uses Pb path)
                    _update((prefix + (c,), c), NEG_INF, log_pb + log_pc)
                else:
                    # Different char: always extends the prefix
                    _update((prefix + (c,), c), NEG_INF, log_p_total + log_pc)

        # Prune to top beam_width hypotheses by total log-probability
        beam = dict(
            sorted(
                new_beam.items(),
                key=lambda kv: rank(kv[0], kv[1]),
                reverse=True,
            )[:beam_width]
        )
        if lm is not None and lm_weight:
            # Only surviving prefixes can become parents on the next frame.
            lm_scores = {key[0]: lm_scores[key[0]] for key in beam}

    # Return the best prefix (highest total log-probability)
    best_key = max(beam, key=lambda k: rank(k, beam[k]))
    best_prefix_ids = list(best_key[0])
    return tokenizer.decode(best_prefix_ids, skip_special=True)


def beam_decode(
    log_probs: torch.Tensor,
    tokenizer,
    beam_width: int = 10,
    lm=None,
    lm_weight: float = 0.0,
    token_reward: float = 0.0,
) -> list[str]:
    """Run ``prefix_beam_search`` on each element of a batch.

    Args:
        log_probs:  (T, B, vocab_size) model output, time-first
        tokenizer:  CharTokenizer
        beam_width: beams per example

    Returns:
        one decoded string per batch element
    """
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(1)

    T, B, V = log_probs.shape
    lp_np = log_probs.detach().cpu().numpy()  # (T, B, V)

    results = []
    for b in range(B):
        # lp_np[:, b, :] has shape (T, V) — exactly what prefix_beam_search expects
        text = prefix_beam_search(
            lp_np[:, b, :],
            tokenizer,
            beam_width,
            lm=lm,
            lm_weight=lm_weight,
            token_reward=token_reward,
        )
        results.append(text)

    return results
