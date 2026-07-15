"""
speech_engine/decode.py — CTC decoders, implemented from scratch.

Two decoders are provided. Both are written by hand — no ctcdecode library,
no external beam search package.

Greedy decode:
    At each time step, pick the token with the highest log-probability.
    Apply CTC collapse (merge consecutive duplicates, remove blanks).
    Fast (O(T * B)) but suboptimal — it can't recover from a bad local decision.

Prefix beam search:
    Maintain the top-k most probable prefixes across the whole sequence.
    At each time step, extend each beam with every possible next token and
    re-score. This allows the decoder to consider multiple transcription
    hypotheses simultaneously and pick the globally best one.

    Implemented in log-space for numerical stability (we sum log-probabilities
    instead of multiplying raw probabilities, which would underflow to 0.0
    for long sequences).

Why beam search gives lower WER:
    Greedy decode commits to the locally best token at every frame. If the
    model is momentarily uncertain and picks the wrong character, it's stuck.
    Beam search keeps k alternative hypotheses alive, so a wrong local decision
    can be overruled later when the surrounding context makes the right answer
    clearer.

    Example: the model might output 'h' at frame t with log-prob -0.3, and
    'n' with log-prob -0.4. Greedy picks 'h'. But if the next 10 frames
    strongly support "new" and not "hew", beam search (with k≥2) would
    have kept the 'n' hypothesis alive and ultimately produce the right word.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import numpy as np


# ── Greedy CTC decode ─────────────────────────────────────────────────────────

def greedy_decode(
    log_probs: torch.Tensor,
    tokenizer,
) -> list[str]:
    """Greedy CTC decode: argmax at every frame, then apply CTC collapse.

    Args:
        log_probs: (T, B, vocab_size) — model output, time-first
        tokenizer: CharTokenizer for converting IDs to text

    Returns:
        list of decoded strings, one per batch element

    This is O(T * B) and deterministic. For real data with a well-trained
    model, greedy decode typically gives WER within ~2% absolute of beam search
    with beam_width=10. For poorly trained or ambiguous outputs, the gap is larger.
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


# ── CTC prefix beam search ────────────────────────────────────────────────────

def _logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)).

    Working in log-space: instead of multiplying probabilities (which
    underflow to 0.0 for long sequences), we add log-probabilities.
    But adding two log-probabilities means computing log(exp(a) + exp(b))
    — which can overflow if a and b are large. The trick is:
        log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(min-max))
    The subtraction exp(min-max) is always ≤ 1.0, so no overflow.
    """
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
        lp = log_probs_1d[t]           # (V,) log-probabilities at frame t
        new_beam: dict[tuple, tuple[float, float]] = {}

        def _update(key: tuple, add_log_pb: float, add_log_pn: float) -> None:
            old_pb, old_pn = new_beam.get(key, (NEG_INF, NEG_INF))
            new_beam[key] = (
                _logsumexp(old_pb, add_log_pb),
                _logsumexp(old_pn, add_log_pn),
            )

        for (prefix, last_c), (log_pb, log_pn) in beam.items():
            log_p_total = _logsumexp(log_pb, log_pn)

            # ── Case 1: output blank ──────────────────────────────────────────
            _update((prefix, last_c), log_p_total + lp[BLANK], NEG_INF)

            # ── Case 2 & 3: output non-blank character ────────────────────────
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
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    # Return the best prefix (highest total log-probability)
    best_key = max(beam, key=lambda k: _logsumexp(beam[k][0], beam[k][1]))
    best_prefix_ids = list(best_key[0])
    return tokenizer.decode(best_prefix_ids, skip_special=True)


def beam_decode(
    log_probs: torch.Tensor,
    tokenizer,
    beam_width: int = 10,
) -> list[str]:
    """Batch wrapper for prefix_beam_search.

    Runs prefix_beam_search independently on each element in the batch.
    Beam search is sequential per example — parallelism happens across
    the batch (could be run with multiprocessing.Pool for large batches).

    Args:
        log_probs:  (T, B, vocab_size) — model output, time-first
        tokenizer:  CharTokenizer
        beam_width: number of beams per example

    Returns:
        list of decoded strings, one per batch element
    """
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(1)

    T, B, V = log_probs.shape
    lp_np = log_probs.detach().cpu().numpy()  # (T, B, V)

    results = []
    for b in range(B):
        # lp_np[:, b, :] has shape (T, V) — exactly what prefix_beam_search expects
        text = prefix_beam_search(lp_np[:, b, :], tokenizer, beam_width)
        results.append(text)

    return results
