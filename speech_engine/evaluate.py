"""
speech_engine/evaluate.py — CER and WER computed from scratch.

No jiwer. No external libraries. Just the Levenshtein edit distance algorithm
implemented directly, then applied at character level (CER) and word level (WER).

Why implement it ourselves?
    The algorithm is short (~25 lines), the math is fundamental to all NLP
    evaluation, and understanding it means you can reason about why a model
    with CER=15% is better or worse than one with WER=20%. Libraries hide this.

Levenshtein distance:
    The minimum number of single-character edits (insertions, deletions,
    substitutions) required to change one sequence into another.

    For ASR:
        reference = "hello world"        (what was actually said)
        hypothesis = "helo wrld"         (what the model predicted)
        edits: delete one 'l', insert 'o' back... → distance = 3

CER (Character Error Rate):
    edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    Measures how many characters the model got wrong, as a fraction of the total.
    CER=0.0 is perfect, CER=1.0 means every character was wrong.

WER (Word Error Rate):
    edit_distance(ref_words, hyp_words) / len(ref_words)
    Same idea but at word level. WER is the standard benchmark metric for ASR.
    Whisper large-v3 achieves WER < 5% on clean English. Our CTC model
    trained on 100h should aim for WER < 30% on LibriSpeech test-clean.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Callable


# ── Edit distance (Levenshtein) ───────────────────────────────────────────────

def levenshtein(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Uses the standard O(|ref| * |hyp|) dynamic programming algorithm
    with an O(|hyp|) space optimisation (rolling row).

    Args:
        ref: reference sequence (list of chars or words)
        hyp: hypothesis sequence (list of chars or words)

    Returns:
        int — minimum edit distance (insertions + deletions + substitutions)

    Example::

        levenshtein(list("hello"), list("helo"))   # → 1  (one deletion)
        levenshtein(["the", "cat"], ["a", "cat"])  # → 1  (one substitution)
    """
    R, H = len(ref), len(hyp)

    if R == 0:
        return H   # all insertions
    if H == 0:
        return R   # all deletions

    # dp[j] = edit distance between ref[:i] and hyp[:j]
    # We only keep the current and previous rows (space optimisation)
    prev = list(range(H + 1))  # prev[j] = dp[0][j] = j insertions

    for i in range(1, R + 1):
        curr = [i] + [0] * H   # curr[0] = i deletions
        for j in range(1, H + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]   # characters match — no edit needed
            else:
                curr[j] = 1 + min(
                    prev[j],      # deletion:     remove ref[i-1]
                    curr[j - 1],  # insertion:    insert hyp[j-1]
                    prev[j - 1],  # substitution: replace ref[i-1] with hyp[j-1]
                )
        prev = curr

    return prev[H]


# ── CER and WER ───────────────────────────────────────────────────────────────

def char_error_rate(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """Character Error Rate over a list of (reference, hypothesis) pairs.

    CER = sum(edit_distance(ref_chars, hyp_chars)) / sum(len(ref_chars))

    The numerator and denominator are summed across all pairs before dividing.
    This is the corpus-level CER, which weights longer utterances more heavily
    and is the standard way to report the metric.

    Args:
        references:  list of reference strings (ground truth)
        hypotheses:  list of hypothesis strings (model output)

    Returns:
        float — CER in [0, ∞). Values > 1.0 are possible when the hypothesis
        contains many insertions relative to the reference length.

    Example::

        char_error_rate(["hello world"], ["helo wrld"])  # → 3/11 ≈ 0.273
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Lengths must match: {len(references)} references, {len(hypotheses)} hypotheses"
        )

    total_edits = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        total_edits += levenshtein(ref_chars, hyp_chars)
        total_chars += len(ref_chars)

    if total_chars == 0:
        return 0.0

    return total_edits / total_chars


def word_error_rate(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """Word Error Rate over a list of (reference, hypothesis) pairs.

    WER = sum(edit_distance(ref_words, hyp_words)) / sum(len(ref_words))

    This is the standard ASR evaluation metric. Each word is treated as one
    unit; a substitution of "world" → "word" costs 1 edit, same as a deletion.

    Args:
        references:  list of reference strings (space-separated words)
        hypotheses:  list of hypothesis strings (space-separated words)

    Returns:
        float — WER in [0, ∞).

    Example::

        word_error_rate(["hello world"], ["hello"])  # → 1/2 = 0.5
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Lengths must match: {len(references)} references, {len(hypotheses)} hypotheses"
        )

    total_edits = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_edits += levenshtein(ref_words, hyp_words)
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0

    return total_edits / total_words


# ── Dataset-level evaluation ──────────────────────────────────────────────────

def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer,
    decoder_fn: Callable,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """Run the model over a DataLoader and compute CER and WER.

    Args:
        model:       CTCModel (or any model with the same forward signature)
        loader:      DataLoader yielding {"mel", "mel_lengths", "tokens", "token_lengths"}
        tokenizer:   CharTokenizer for decoding token IDs back to text
        decoder_fn:  callable(log_probs, tokenizer) → list[str]
                     e.g. greedy_decode from decode.py
        device:      torch.device to run inference on
        max_batches: if set, only evaluate this many batches (for quick checks)

    Returns:
        dict with keys:
            "cer":       float — corpus CER
            "wer":       float — corpus WER
            "n_examples": int — number of examples evaluated
            "examples":  list of {"ref": str, "hyp": str, "cer": float, "wer": float}
                         for the first 10 examples (for inspection)
    """
    model.eval()

    all_refs: list[str] = []
    all_hyps: list[str] = []
    examples: list[dict] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            mel          = batch["mel"].to(device)
            mel_lengths  = batch["mel_lengths"].to(device)
            token_seqs   = batch["tokens"]
            token_lengths = batch["token_lengths"]

            # Forward pass
            log_probs = model(mel)                             # (T_out, B, V)

            # Decode hypotheses
            hyps = decoder_fn(log_probs, tokenizer)

            # Decode references from token IDs
            refs = []
            for i in range(token_seqs.shape[0]):
                L = int(token_lengths[i])
                ids = token_seqs[i, :L].tolist()
                refs.append(tokenizer.decode(ids))

            all_refs.extend(refs)
            all_hyps.extend(hyps)

            # Save first 10 examples for inspection
            for ref, hyp in zip(refs, hyps):
                if len(examples) < 10:
                    ex_cer = char_error_rate([ref], [hyp])
                    ex_wer = word_error_rate([ref], [hyp])
                    examples.append({"ref": ref, "hyp": hyp, "cer": ex_cer, "wer": ex_wer})

    cer = char_error_rate(all_refs, all_hyps)
    wer = word_error_rate(all_refs, all_hyps)

    return {
        "cer": cer,
        "wer": wer,
        "n_examples": len(all_refs),
        "examples": examples,
    }


# Avoid circular import with Optional type hint
from typing import Optional
