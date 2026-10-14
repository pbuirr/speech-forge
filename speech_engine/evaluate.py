"""Edit-distance metrics for speech recognition.

``char_error_rate`` and ``word_error_rate`` are Levenshtein distance divided by
reference length. ``edit_counts`` and ``align_units`` break errors into
substitutions, deletions and insertions.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from typing import Callable


# Edit distance (Levenshtein)


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
        return H  # all insertions
    if H == 0:
        return R  # all deletions

    # dp[j] = edit distance between ref[:i] and hyp[:j]
    # We only keep the current and previous rows (space optimisation)
    prev = list(range(H + 1))  # prev[j] = dp[0][j] = j insertions

    for i in range(1, R + 1):
        curr = [i] + [0] * H  # curr[0] = i deletions
        for j in range(1, H + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]  # characters match — no edit needed
            else:
                curr[j] = 1 + min(
                    prev[j],  # deletion:     remove ref[i-1]
                    curr[j - 1],  # insertion:    insert hyp[j-1]
                    prev[j - 1],  # substitution: replace ref[i-1] with hyp[j-1]
                )
        prev = curr

    return prev[H]


# CER and WER


def align_units(reference: list, hypothesis: list) -> list:
    """One minimum-edit alignment as (reference, hypothesis) pairs.

    None marks an insertion or deletion. Ties prefer substitution, so the result
    is deterministic.
    """
    table = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        table[i][0] = i
    for j in range(len(hypothesis) + 1):
        table[0][j] = j
    for i, r in enumerate(reference, 1):
        for j, h in enumerate(hypothesis, 1):
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + (r != h)
            )
    pairs = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and table[i][j] == table[i - 1][j - 1]:
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and j and table[i][j] == table[i - 1][j - 1] + 1:
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hypothesis[j - 1]))
            j -= 1
    return pairs[::-1]


def edit_counts(reference: list, hypothesis: list) -> dict:
    """Substitution, deletion and insertion counts from ``align_units``.

    Works on words, characters or any other token units.
    """
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0}
    for r, h in align_units(reference, hypothesis):
        counts[
            "insertions"
            if r is None
            else "deletions"
            if h is None
            else "matches"
            if r == h
            else "substitutions"
        ] += 1
    return counts


def char_error_rate(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """Corpus-level character error rate.

        CER = Σ edit_distance(ref, hyp) / Σ len(ref)

    Totals are summed before dividing, so longer utterances weigh more. Values
    above 1.0 are possible with many insertions.

    Example::

        char_error_rate(["hello world"], ["helo wrld"])  # 3/11 ≈ 0.273
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
    """Corpus-level word error rate over whitespace-separated words.

        WER = Σ edit_distance(ref_words, hyp_words) / Σ len(ref_words)

    Example::

        word_error_rate(["hello world"], ["hello"])  # 0.5
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


# Dataset-level evaluation


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

            mel = batch["mel"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            token_seqs = batch["tokens"]
            token_lengths = batch["token_lengths"]

            # Forward pass
            log_probs = model(mel, mel_lengths)  # (T_out, B, V)

            # Decode hypotheses
            # Padded frames must never contribute characters to the hypothesis.
            output_lengths = model.encoder_output_length(mel_lengths)
            hyps = []
            for i, length in enumerate(output_lengths):
                hyps.extend(decoder_fn(log_probs[: int(length), i : i + 1], tokenizer))

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
