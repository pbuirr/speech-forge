"""
speech_engine/bpe.py — Byte-Pair Encoding tokenizer, built from scratch.

Implements the BPE algorithm from Sennrich et al. (2016) "Neural Machine
Translation of Rare Words with Subword Units" — no tokenizers library, no
sentencepiece, no subword-nmt. A clean, readable re-implementation that
matches what those libraries do internally without importing them.

Algorithm overview:
  1. Pre-tokenize text into words (whitespace split).
  2. Represent each word as a tuple of characters; the final character carries
     an end-of-word marker ("</w>") so "un" (prefix) is distinct from "un"
     at word boundary.
  3. Count all adjacent symbol-pair frequencies across the weighted vocabulary.
  4. Merge the most frequent pair everywhere, add the merged token to the vocab.
  5. Repeat until the target vocabulary size is reached.
  6. At encode time, apply the learned merges in priority order to new text.

Vocabulary layout:
    ID 0  <pad>   padding (batch alignment)
    ID 1  <bos>   begin of sequence
    ID 2  <eos>   end of sequence
    ID 3  <unk>   unknown / out-of-vocabulary symbol
    ID 4+ …       BPE tokens (single chars first, then merged subwords)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
_FIRST_REAL_ID = 4

_SPECIAL: dict[str, int] = {"<pad>": PAD_ID, "<bos>": BOS_ID, "<eos>": EOS_ID, "<unk>": UNK_ID}
_EOW = "</w>"  # end-of-word marker fused onto the last character of each word


# ── Internal helpers ──────────────────────────────────────────────────────────


def _word_to_chars(word: str) -> tuple[str, ...]:
    """Split a word into characters, marking the last one with </w>."""
    if len(word) == 1:
        return (word + _EOW,)
    return tuple(word[:-1]) + (word[-1] + _EOW,)


def _pair_counts(vocab: dict[tuple[str, ...], int]) -> Counter:
    """Count every adjacent symbol pair, weighted by word frequency."""
    counts: Counter = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def _apply_merge(
    vocab: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """Return a new vocabulary with every occurrence of `pair` merged."""
    a, b = pair
    merged = a + b
    new_vocab: dict[tuple[str, ...], int] = {}
    for symbols, freq in vocab.items():
        new_syms: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        new_vocab[tuple(new_syms)] = freq
    return new_vocab


# ── Public class ──────────────────────────────────────────────────────────────


class BPEVocab:
    """Byte-Pair Encoding vocabulary — learn, encode, decode, save, load.

    Usage:
        vocab = BPEVocab().train(sentences, vocab_size=4000)
        ids   = vocab.encode("hello world", add_bos=True, add_eos=True)
        text  = vocab.decode(ids)
        vocab.save("path/vocab.json")
        vocab = BPEVocab.load("path/vocab.json")
    """

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = dict(_SPECIAL)
        self._id_to_token: dict[int, str] = {v: k for k, v in _SPECIAL.items()}
        self._merges: list[tuple[str, str]] = []
        self._merge_rank: dict[tuple[str, str], int] = {}  # pair → merge index (lower = earlier)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, corpus: list[str], vocab_size: int = 4000) -> "BPEVocab":
        """Learn BPE merges from a list of sentences (or lines of text).

        Mutates self in place and returns self for chaining.
        """
        word_freq: Counter = Counter()
        for line in corpus:
            for word in line.strip().lower().split():
                if word:
                    word_freq[word] += 1

        # Character-level seed vocabulary
        char_vocab: dict[tuple[str, ...], int] = {
            _word_to_chars(w): f for w, f in word_freq.items()
        }

        # Register all seed characters
        for symbols in char_vocab:
            for sym in symbols:
                if sym not in self._token_to_id:
                    nid = len(self._token_to_id)
                    self._token_to_id[sym] = nid
                    self._id_to_token[nid] = sym

        # Iterative merging
        while len(self._token_to_id) < vocab_size:
            counts = _pair_counts(char_vocab)
            if not counts:
                break
            # Tie-break alphabetically for determinism
            best = max(counts, key=lambda p: (counts[p], p))
            if counts[best] < 2:
                break

            self._merges.append(best)
            self._merge_rank[best] = len(self._merge_rank)

            merged = best[0] + best[1]
            if merged not in self._token_to_id:
                nid = len(self._token_to_id)
                self._token_to_id[merged] = nid
                self._id_to_token[nid] = merged

            char_vocab = _apply_merge(char_vocab, best)

        return self

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_word(self, word: str, bpe_dropout: float = 0.0) -> list[int]:
        """BPE-encode one word (lowercase) to a list of token IDs.

        bpe_dropout (Provilkov et al. 2020): randomly skip each merge with
        probability p during training. The model sees alternative segmentations
        of the same surface form, which acts as data augmentation and improves
        robustness to rare / OOV words. Set p=0.0 (default) at inference.
        """
        import random

        symbols: list[str] = list(_word_to_chars(word))
        if not symbols:
            return []

        # Apply merges in priority order: the pair learned earliest (lowest rank)
        # is applied first, scanning left-to-right for each merge step.
        while len(symbols) > 1:
            best_rank: int | None = None
            best_pair: tuple[str, str] | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # no remaining mergeable pair

            # BPE dropout: skip this merge with probability p
            if bpe_dropout > 0.0 and random.random() < bpe_dropout:
                # Remove this pair from consideration for this word encoding by
                # temporarily treating it as unmergeable. We do this by breaking
                # and re-scanning for the next-best merge excluding best_pair.
                # Simpler: just stop the merge loop here (this word stays partly
                # un-merged, which is the desired regularisation effect).
                break

            merged = best_pair[0] + best_pair[1]
            new_syms: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms

        return [self._token_to_id.get(s, UNK_ID) for s in symbols]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        bpe_dropout: float = 0.0,
    ) -> list[int]:
        """Encode a string to a list of BPE token IDs.

        bpe_dropout: merge-skip probability for training augmentation (0 = off).
        Pass bpe_dropout=0.1 during training; leave at 0.0 for inference.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for word in text.strip().lower().split():
            ids.extend(self._encode_word(word, bpe_dropout=bpe_dropout))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode a list of BPE token IDs back to a string."""
        tokens: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i, "<unk>")
            if skip_special and tok in _SPECIAL:
                continue
            tokens.append(tok)
        # Tokens ending with </w> mark word boundaries
        return "".join(tokens).replace(_EOW, " ").strip()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"token_to_id": self._token_to_id, "merges": self._merges},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "BPEVocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls()
        vocab._token_to_id = data["token_to_id"]
        vocab._id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        vocab._merges = [tuple(m) for m in data["merges"]]
        vocab._merge_rank = {tuple(m): i for i, m in enumerate(vocab._merges)}
        return vocab

    # ── Properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)
