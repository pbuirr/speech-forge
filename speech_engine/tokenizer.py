"""
speech_engine/tokenizer.py — Character-level tokenizer for ASR.

Maps between text strings and sequences of integer IDs that the model
can compute on. We start character-level because:

  - Zero out-of-vocabulary problem: any character seen at inference was in training
  - The model learns to spell, not just recognize whole-word tokens
  - Straightforward to extend to new languages: just add their characters
  - Easy to debug: you can read the vocabulary by inspection

The vocabulary layout is fixed and intentional:

    ID 0 → <blank>   Required at index 0 for nn.CTCLoss (blank=0 is the default)
    ID 1 → <pad>     Padding token for batching variable-length sequences
    ID 2 → <bos>     Beginning-of-sequence (used by the seq2seq decoder in Phase 4)
    ID 3 → <eos>     End-of-sequence marker
    ID 4+ → chars    Actual characters, starting with the English base set

CTC decoding refresher:
    The CTC model outputs a probability distribution over the vocabulary
    (including <blank>) at every time frame. The blank token means "no output
    at this frame." Consecutive identical non-blank outputs are collapsed into
    one. So "hh<blank>ee<blank>lll<blank>lo" → "hello".

Phase 4 upgrade path:
    The CharTokenizer interface is a drop-in replacement target for a BPE
    tokenizer. The model_transformer.py will call .encode() and .decode()
    the same way. Switching from chars to BPE subwords only requires swapping
    the tokenizer and retraining — nothing else changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ── Special token IDs (never change, regardless of language) ──────────────────
BLANK_ID      = 0   # CTC blank — must be 0 for nn.CTCLoss default
PAD_ID        = 1   # padding
BOS_ID        = 2   # beginning of sequence
EOS_ID        = 3   # end of sequence
FIRST_CHAR_ID = 4   # first real character starts here

_SPECIAL_TOKENS: dict[str, int] = {
    "<blank>": BLANK_ID,
    "<pad>":   PAD_ID,
    "<bos>":   BOS_ID,
    "<eos>":   EOS_ID,
}

# Pre-built English base: 26 lowercase letters + space + apostrophe + hyphen.
# Covers ~99.9% of LibriSpeech and Common Voice (English) transcripts.
# Order is alphabetical so the vocabulary is deterministic and human-readable.
_ENGLISH_CHARS: list[str] = sorted("abcdefghijklmnopqrstuvwxyz '-")


# ── Text normalization ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize a raw transcript into the character set the model uses.

    Steps applied in order:
        1. Lowercase
        2. Keep only: letters a-z, space, apostrophe, hyphen
           (strips digits, punctuation, accents that aren't in the base set)
        3. Collapse runs of whitespace to a single space
        4. Strip leading/trailing whitespace

    Why strip digits?
        LibriSpeech and most clean ASR corpora spell out numbers ("nineteen
        eighty four" not "1984"). If your dataset has digits, call .fit() on
        your transcripts and they'll be added to the vocabulary automatically.

    Example:
        "Hello, World! It's 2024." → "hello world it's"
    """
    text = text.lower()
    text = re.sub(r"[^a-z '-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class CharTokenizer:
    """Character-level tokenizer with fixed special tokens and an extensible vocabulary.

    Out of the box it handles English. Call .fit(transcripts) to extend the
    vocabulary with characters from any language (French accents, Arabic script,
    etc.) without breaking the existing ID assignments.

    Usage:
        tok = CharTokenizer()
        ids = tok.encode("hello world")      # [11, 8, 15, 15, 18, 1, 26, 18, 21, 15, 7]
        text = tok.decode(ids)               # "hello world"
        text = tok.ctc_collapse(model_out)   # greedy CTC decode
        tok.save("data/tokenizer.json")
        tok2 = CharTokenizer.load("data/tokenizer.json")
    """

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._init_special_tokens()
        self._add_chars(_ENGLISH_CHARS)

    def _init_special_tokens(self) -> None:
        for token, idx in _SPECIAL_TOKENS.items():
            self._char_to_id[token] = idx
            self._id_to_char[idx] = token

    def _add_chars(self, chars: list[str]) -> None:
        """Append new characters to the vocabulary, skipping any already present."""
        next_id = max(self._id_to_char) + 1 if self._id_to_char else FIRST_CHAR_ID
        for ch in chars:
            if ch not in self._char_to_id:
                self._char_to_id[ch] = next_id
                self._id_to_char[next_id] = ch
                next_id += 1

    # ── Building vocabulary ────────────────────────────────────────────────────

    def fit(self, transcripts: list[str]) -> "CharTokenizer":
        """Extend the vocabulary with any characters found in the corpus.

        Existing character IDs are never changed. New characters are appended
        at the end of the vocabulary in sorted order (for determinism).

        Args:
            transcripts: list of raw transcript strings (normalization applied internally)

        Returns:
            self (for chaining: tokenizer = CharTokenizer().fit(my_transcripts))
        """
        seen = set()
        for t in transcripts:
            seen.update(normalize_text(t))
        new_chars = sorted(c for c in seen if c not in self._char_to_id)
        if new_chars:
            self._add_chars(new_chars)
            print(f"[tokenizer] Added {len(new_chars)} new chars: {new_chars}")
        return self

    # ── Encoding ───────────────────────────────────────────────────────────────

    def encode(self, text: str, normalize: bool = True) -> list[int]:
        """Convert a text string to a list of character IDs.

        Args:
            text:      raw transcript string
            normalize: if True, applies normalize_text() first (recommended)

        Returns:
            list[int] — no BOS or EOS tokens (use encode_with_eos for seq2seq)

        Unknown characters are silently skipped. After normalize_text(), this
        should only affect characters outside the base Latin set.
        """
        if normalize:
            text = normalize_text(text)
        return [self._char_to_id[ch] for ch in text if ch in self._char_to_id]

    def encode_with_eos(self, text: str, normalize: bool = True) -> list[int]:
        """Encode with BOS at the start and EOS at the end.

        Used by the seq2seq decoder in Phase 4. The decoder is trained to
        start from BOS and predict each character until it outputs EOS.
        """
        return [BOS_ID] + self.encode(text, normalize) + [EOS_ID]

    # ── Decoding ───────────────────────────────────────────────────────────────

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert a list of IDs back to a text string.

        Args:
            ids:          list of integer token IDs
            skip_special: if True, removes <blank>, <pad>, <bos>, <eos> from output.
                          Set False to inspect raw model output including specials.

        Returns:
            str — reconstructed text
        """
        special_ids = set(_SPECIAL_TOKENS.values()) if skip_special else set()
        return "".join(
            self._id_to_char.get(i, "?")
            for i in ids
            if i not in special_ids
        )

    def ctc_collapse(self, ids: list[int]) -> str:
        """Greedy CTC decode: collapse consecutive runs, then remove blanks.

        CTC allows the model to output the same character multiple times in
        a row (to "hold" a prediction while uncertain) and uses blank to
        separate two intentionally identical consecutive characters.

        Order matters — collapse runs FIRST, strip blanks SECOND:
            If you strip blanks first, [l, <blank>, l] → [l, l] → [l],
            incorrectly merging two distinct 'l' characters into one.

        Example — model output for "hello":
            [h, h, <blank>, e, l, l, <blank>, l, o, o]
            → collapse consecutive runs: [h, <blank>, e, l, <blank>, l, o]
            → remove blanks:             [h, e, l, l, o]
            → decode:                    "hello"  ✓

        This is greedy (best-probability path at each frame). Beam search
        decoding (Phase 4) considers multiple hypotheses and gives lower WER.
        """
        # Step 1: collapse consecutive identical tokens (merge repeated outputs)
        deduped = [tok for i, tok in enumerate(ids) if i == 0 or tok != ids[i - 1]]
        # Step 2: remove blank tokens — what remains is the predicted transcript
        no_blank = [tok for tok in deduped if tok != BLANK_ID]
        return self.decode(no_blank)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize vocabulary to a JSON file.

        The file stores char_to_id so loading reconstructs both dicts.
        Human-readable: open it in any text editor to inspect the vocabulary.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": "1.0", "char_to_id": self._char_to_id}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load vocabulary from a JSON file written by .save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj._char_to_id = data["char_to_id"]
        obj._id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        return obj

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including all special tokens and characters."""
        return len(self._id_to_char)

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def __repr__(self) -> str:
        char_count = self.vocab_size - len(_SPECIAL_TOKENS)
        return (
            f"CharTokenizer(vocab_size={self.vocab_size}, "
            f"chars={char_count}, "
            f"blank={BLANK_ID}, pad={PAD_ID}, bos={BOS_ID}, eos={EOS_ID})"
        )
