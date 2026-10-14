"""Character tokenizer for CTC speech recognition.

IDs: 0 <blank> (CTC blank), 1 <pad>, 2 <bos>, 3 <eos>, characters from 4.
The default English normalisation matches the existing checkpoints; use
``normalization='unicode'`` for other scripts. ``ctc_collapse`` removes
repeats and blanks from a frame-level sequence.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# Special token IDs (never change, regardless of language)
BLANK_ID = 0  # CTC blank — must be 0 for nn.CTCLoss default
PAD_ID = 1  # padding
BOS_ID = 2  # beginning of sequence
EOS_ID = 3  # end of sequence
FIRST_CHAR_ID = 4  # first real character starts here

_SPECIAL_TOKENS: dict[str, int] = {
    "<blank>": BLANK_ID,
    "<pad>": PAD_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
}

# Pre-built English base: 26 lowercase letters + space + apostrophe + hyphen.
# Coverage must be checked against each corpus; unsupported lexical characters
# are rejected by the supervised dataset instead of silently discarded.
# Order is alphabetical so the vocabulary is deterministic and human-readable.
_ENGLISH_CHARS: list[str] = sorted("abcdefghijklmnopqrstuvwxyz '-")


# Text normalization


def normalize_text(text: str, mode: str = "english") -> str:
    """Normalize a raw transcript into the character set the model uses.

    Steps applied in order:
        1. Lowercase
        2. Keep only: letters a-z, space, apostrophe, hyphen
           (strips digits, punctuation, accents that aren't in the base set)
        3. Collapse runs of whitespace to a single space
        4. Strip leading/trailing whitespace

    Example:
        "Hello, World! It's 2024." → "hello world it's"
    """
    if mode not in ("english", "unicode"):
        raise ValueError("Unknown transcript normalization policy")
    text = unicodedata.normalize("NFC", text.lower())
    if mode == "unicode":
        text = "".join(
            c if unicodedata.category(c)[0] in "LMN" or c in " '-" else " " for c in text
        )
    else:
        text = re.sub(r"[^a-z '-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Tokenizer


class CharTokenizer:
    """Character tokenizer with fixed special tokens and an extensible vocabulary.

    The default English normalisation keeps compatibility with existing
    checkpoints. ``CharTokenizer(normalization='unicode')`` keeps letters,
    combining marks and digits for other scripts; the policy is saved with the
    tokenizer.

    Usage::

        tok = CharTokenizer().fit(transcripts)
        ids = tok.encode("hello world")
        text = tok.decode(ids)
        tok.save("data/tokenizer.json")
        tok = CharTokenizer.load("data/tokenizer.json")
    """

    def __init__(self, normalization: str = "english") -> None:
        normalize_text("", normalization)  # Validate and persist the training policy.
        self.normalization = normalization
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

    # Building vocabulary

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
            seen.update(normalize_text(t, self.normalization))
        new_chars = sorted(c for c in seen if c not in self._char_to_id)
        if new_chars:
            self._add_chars(new_chars)
            print(f"[tokenizer] Added {len(new_chars)} new chars: {new_chars}")
        return self

    # Encoding

    def encode(self, text: str, normalize: bool = True) -> list[int]:
        """Convert text to character IDs, skipping characters not in the vocabulary.

        Args:
            text:      raw transcript
            normalize: apply ``normalize_text`` first

        Returns:
            list[int] without BOS/EOS (see ``encode_with_eos``)
        """
        if normalize:
            text = normalize_text(text, self.normalization)
        return [self._char_to_id[ch] for ch in text if ch in self._char_to_id]

    def encode_with_eos(self, text: str, normalize: bool = True) -> list[int]:
        """Encode with BOS and EOS, for the sequence-to-sequence decoder."""
        return [BOS_ID] + self.encode(text, normalize) + [EOS_ID]

    # Decoding

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
        return "".join(self._id_to_char.get(i, "?") for i in ids if i not in special_ids)

    def ctc_collapse(self, ids: list[int]) -> str:
        """Greedy CTC collapse: merge consecutive repeats, then remove blanks.

        The order matters: [l, <blank>, l] must stay "ll".

            [h, h, <blank>, e, l, l, <blank>, l, o, o] → "hello"
        """
        # Step 1: collapse consecutive identical tokens (merge repeated outputs)
        deduped = [tok for i, tok in enumerate(ids) if i == 0 or tok != ids[i - 1]]
        # Step 2: remove blank tokens — what remains is the predicted transcript
        no_blank = [tok for tok in deduped if tok != BLANK_ID]
        return self.decode(no_blank)

    # Persistence

    def save(self, path: str | Path) -> None:
        """Save the vocabulary and normalisation policy as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.1",
            "normalization": self.normalization,
            "char_to_id": self._char_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load vocabulary from a JSON file written by .save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.normalization = data.get("normalization", "english")
        normalize_text("", obj.normalization)
        obj._char_to_id = data["char_to_id"]
        obj._id_to_char = {int(v): k for k, v in data["char_to_id"].items()}
        return obj

    # Properties

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
