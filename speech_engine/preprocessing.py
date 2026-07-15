"""
speech_engine/preprocessing.py — Linguistic preprocessing for NMT input.

Rather than passing raw text straight to BPE, we normalize constructs the
neural model has never seen: currency symbols ($20 → "20 dólares"), percentages,
abbreviations, and mixed-language tokens. We also extract *linguistic anchors*
— numbers, amounts, dates — so the translation layer can verify the output
actually covers what the source said.

This is the gap between a neural lookup table (memorize training pairs) and a
language-aware system (understand structure and route unknown constructs
intelligently).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Linguistic constants ──────────────────────────────────────────────────────

_CURRENCY_ES = {
    "$": "dólares", "€": "euros", "£": "libras", "¥": "yenes",
}
_CURRENCY_EN = {
    "$": "dollars", "€": "euros", "£": "pounds", "¥": "yen",
}

_ABBREV_ES: dict[str, str] = {
    "Sr.":    "señor",
    "Sra.":   "señora",
    "Srta.":  "señorita",
    "Dr.":    "doctor",
    "Dra.":   "doctora",
    "Prof.":  "profesor",
    "aprox.": "aproximadamente",
    "p.ej.":  "por ejemplo",
    "i.e.":   "es decir",
    "e.g.":   "por ejemplo",
    "Av.":    "avenida",
    "km.":    "kilómetros",
    "m.":     "metros",
    "EE.UU.": "estados unidos",
    "EUA":    "estados unidos",
}

# Spanish ordinals → cardinal for simpler NMT input
_ORDINAL_ES = {
    "1º": "primero", "2º": "segundo", "3º": "tercero",
    "1ª": "primera", "2ª": "segunda", "3ª": "tercera",
    "4º": "cuarto",  "5º": "quinto",
}

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class LinguisticAnchor:
    """
    Something in the source that MUST be reflected in the translation.
    Numbers, amounts, and dates are the most common anchors — a translation
    that drops "$30" is linguistically wrong regardless of fluency.
    """
    source_str: str    # original text ("$30")
    numeric_val: str   # the digit string ("30") for verification
    anchor_type: str   # "CURRENCY" | "NUMBER" | "PERCENT" | "DATE"
    target_hint: str   # expected English form ("30 dollars")


@dataclass
class PrepResult:
    """Output of `normalize_es_input`."""
    normalized: str
    anchors: list[LinguisticAnchor] = field(default_factory=list)
    original: str = ""

    def coverage_failures(self, translation: str) -> list[LinguisticAnchor]:
        """Return anchors whose numeric value does not appear in the translation."""
        failures = []
        digits_in_output = set(re.findall(r"\d+", translation))
        for a in self.anchors:
            if a.numeric_val and a.numeric_val not in digits_in_output:
                failures.append(a)
        return failures

    def repair(self, translation: str) -> str:
        """
        If a numeric anchor is missing from the translation, append a note.
        This is a last-resort fallback — good preprocessing should prevent this.
        """
        failures = self.coverage_failures(translation)
        if not failures:
            return translation
        # Append missing values parenthetically so the user still sees them
        extras = ", ".join(f.target_hint for f in failures)
        return f"{translation} ({extras})"


# ── Core normalizer ────────────────────────────────────────────────────────────

# Currency pattern: $20, $1,000.50, 20€, 30$
# Commas only allowed as thousands separators (,\d+ not ,\s or ,end)
_RE_CURRENCY_PRE  = re.compile(r"([$€£¥])([\d](?:\d|,(?=\d))*(?:\.\d+)?)")
_RE_CURRENCY_POST = re.compile(r"([\d](?:\d|,(?=\d))*(?:\.\d+)?)\s*([$€£¥])")
# Percentage: 50%, 3.5%
_RE_PERCENT = re.compile(r"([\d]+(?:\.\d+)?)\s*%")
# Bare integers and decimals (for anchor extraction only, not substitution)
_RE_NUMBER = re.compile(r"\b(\d+(?:[,\.]\d+)*)\b")
# OOV non-alphanumeric non-punctuation (strip before BPE)
_RE_OOV_SYMBOL = re.compile(r"[^\w\s.,;:!?¿¡\-'\"()\[\]áéíóúüñÁÉÍÓÚÜÑ]")


def normalize_es_input(text: str) -> PrepResult:
    """
    Normalize a Spanish sentence for NMT input.

    Pipeline:
      1. Expand abbreviations (Sr. → señor)
      2. Normalize currency (  $20 → 20 dólares)
      3. Normalize percentages (50% → 50 por ciento)
      4. Record numeric anchors for post-translation verification
      5. Strip symbols that have no BPE token (# @ & ^ etc.)

    Returns a PrepResult with the normalized text and extracted anchors.
    """
    anchors: list[LinguisticAnchor] = []
    result = text.strip()

    # 1. Abbreviation expansion (order matters — longest match first)
    for abbrev, expansion in sorted(_ABBREV_ES.items(), key=lambda x: -len(x[0])):
        result = result.replace(abbrev, expansion)

    # Ordinals
    for ordinal, expansion in _ORDINAL_ES.items():
        result = result.replace(ordinal, expansion)

    # 2. Currency — pre-symbol: $20 → 20 dólares
    def _expand_pre(m: re.Match) -> str:
        sym, amt = m.group(1), m.group(2)
        unit_es = _CURRENCY_ES.get(sym, sym)
        unit_en = _CURRENCY_EN.get(sym, "")
        digits = re.sub(r"[,\.]", "", amt.split(".")[0])
        anchors.append(LinguisticAnchor(
            source_str=m.group(0), numeric_val=digits,
            anchor_type="CURRENCY",
            target_hint=f"{digits} {unit_en}" if unit_en else digits,
        ))
        return f"{amt} {unit_es}"

    result = _RE_CURRENCY_PRE.sub(_expand_pre, result)

    # Currency — post-symbol: 20$ → 20 dólares
    def _expand_post(m: re.Match) -> str:
        amt, sym = m.group(1), m.group(2)
        unit_es = _CURRENCY_ES.get(sym, sym)
        unit_en = _CURRENCY_EN.get(sym, "")
        digits = re.sub(r"[,\.]", "", amt.split(".")[0])
        anchors.append(LinguisticAnchor(
            source_str=m.group(0), numeric_val=digits,
            anchor_type="CURRENCY",
            target_hint=f"{digits} {unit_en}" if unit_en else digits,
        ))
        return f"{amt} {unit_es}"

    result = _RE_CURRENCY_POST.sub(_expand_post, result)

    # 3. Percentages: 50% → 50 por ciento
    def _expand_pct(m: re.Match) -> str:
        num = m.group(1)
        anchors.append(LinguisticAnchor(
            source_str=m.group(0), numeric_val=num.split(".")[0],
            anchor_type="PERCENT",
            target_hint=f"{num} percent",
        ))
        return f"{num} por ciento"

    result = _RE_PERCENT.sub(_expand_pct, result)

    # 4. Record any remaining bare numbers as anchors (for coverage verification)
    for m in _RE_NUMBER.finditer(result):
        digits = re.sub(r"[,\.]", "", m.group(0))
        if not any(a.numeric_val == digits for a in anchors):
            anchors.append(LinguisticAnchor(
                source_str=m.group(0), numeric_val=digits,
                anchor_type="NUMBER",
                target_hint=digits,
            ))

    # 5. Strip OOV symbols that would become UNK tokens and confuse the encoder
    result = _RE_OOV_SYMBOL.sub(" ", result)
    result = re.sub(r" {2,}", " ", result).strip()

    return PrepResult(normalized=result, anchors=anchors, original=text)


# ── Post-processing ────────────────────────────────────────────────────────────

def postprocess_en_output(text: str) -> str:
    """
    Clean up raw decoder output for display:
    - Capitalize first letter
    - Remove adjacent duplicate words (already done in server, kept here for pipeline use)
    - Trim whitespace
    """
    text = text.strip()
    if not text:
        return text
    # Deduplicate adjacent words
    words = text.split()
    deduped = [words[0]]
    for w in words[1:]:
        if w != deduped[-1]:
            deduped.append(w)
    text = " ".join(deduped)
    # Capitalize
    return text[0].upper() + text[1:] if text else text


def detect_translation_failure(
    source: str,
    translation: str,
    avg_entropy: float | None = None,
    src_tokens: list[str] | None = None,
) -> bool:
    """
    Flag a translation as likely failed using four independent signals:

    1. Length truncation — output < 30% source length (truncated output)
    2. Degeneration     — repeated characters (e.g. "aaaa")
    3. Attention entropy — avg > 0.50 means scattered cross-attention,
                          i.e. the model couldn't find clear source-target
                          alignments — strong signal of OOV / domain mismatch
    4. BPE fragmentation — if >35% of BPE source tokens are single characters,
                          the input contains words the vocab never saw, and
                          the model is pattern-matching on individual letters,
                          which produces nonsense
    """
    src_words = len(source.split())
    out_words = len(translation.split())

    # 1. Truncation
    if src_words > 3 and out_words < max(2, src_words * 0.3):
        return True

    # 2. Degeneration
    if re.search(r"(.)\1{3,}", translation):
        return True

    # 3. Attention entropy gate
    if avg_entropy is not None and src_words >= 5 and avg_entropy > 0.50:
        return True

    # 4. BPE fragmentation — single-char tokens mean OOV words being split
    if src_tokens is not None and len(src_tokens) > 4:
        single_chars = sum(
            1 for t in src_tokens
            if len(t.replace("</w>", "")) == 1 and t not in ("<unk>", "<pad>")
        )
        if single_chars / len(src_tokens) > 0.35:
            return True

    return False
