"""Protects names, numbers and URLs during translation.

Entities are replaced with placeholders (NE_0, NE_1, ...) before translation
and restored afterwards:

    "Tesla verkauft Autos in Berlin" → "NE_0 verkauft Autos in NE_1"
    → "NE_0 sells cars in NE_1" → "Tesla sells cars in Berlin"

Rules, in priority order: URLs/emails/paths, acronyms, tokens with digits,
@handles and #hashtags, and capitalised words not at sentence start. For
German source, common capitalised nouns are excluded from the last rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Patterns

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HANDLE_RE = re.compile(r"[@#]\w+")
_DIGIT_RE = re.compile(r"\S*\d\S*")  # any token containing a digit
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9\-]{1,}$")  # ≥2 uppercase (acronym)

# High-frequency German common nouns that should NOT be protected as NEs.
# Without this, NER would protect "Ich", "Das", "Die" etc. in German text.
_DE_COMMON_CAPS = frozenset(
    {
        "Ich",
        "Er",
        "Sie",
        "Es",
        "Wir",
        "Ihr",
        "Der",
        "Die",
        "Das",
        "Den",
        "Dem",
        "Des",
        "Ein",
        "Eine",
        "Einen",
        "Einem",
        "Einer",
        "Eines",
        "Kein",
        "Keine",
        "Nicht",
        "Und",
        "Oder",
        "Aber",
        "Auch",
        "Noch",
        "Sehr",
        "Dann",
        "Wenn",
        "Weil",
        "Dass",
        "Mit",
        "Von",
        "Aus",
        "Bei",
        "Auf",
        "Für",
        "Durch",
        "Nach",
        "Über",
        "Unter",
        "Vor",
        "Hinter",
        "Kann",
        "Muss",
        "Soll",
        "Will",
        "Mag",
        "Darf",
        "Haben",
        "Sein",
        "Werden",
        "Werden",
        "Gehen",
        "Kommen",
        "Sehen",
        "Wissen",
    }
)


@dataclass
class EntityProtector:
    """
    Masks named entities with placeholders before translation,
    then restores originals in the output.

    Usage::

        ep = EntityProtector(src_lang="de")
        masked, entities = ep.mask("Tesla verkauft Autos in Berlin")
        # masked   = "NE_0 verkauft Autos in NE_1"
        # entities = ["Tesla", "Berlin"]

        translated = model.translate(masked)
        # translated = "NE_0 sells cars in NE_1"

        result = ep.restore(translated, entities)
        # result = "Tesla sells cars in Berlin"
    """

    src_lang: str = "?"
    placeholder_fmt: str = "NE_{i}"

    def _is_entity(self, token: str, position: int, is_sentence_start: bool) -> bool:
        """Return True if this token should be protected as a named entity."""
        # Rule 1: URL or email
        if _URL_RE.match(token) or _EMAIL_RE.match(token):
            return True

        # Rule 2: @handle or #hashtag
        if _HANDLE_RE.match(token):
            return True

        # Rule 3: contains a digit (years, phone numbers, product codes)
        if _DIGIT_RE.match(token) and not token.isdigit():
            # bare digits like "2024" are kept if >4 chars (product codes, IDs)
            # but simple "2" or "10" are not protected
            return len(token) > 4 or any(c.isalpha() for c in token)
        if token.isdigit() and len(token) >= 4:
            # 4-digit numbers are years / codes — protect them
            return True

        # Rule 4: all-caps acronym (≥ 2 uppercase letters)
        if _ALLCAPS_RE.match(token):
            return True

        # Rule 5: capitalized token not at sentence start
        if not is_sentence_start and token[0].isupper() and token.isalpha():
            # For German: skip common capitalized common nouns
            if self.src_lang == "de" and token in _DE_COMMON_CAPS:
                return False
            return True

        return False

    def mask(self, text: str) -> tuple[str, list[str]]:
        """
        Replace named entities with placeholders.

        Returns:
            (masked_text, entity_list)
            entity_list[i] is the original string for placeholder NE_i
        """
        tokens = text.split()
        entities: list[str] = []
        masked_tokens: list[str] = []
        placeholder_to_entity: dict[str, str] = {}

        for i, tok in enumerate(tokens):
            is_start = i == 0
            # Strip trailing punctuation for the check but keep it in output
            clean = tok.rstrip(".,;:!?\"'")
            suffix = tok[len(clean) :]

            if clean and self._is_entity(clean, position=i, is_sentence_start=is_start):
                # Reuse placeholder if same entity seen before
                existing = placeholder_to_entity.get(clean)
                if existing is not None:
                    masked_tokens.append(existing + suffix)
                else:
                    ph = self.placeholder_fmt.format(i=len(entities))
                    entities.append(clean)
                    placeholder_to_entity[clean] = ph
                    masked_tokens.append(ph + suffix)
            else:
                masked_tokens.append(tok)

        return " ".join(masked_tokens), entities

    def restore(self, translated: str, entities: list[str]) -> str:
        """
        Replace NE_i placeholders in the translated text with original entities.

        Handles partial restorations gracefully: if the NMT dropped a
        placeholder (e.g. it translated "NE_0" as "the NE_0 company" or
        omitted it), the remaining placeholders are still restored.
        """
        result = translated
        for i, entity in enumerate(entities):
            ph = self.placeholder_fmt.format(i=i)
            result = result.replace(ph, entity)
        return result


# Convenience function


def protect_and_translate(
    text: str,
    translate_fn,
    src_lang: str = "?",
) -> str:
    """
    Full protect → translate → restore pipeline.

    Args:
        text:         source sentence
        translate_fn: callable(str) → str — the translation function
        src_lang:     source language code (for language-specific rules)

    Returns:
        translated string with named entities restored

    Example::

        result = protect_and_translate(
            "Elon Musk founded Tesla in 2003",
            translate_fn=lambda s: model.translate(s),
            src_lang="en",
        )
    """
    ep = EntityProtector(src_lang=src_lang)
    masked, entities = ep.mask(text)
    translated = translate_fn(masked)
    return ep.restore(translated, entities)
