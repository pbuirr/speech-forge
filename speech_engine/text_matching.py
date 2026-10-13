"""Conservative learning-answer matching, separate from model tokenization.

Canonical Unicode equivalence is accepted. Accents, joiners, compatibility
characters, and spelling distinctions are preserved. This is not a semantic
equivalence test or a complete locale-sensitive casing implementation.
"""
import unicodedata

from speech_engine.linguistic_resources import language_code


def answer_key(text: str, lang: str = 'en', *, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize('NFC', text.strip())
    if not case_sensitive:
        if language_code(lang) in {'tr', 'az'}:
            text = text.replace('I', 'ı').replace('İ', 'i')
        # Lowercase, rather than compatibility normalization or full casefold:
        # e.g. don't automatically accept 'ss' as the spelling of German 'ß'.
        text = text.lower()
    return unicodedata.normalize('NFC', text)
