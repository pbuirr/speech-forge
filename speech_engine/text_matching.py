"""Answer matching for exercises.

Accepts canonically equivalent Unicode but keeps accents and spelling
differences significant.
"""

import unicodedata

from speech_engine.linguistic_resources import language_code


def answer_key(text: str, lang: str = "en", *, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    if not case_sensitive:
        if language_code(lang) in {"tr", "az"}:
            text = text.replace("I", "ı").replace("İ", "i")
        # Lowercase, rather than compatibility normalization or full casefold:
        # e.g. don't automatically accept 'ss' as the spelling of German 'ß'.
        text = text.lower()
    return unicodedata.normalize("NFC", text)
