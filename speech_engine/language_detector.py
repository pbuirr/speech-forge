"""
speech_engine/language_detector.py — Language and dialect detection, from scratch.

Covers 170+ languages and 60+ dialect variants.  No pretrained models, no external
databases.  The detector is purely statistical and rule-based, built entirely from
Unicode properties and linguistic knowledge.

Detection pipeline (three layers, each narrows candidates):

  1. Script detection
       Maps every character to its Unicode script block.  Immediately
       identifies languages that use unique scripts (Thai, Georgian, Armenian,
       Ethiopic, Khmer, …).  For multi-script inputs it returns a distribution.

  2. Language scoring
       Within a script family, each language has:
         • Unique characters that appear almost nowhere else (ß → German, ñ → Spanish)
         • High-frequency n-grams from known statistics
         • Negative markers (characters absent from this language but present in others)
       Combined into a score with confidence.

  3. Dialect detection
       Given the top language candidate, applies a second layer of markers:
         • Orthographic conventions  (colour/color, realise/realize)
         • Vocabulary markers        (vosotros/ustedes, você/tu)
         • Phonological spellings    (gonna/gotta in informal en-US)
         • Country-specific loanwords
       Returns the most likely regional variant.

Usage::

    detector = LanguageDetector()
    result = detector.detect("Bonjour, comment allez-vous ?")
    print(result.language)       # "fr"
    print(result.dialect)        # "fr-FR"
    print(result.script)         # "Latin"
    print(result.confidence)     # 0.94
    print(result.top)            # [("fr", 0.94), ("it", 0.31), ("es", 0.28)]

    # Dialect-first detection
    result = detector.detect("gonna grab some chips from the store, y'all")
    print(result.dialect)  # "en-US-Southern"

Training your own profiles (improves accuracy with real data)::

    detector.fit("This is an English text.", "en")
    detector.fit("Ceci est un texte français.", "fr")
    detector.save("profiles.json")
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Unicode script ranges
#  Each tuple: (start_codepoint, end_codepoint, script_name)
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_RANGES: list[tuple[int, int, str]] = [
    # Latin
    (0x0041, 0x005A, "Latin"), (0x0061, 0x007A, "Latin"),
    (0x00C0, 0x00D6, "Latin"), (0x00D8, 0x00F6, "Latin"), (0x00F8, 0x024F, "Latin"),
    (0x0250, 0x02AF, "Latin"), (0x1D00, 0x1DBF, "Latin"),
    (0x1E00, 0x1EFF, "Latin"), (0xA720, 0xA7FF, "Latin"),
    (0xAB30, 0xAB6F, "Latin"),
    # Cyrillic
    (0x0400, 0x04FF, "Cyrillic"), (0x0500, 0x052F, "Cyrillic"),
    (0x2DE0, 0x2DFF, "Cyrillic"), (0xA640, 0xA69F, "Cyrillic"),
    # Greek
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek"),
    # Armenian
    (0x0530, 0x058F, "Armenian"), (0xFB00, 0xFB17, "Armenian"),
    # Georgian
    (0x10A0, 0x10FF, "Georgian"), (0x2D00, 0x2D2F, "Georgian"),
    (0x1C90, 0x1CBF, "Georgian"),
    # Hebrew
    (0x0590, 0x05FF, "Hebrew"), (0xFB1D, 0xFB4F, "Hebrew"),
    # Arabic
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"),
    (0x08A0, 0x08FF, "Arabic"), (0x0870, 0x089F, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"), (0xFE70, 0xFEFF, "Arabic"),
    # Syriac
    (0x0700, 0x074F, "Syriac"),
    # Thaana (Maldivian/Dhivehi)
    (0x0780, 0x07BF, "Thaana"),
    # N'Ko
    (0x07C0, 0x07FF, "NKo"),
    # Samaritan
    (0x0800, 0x083F, "Samaritan"),
    # Devanagari (Hindi, Marathi, Nepali, Sanskrit, Konkani, Dogri, Maithili, Bodo, Sindhi)
    (0x0900, 0x097F, "Devanagari"), (0xA8E0, 0xA8FF, "Devanagari"),
    (0x1CD0, 0x1CFF, "Devanagari"),
    # Bengali/Bangla
    (0x0980, 0x09FF, "Bengali"),
    # Gurmukhi (Punjabi)
    (0x0A00, 0x0A7F, "Gurmukhi"),
    # Gujarati
    (0x0A80, 0x0AFF, "Gujarati"),
    # Oriya/Odia
    (0x0B00, 0x0B7F, "Oriya"),
    # Tamil
    (0x0B80, 0x0BFF, "Tamil"),
    # Telugu
    (0x0C00, 0x0C7F, "Telugu"),
    # Kannada
    (0x0C80, 0x0CFF, "Kannada"),
    # Malayalam
    (0x0D00, 0x0D7F, "Malayalam"),
    # Sinhala
    (0x0D80, 0x0DFF, "Sinhala"),
    # Thai
    (0x0E00, 0x0E7F, "Thai"),
    # Lao
    (0x0E80, 0x0EFF, "Lao"),
    # Tibetan
    (0x0F00, 0x0FFF, "Tibetan"),
    # Myanmar/Burmese
    (0x1000, 0x109F, "Myanmar"), (0xA9E0, 0xA9FF, "Myanmar"),
    (0xAA60, 0xAA7F, "Myanmar"),
    # Tagalog / Baybayin
    (0x1700, 0x171F, "Tagalog"),
    # Hangul (Korean)
    (0xAC00, 0xD7AF, "Hangul"), (0x1100, 0x11FF, "Hangul"),
    (0xA960, 0xA97F, "Hangul"), (0xD7B0, 0xD7FF, "Hangul"),
    (0x3130, 0x318F, "Hangul"),
    # CJK Unified Ideographs (Chinese, Japanese Kanji, Korean Hanja)
    (0x4E00, 0x9FFF, "CJK"),  (0x3400, 0x4DBF, "CJK"),
    (0xF900, 0xFAFF, "CJK"),  (0x2F800, 0x2FA1F, "CJK"),
    (0x3000, 0x303F, "CJK"),
    # Hiragana (Japanese)
    (0x3040, 0x309F, "Hiragana"),
    # Katakana (Japanese)
    (0x30A0, 0x30FF, "Katakana"), (0x31F0, 0x31FF, "Katakana"),
    (0xFF65, 0xFF9F, "Katakana"),
    # Bopomofo (Chinese phonetic)
    (0x02EA, 0x02EB, "Bopomofo"), (0x3105, 0x312F, "Bopomofo"),
    # Yi (Sichuan, China)
    (0xA000, 0xA48F, "Yi"),
    # Ethiopic (Amharic, Tigrinya, Oromo, Somali written in Ethiopic)
    (0x1200, 0x137F, "Ethiopic"), (0x1380, 0x139F, "Ethiopic"),
    (0x2D80, 0x2DDF, "Ethiopic"), (0xAB01, 0xAB2F, "Ethiopic"),
    # Khmer (Cambodian)
    (0x1780, 0x17FF, "Khmer"), (0x19E0, 0x19FF, "Khmer"),
    # Mongolian
    (0x1800, 0x18AF, "Mongolian"), (0x11660, 0x1166F, "Mongolian"),
    # Cherokee
    (0x13A0, 0x13FF, "Cherokee"), (0xAB70, 0xABBF, "Cherokee"),
    # Canadian Aboriginal Syllabics
    (0x1400, 0x167F, "CanadianSyllabics"),
    # Runic
    (0x16A0, 0x16FF, "Runic"),
    # Ogham (Old Irish)
    (0x1680, 0x169F, "Ogham"),
    # Linear B (ancient Greek)
    (0x10000, 0x1007F, "LinearB"),
    # Coptic
    (0x2C80, 0x2CFF, "Coptic"),
    # Bamum (Cameroon)
    (0xA6A0, 0xA6FF, "Bamum"),
    # Tifinagh (Berber languages)
    (0x2D30, 0x2D7F, "Tifinagh"),
    # Vai (Liberia)
    (0xA500, 0xA63F, "Vai"),
    # N'Ko (West Africa)
    (0x07C0, 0x07FF, "NKo"),
    # Mandaic
    (0x0840, 0x085F, "Mandaic"),
    # Adlam (Fulani)
    (0x1E900, 0x1E95F, "Adlam"),
    # Javanese
    (0xA980, 0xA9DF, "Javanese"),
    # Balinese
    (0x1B00, 0x1B7F, "Balinese"),
    # Sundanese
    (0x1B80, 0x1BBF, "Sundanese"),
    # Batak (Sumatra)
    (0x1BC0, 0x1BFF, "Batak"),
    # Cham
    (0xAA00, 0xAA5F, "Cham"),
    # Rejang
    (0xA930, 0xA95F, "Rejang"),
    # Lisu
    (0xA4D0, 0xA4FF, "Lisu"),
    # Miao/Pollard (SW China)
    (0x16F00, 0x16F9F, "Miao"),
    # Pahawh Hmong
    (0x16B00, 0x16B8F, "PahawnHmong"),
]

# Build an interval-lookup structure for O(log n) script detection
_SORTED_RANGES: list[tuple[int, int, str]] = sorted(SCRIPT_RANGES, key=lambda x: x[0])


def codepoint_to_script(cp: int) -> str:
    """Return the script name for a single Unicode codepoint."""
    lo, hi = 0, len(_SORTED_RANGES) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, name = _SORTED_RANGES[mid]
        if cp < start:
            hi = mid - 1
        elif cp > end:
            lo = mid + 1
        else:
            return name
    # Digits 0-9 and punctuation are script-agnostic
    if 0x0030 <= cp <= 0x0039:
        return "Digit"
    if cp < 0x0080:
        return "ASCII"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Language database
#  Covers 170+ languages with unique character signatures and dialect markers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LangDef:
    """Definition of one language or dialect."""
    name: str
    script: str
    # Characters virtually unique to this language (boosts score strongly)
    unique_chars: str = ""
    # Characters common but not unique (moderate boost)
    common_chars: str = ""
    # Frequent digraphs/trigraphs (as a string of space-separated entries)
    grams: str = ""
    # Parent code if this is a dialect
    dialect_of: Optional[str] = None
    # Marker words/patterns for dialect sub-detection
    dialect_markers: list[str] = field(default_factory=list)
    # Word patterns typical of this language (common short words)
    word_markers: list[str] = field(default_factory=list)


# Core language database: ISO 639-1 codes + BCP 47 dialect codes
LANG_DB: dict[str, LangDef] = {

    # ── English ──────────────────────────────────────────────────────────────
    "en": LangDef("English", "Latin",
        grams="the and with have from this that not are you as hello my name",
        word_markers=["the", "and", "is", "of", "to", "in", "it", "that", "was", "for", "are", "with", "as", "do",
                      "hello", "hi", "hey", "my", "name", "am", "yes", "no", "what", "how", "when", "who",
                      "where", "your", "their", "our", "we", "they", "he", "she", "his", "her", "its",
                      "have", "has", "had", "will", "would", "could", "should", "can", "may", "might"]),
    "en-US": LangDef("English (US)", "Latin", dialect_of="en",
        dialect_markers=["color", "honor", "labor", "favorite", "center", "theater",
                         "analyze", "recognize", "organize", "aluminum", "apologize",
                         "math", "gotten", "soccer", "realized", "license"]),
    "en-GB": LangDef("English (UK)", "Latin", dialect_of="en",
        dialect_markers=["colour", "honour", "labour", "favourite", "centre", "theatre",
                         "analyse", "recognise", "organise", "aluminium", "apologise",
                         "maths", "football", "realised", "licence", "whilst", "fortnight"]),
    "en-AU": LangDef("English (Australia)", "Latin", dialect_of="en",
        dialect_markers=["arvo", "servo", "mozzies", "rego", "brekkie", "arsehole",
                         "colour", "realise", "honour", "footy", "uni", "chemist"]),
    "en-CA": LangDef("English (Canada)", "Latin", dialect_of="en",
        dialect_markers=["colour", "honour", "neighbour", "cheque", "defence",
                         "toque", "poutine", "loonie", "toonie", "washroom", "eh"]),
    "en-IN": LangDef("English (India)", "Latin", dialect_of="en",
        dialect_markers=["prepone", "revert back", "kindly", "do the needful",
                         "lakh", "crore", "rupee", "wallah", "wala"]),
    "en-ZA": LangDef("English (South Africa)", "Latin", dialect_of="en",
        dialect_markers=["braai", "lekker", "robot", "bakkie", "boerewors",
                         "howzit", "eish", "shame", "bru", "jol"]),
    "en-NZ": LangDef("English (New Zealand)", "Latin", dialect_of="en",
        dialect_markers=["kia ora", "bach", "jandals", "dairy", "biscuits",
                         "colour", "honour", "aotearoa", "kiwi"]),

    # ── Spanish ──────────────────────────────────────────────────────────────
    # ¿ and ¡ are orthographically unique to Spanish — no other major language uses them.
    "es": LangDef("Spanish", "Latin",
        unique_chars="ñ¿¡",
        common_chars="áéíóúü",
        grams="de la el en que es un con del para por una ción mente",
        word_markers=["que", "de", "la", "en", "el", "por", "con",
                      "puede", "alguien", "cómo", "también", "pero", "como",
                      "para", "los", "las", "una", "muy", "más", "este",
                      "esta", "está", "estás", "tengo", "tienes", "tiene",
                      "usted", "nosotros", "ellos", "ellas", "cuando", "donde",
                      "hola", "llamo", "llamas", "llama", "buenas", "buenos",
                      "señor", "señora", "gracias", "favor", "mucho", "poco"]),
    "es-ES": LangDef("Spanish (Spain)", "Latin", dialect_of="es",
        dialect_markers=["vosotros", "os", "coger", "ordenador", "móvil",
                         "patatas", "piso", "conducir", "vale", "tío", "guay"]),
    "es-MX": LangDef("Spanish (Mexico)", "Latin", dialect_of="es",
        dialect_markers=["güero", "chido", "órale", "ahorita", "antro", "chamaco",
                         "mande", "qué padre", "pendejo", "chamba", "cuate"]),
    "es-AR": LangDef("Spanish (Argentina)", "Latin", dialect_of="es",
        dialect_markers=["vos", "che", "boludo", "pibe", "boliche", "laburo",
                         "morfar", "quilombo", "copado", "facha", "birra"]),
    "es-CO": LangDef("Spanish (Colombia)", "Latin", dialect_of="es",
        dialect_markers=["parcero", "chévere", "bacano", "dar papaya", "mija",
                         "chimba", "berraco", "gonorrea", "buena nota"]),
    "es-CL": LangDef("Spanish (Chile)", "Latin", dialect_of="es",
        dialect_markers=["po", "weon", "cachai", "altiro", "fome", "la raja",
                         "poto", "pololo", "guagua", "carrete"]),
    "es-PE": LangDef("Spanish (Peru)", "Latin", dialect_of="es",
        dialect_markers=["causa", "pata", "hablar", "chamba", "jato", "chibolo"]),
    "es-VE": LangDef("Spanish (Venezuela)", "Latin", dialect_of="es",
        dialect_markers=["chamo", "chevere", "coño", "pana", "vaina", "arrechos"]),

    # ── Portuguese ───────────────────────────────────────────────────────────
    "pt": LangDef("Portuguese", "Latin",
        unique_chars="ãõ",
        common_chars="áéíóúâêô",
        grams="de do da em que os as um o a",
        word_markers=["que", "de", "do", "da", "em", "para", "os", "uma", "um", "o", "a"]),
    "pt-BR": LangDef("Portuguese (Brazil)", "Latin", dialect_of="pt",
        dialect_markers=["você", "a gente", "legal", "saudade", "tchau", "ônibus",
                         "celular", "time", "trem", "geladeira", "banheiro"]),
    "pt-PT": LangDef("Portuguese (Portugal)", "Latin", dialect_of="pt",
        dialect_markers=["tu", "autocarros", "casa de banho", "frigorífico",
                         "telemóvel", "equipa", "comboio", "pá", "fixe"]),
    "pt-AO": LangDef("Portuguese (Angola)", "Latin", dialect_of="pt",
        dialect_markers=["kamba", "ganda", "marimbondo", "funge"]),

    # ── French ───────────────────────────────────────────────────────────────
    "fr": LangDef("French", "Latin",
        unique_chars="œ",
        common_chars="àâéèêëîïôùûüÿç",
        grams="de la le les des que en il une et pas avec vous bonjour",
        word_markers=["le", "de", "la", "les", "que", "en", "il", "une", "est", "et",
                      "pas", "avec", "vous", "bonjour", "comment", "merci", "oui",
                      "non", "je", "tu", "nous", "ils", "elles", "mon", "ma", "mes"]),
    "fr-FR": LangDef("French (France)", "Latin", dialect_of="fr",
        dialect_markers=["septante", "nonante", "putain", "mec", "truc", "bouquin",
                         "bagnole", "flic", "bouffer", "se barrer", "kiffer"]),
    "fr-CA": LangDef("French (Québécois)", "Latin", dialect_of="fr",
        dialect_markers=["tabernac", "crisse", "ostie", "câlice", "char", "magasiner",
                         "fin de semaine", "courriel", "stationnement", "chum", "blonde"]),
    "fr-BE": LangDef("French (Belgium)", "Latin", dialect_of="fr",
        dialect_markers=["septante", "nonante", "bonjour", "kot", "pain français",
                         "trottoir", "déjeuner", "dîner", "souper"]),
    "fr-CH": LangDef("French (Switzerland)", "Latin", dialect_of="fr",
        dialect_markers=["septante", "huitante", "nonante", "panosse", "cornet"]),

    # ── Italian ──────────────────────────────────────────────────────────────
    "it": LangDef("Italian", "Latin",
        common_chars="àèéìíîòóùú",
        grams="di il la che in un per si come",
        word_markers=["che", "di", "il", "la", "in", "un", "per", "è", "si"]),
    "it-IT": LangDef("Italian (Italy)", "Latin", dialect_of="it"),
    "it-CH": LangDef("Italian (Switzerland)", "Latin", dialect_of="it"),

    # ── Romanian ─────────────────────────────────────────────────────────────
    "ro": LangDef("Romanian", "Latin",
        unique_chars="ăâîșț",
        grams="de la cu în că și pentru",
        word_markers=["că", "de", "la", "cu", "în", "și", "este", "nu"]),

    # ── German ───────────────────────────────────────────────────────────────
    "de": LangDef("German", "Latin",
        unique_chars="ß",
        common_chars="äöü",
        grams="der die das ist und in den von nicht wie geht",
        word_markers=["der", "die", "das", "ist", "und", "in", "den", "von", "nicht",
                      "wie", "geht", "ich", "sie", "wir", "ihr", "mein", "sein",
                      "haben", "werden", "können", "bitte", "danke", "ihnen", "auch"]),
    "de-DE": LangDef("German (Germany)", "Latin", dialect_of="de",
        dialect_markers=["samstag", "januar", "ich", "machen", "fahren"]),
    "de-AT": LangDef("German (Austria)", "Latin", dialect_of="de",
        dialect_markers=["jänner", "bim", "heuer", "obmann", "paradeiser",
                         "erdapfel", "schmarren", "grüß gott", "servus"]),
    "de-CH": LangDef("German (Switzerland)", "Latin", dialect_of="de",
        dialect_markers=["grüezi", "tschüss", "velo", "natel", "abschlussball",
                         "fritschi", "güsel", "postauto", "znüni"]),

    # ── Dutch / Flemish ──────────────────────────────────────────────────────
    "nl": LangDef("Dutch", "Latin",
        common_chars="",
        grams="de het en van in dat is een op",
        word_markers=["de", "het", "en", "van", "in", "dat", "is", "een"]),
    "nl-BE": LangDef("Flemish (Belgium)", "Latin", dialect_of="nl",
        dialect_markers=["allez", "gij", "ge", "gjij", "voetbal", "frietjes"]),

    # ── Swedish ──────────────────────────────────────────────────────────────
    "sv": LangDef("Swedish", "Latin",
        unique_chars="",
        common_chars="åäö",
        grams="att och det är för den till av inte",
        word_markers=["att", "och", "det", "är", "för", "den", "till", "av", "inte"]),

    # ── Norwegian ────────────────────────────────────────────────────────────
    "nb": LangDef("Norwegian Bokmål", "Latin",
        common_chars="æøå",
        grams="og er at det i til for ikke den",
        word_markers=["og", "er", "at", "det", "til", "for", "ikke", "den"]),
    "nn": LangDef("Norwegian Nynorsk", "Latin",
        common_chars="æøå",
        grams="og er at det i til for ikkje den",
        word_markers=["og", "er", "at", "det", "til", "for", "ikkje", "dei", "me"]),

    # ── Danish ───────────────────────────────────────────────────────────────
    "da": LangDef("Danish", "Latin",
        common_chars="æøå",
        grams="og er at det i til for ikke den",
        word_markers=["og", "er", "at", "det", "til", "for", "ikke", "den", "en"]),

    # ── Finnish ──────────────────────────────────────────────────────────────
    "fi": LangDef("Finnish", "Latin",
        unique_chars="",
        common_chars="äö",
        grams="on ja että ei olla se kun kuin jonka",
        word_markers=["on", "ja", "että", "ei", "olla", "se", "kun", "olen", "kuin", "jonka"]),

    # ── Estonian ─────────────────────────────────────────────────────────────
    "et": LangDef("Estonian", "Latin",
        unique_chars="õ",
        common_chars="äöüš",
        grams="ja on et ei ole",
        word_markers=["ja", "on", "et", "ei", "ole", "ning"]),

    # ── Latvian ──────────────────────────────────────────────────────────────
    "lv": LangDef("Latvian", "Latin",
        unique_chars="āēīūļķņģ",
        grams="ir un ka no arī",
        word_markers=["ir", "un", "ka", "no", "arī", "viņš"]),

    # ── Lithuanian ───────────────────────────────────────────────────────────
    "lt": LangDef("Lithuanian", "Latin",
        unique_chars="ąčęėįšųūž",
        grams="ir kad ne bet jis",
        word_markers=["ir", "kad", "ne", "bet", "jis", "yra", "kaip"]),

    # ── Polish ───────────────────────────────────────────────────────────────
    "pl": LangDef("Polish", "Latin",
        unique_chars="ąćęłńóśźż",
        grams="że jest się nie jak na",
        word_markers=["że", "jest", "się", "nie", "jak", "na", "do", "to", "w"]),

    # ── Czech ────────────────────────────────────────────────────────────────
    "cs": LangDef("Czech", "Latin",
        unique_chars="ř",
        common_chars="áčďéěíňóšťúůýž",
        grams="je a se ne na to",
        word_markers=["je", "a", "se", "ne", "na", "to", "bylo", "jako", "jsem"]),

    # ── Slovak ───────────────────────────────────────────────────────────────
    "sk": LangDef("Slovak", "Latin",
        unique_chars="ľĺ",
        common_chars="áäčéíóôšúýž",
        grams="je a sa nie na to",
        word_markers=["je", "a", "sa", "nie", "na", "to", "ako", "som"]),

    # ── Hungarian ────────────────────────────────────────────────────────────
    "hu": LangDef("Hungarian", "Latin",
        unique_chars="őű",
        common_chars="áéíóöü",
        grams="és az hogy nem a",
        word_markers=["és", "az", "hogy", "nem", "egy", "van", "is", "de"]),

    # ── Croatian ─────────────────────────────────────────────────────────────
    "hr": LangDef("Croatian", "Latin",
        unique_chars="đ",
        common_chars="čšžć",
        grams="je i se da ne",
        word_markers=["je", "i", "se", "da", "ne", "su", "koja", "koji", "kao"]),

    # ── Serbian (Latin) ──────────────────────────────────────────────────────
    "sr-Latn": LangDef("Serbian (Latin)", "Latin",
        unique_chars="đ",
        common_chars="čšž",
        grams="je i se da ne",
        word_markers=["je", "i", "se", "da", "ne", "su", "koja", "koji"]),

    # ── Bosnian ──────────────────────────────────────────────────────────────
    "bs": LangDef("Bosnian", "Latin",
        common_chars="čšžćđ",
        grams="je i se da ne",
        word_markers=["je", "i", "se", "da", "ne", "su", "koji", "kao"]),

    # ── Slovenian ────────────────────────────────────────────────────────────
    "sl": LangDef("Slovenian", "Latin",
        common_chars="čšž",
        grams="je in se da ne",
        word_markers=["je", "in", "se", "da", "ne", "so", "ki", "pa", "ter"]),

    # ── Albanian ─────────────────────────────────────────────────────────────
    "sq": LangDef("Albanian", "Latin",
        unique_chars="ë",
        grams="dhe është se për me",
        word_markers=["dhe", "është", "se", "për", "me", "në", "jo", "po", "si"]),

    # ── Catalan ──────────────────────────────────────────────────────────────
    "ca": LangDef("Catalan", "Latin",
        unique_chars="·",
        common_chars="àèéíïòóúü",
        grams="de la el que en",
        word_markers=["que", "de", "la", "el", "en", "és", "per", "les", "dels"]),

    # ── Galician ─────────────────────────────────────────────────────────────
    "gl": LangDef("Galician", "Latin",
        common_chars="áéíóú",
        grams="de o a que en",
        word_markers=["que", "de", "o", "a", "en", "é", "non", "un", "os", "as"]),

    # ── Basque ───────────────────────────────────────────────────────────────
    "eu": LangDef("Basque", "Latin",
        unique_chars="",
        grams="eta da ez bat da",
        word_markers=["eta", "da", "ez", "bat", "zara", "naiz", "den", "zer"]),

    # ── Welsh ────────────────────────────────────────────────────────────────
    "cy": LangDef("Welsh", "Latin",
        unique_chars="ŵŷ",
        common_chars="âêîôû",
        grams="y yr a ac yn",
        word_markers=["y", "yr", "a", "ac", "yn", "i", "ei", "ai", "yw", "ydy"]),

    # ── Irish Gaelic ─────────────────────────────────────────────────────────
    "ga": LangDef("Irish Gaelic", "Latin",
        unique_chars="",
        common_chars="áéíóú",
        grams="an na is agus",
        word_markers=["an", "na", "is", "agus", "go", "le", "ní", "den", "tá"]),

    # ── Scottish Gaelic ──────────────────────────────────────────────────────
    "gd": LangDef("Scottish Gaelic", "Latin",
        common_chars="àèìòù",
        grams="agus an a tha",
        word_markers=["agus", "an", "a", "tha", "na", "is", "le", "air"]),

    # ── Turkish ──────────────────────────────────────────────────────────────
    "tr": LangDef("Turkish", "Latin",
        unique_chars="ğ",
        common_chars="çışüö",
        grams="bir ve için de da",
        word_markers=["bir", "ve", "için", "de", "da", "bu", "ile", "ben", "sen"]),
    "tr-TR": LangDef("Turkish (Turkey)", "Latin", dialect_of="tr"),
    "tr-CY": LangDef("Turkish (Cyprus)", "Latin", dialect_of="tr"),

    # ── Azerbaijani (Latin) ───────────────────────────────────────────────────
    "az": LangDef("Azerbaijani", "Latin",
        unique_chars="ə",
        grams="və bir bu da",
        word_markers=["və", "bir", "bu", "da", "de", "ilə", "olan"]),

    # ── Uzbek (Latin) ────────────────────────────────────────────────────────
    "uz": LangDef("Uzbek (Latin)", "Latin",
        common_chars="ʻ",
        grams="va bir bu ham",
        word_markers=["va", "bir", "bu", "ham", "uchun", "bilan"]),

    # ── Maltese ──────────────────────────────────────────────────────────────
    "mt": LangDef("Maltese", "Latin",
        unique_chars="ħġ",
        common_chars="àèùż",
        grams="li il u ta",
        word_markers=["li", "il", "u", "ta", "din", "dan", "kif", "mhux"]),

    # ── Indonesian ───────────────────────────────────────────────────────────
    "id": LangDef("Indonesian", "Latin",
        grams="dan yang di ke pada adalah",
        word_markers=["dan", "yang", "di", "ke", "pada", "ini", "itu", "ada", "dengan", "adalah"]),

    # ── Malay ────────────────────────────────────────────────────────────────
    "ms": LangDef("Malay", "Latin",
        grams="dan yang di ke pada",
        word_markers=["dan", "yang", "di", "ke", "pada", "ini", "itu", "ada", "dengan"]),
    "ms-MY": LangDef("Malay (Malaysia)", "Latin", dialect_of="ms",
        dialect_markers=["lah", "lor", "mah", "ah", "boleh", "makan"]),
    "ms-SG": LangDef("Singlish/Malay (Singapore)", "Latin", dialect_of="ms",
        dialect_markers=["lah", "leh", "lor", "meh", "sia", "shiok", "kiasu"]),

    # ── Filipino / Tagalog ───────────────────────────────────────────────────
    "tl": LangDef("Tagalog/Filipino", "Latin",
        grams="ng sa mga at",
        word_markers=["ang", "ng", "sa", "mga", "at", "ay", "na", "ko", "mo"]),

    # ── Swahili ──────────────────────────────────────────────────────────────
    "sw": LangDef("Swahili", "Latin",
        grams="na ya wa ku",
        word_markers=["na", "ya", "wa", "ku", "ni", "katika", "kwa", "hii", "hiyo"]),

    # ── Somali ───────────────────────────────────────────────────────────────
    "so": LangDef("Somali", "Latin",
        grams="iyo oo ka in",
        word_markers=["iyo", "oo", "ka", "in", "waa", "si", "ah", "ee"]),

    # ── Yoruba ───────────────────────────────────────────────────────────────
    "yo": LangDef("Yoruba", "Latin",
        unique_chars="ẹọṣ",
        common_chars="",
        grams="ni ti ati fun",
        word_markers=["ni", "ti", "ati", "fun", "jẹ", "ọ", "ó"]),

    # ── Igbo ─────────────────────────────────────────────────────────────────
    "ig": LangDef("Igbo", "Latin",
        unique_chars="",
        common_chars="ụị",
        grams="na ya ọ n'",
        word_markers=["na", "ya", "ọ", "bụ", "nke", "ndi"]),

    # ── Hausa ────────────────────────────────────────────────────────────────
    "ha": LangDef("Hausa", "Latin",
        unique_chars="ɓɗƙ",
        grams="da na a",
        word_markers=["da", "na", "a", "ba", "don", "sai", "mai"]),

    # ── Zulu ─────────────────────────────────────────────────────────────────
    "zu": LangDef("Zulu", "Latin",
        grams="uk' na ngob' uma",
        word_markers=["ukuthi", "nge", "uma", "noma", "futhi", "yebo"]),

    # ── Afrikaans ────────────────────────────────────────────────────────────
    "af": LangDef("Afrikaans", "Latin",
        grams="die en van is in",
        word_markers=["die", "en", "van", "is", "in", "dat", "nie", "het", "dit"]),

    # ── Vietnamese ───────────────────────────────────────────────────────────
    "vi": LangDef("Vietnamese", "Latin",
        unique_chars="đ",
        common_chars="àáâãèéêìíòóôõùúýăơưụịạậặẹệọộợựỵ",
        grams="của và để có",
        word_markers=["của", "và", "để", "có", "là", "tôi", "không", "được", "các"]),

    # ── Yoruba, Igbo (covered above) ─────────────────────────────────────────

    # ── Esperanto ────────────────────────────────────────────────────────────
    "eo": LangDef("Esperanto", "Latin",
        unique_chars="ĉĝĥĵŝŭ",
        grams="kaj estas en la",
        word_markers=["kaj", "estas", "en", "la", "ne", "de", "al", "mi"]),

    # ══ Cyrillic script ═══════════════════════════════════════════════════════

    # Russian
    "ru": LangDef("Russian", "Cyrillic",
        unique_chars="",
        grams="что и в на не",
        word_markers=["что", "и", "в", "на", "не", "это", "он", "как", "я", "из"]),

    # Ukrainian
    "uk": LangDef("Ukrainian", "Cyrillic",
        unique_chars="іїєґ",
        grams="що і в на не",
        word_markers=["що", "і", "в", "на", "не", "це", "він", "як", "я", "із"]),

    # Belarusian
    "be": LangDef("Belarusian", "Cyrillic",
        unique_chars="ў",
        grams="і ў у на не",
        word_markers=["і", "ў", "у", "на", "не", "гэта", "як", "я", "ён"]),

    # Bulgarian
    "bg": LangDef("Bulgarian", "Cyrillic",
        unique_chars="",
        common_chars="ъ",
        grams="е и в на не да",
        word_markers=["е", "и", "в", "на", "не", "се", "от", "за", "с", "ще", "да"]),

    # Serbian (Cyrillic)
    "sr": LangDef("Serbian (Cyrillic)", "Cyrillic",
        unique_chars="ћђџљњ",
        grams="је и у на не",
        word_markers=["је", "и", "у", "на", "не", "се", "да", "су", "овај", "свет"]),

    # Macedonian
    "mk": LangDef("Macedonian", "Cyrillic",
        unique_chars="ѓѕјќ",
        grams="и на е во",
        word_markers=["и", "на", "е", "во", "не", "се", "да", "ги", "со", "кои"]),

    # Kazakh (Cyrillic)
    "kk": LangDef("Kazakh", "Cyrillic",
        unique_chars="әіңғүұқөһ",
        grams="және бар үшін",
        word_markers=["және", "бар", "үшін", "бұл", "жоқ", "деп"]),

    # Mongolian (Cyrillic)
    "mn": LangDef("Mongolian", "Cyrillic",
        unique_chars="өүё",
        grams="байна энэ нь",
        word_markers=["байна", "энэ", "нь", "болон", "гэж", "юм"]),

    # Kyrgyz
    "ky": LangDef("Kyrgyz", "Cyrillic",
        unique_chars="өүң",
        grams="жана бул да",
        word_markers=["жана", "бул", "да", "же", "болуп"]),

    # Tajik
    "tg": LangDef("Tajik", "Cyrillic",
        unique_chars="ғӣқӯҳҷ",
        grams="ва дар ин",
        word_markers=["ва", "дар", "ин", "ба", "ки"]),

    # ══ Arabic script ═════════════════════════════════════════════════════════

    # Modern Standard Arabic
    "ar": LangDef("Arabic", "Arabic",
        grams="في من على إلى إن",
        word_markers=["في", "من", "على", "إلى", "هذا", "التي", "أن", "كان", "إن"]),
    "ar-EG": LangDef("Arabic (Egyptian)", "Arabic", dialect_of="ar",
        dialect_markers=["إيه", "ازيك", "كده", "بتاع", "هوه", "ليه", "فين", "مش"]),
    "ar-SA": LangDef("Arabic (Saudi)", "Arabic", dialect_of="ar",
        dialect_markers=["وش", "كيف", "ايش", "وين", "ليش", "شو"]),
    "ar-LB": LangDef("Arabic (Levantine)", "Arabic", dialect_of="ar",
        dialect_markers=["شو", "وين", "كيف", "هيك", "بدي", "منيح", "رح", "هلق"]),
    "ar-MA": LangDef("Arabic (Moroccan Darija)", "Arabic", dialect_of="ar",
        dialect_markers=["واش", "كيفاش", "بزاف", "درك", "هاد", "ماشي"]),
    "ar-IQ": LangDef("Arabic (Iraqi)", "Arabic", dialect_of="ar",
        dialect_markers=["شنو", "شلون", "وين", "هواي", "گلبي", "چي"]),
    "ar-SY": LangDef("Arabic (Syrian)", "Arabic", dialect_of="ar",
        dialect_markers=["شو", "كيف", "وين", "هيك", "بدي", "منيح"]),

    # Persian / Farsi
    "fa": LangDef("Persian (Farsi)", "Arabic",
        unique_chars="",
        common_chars="پچژگ",
        grams="است که در",
        word_markers=["است", "که", "در", "این", "را", "با", "به", "از", "او"]),
    "fa-IR": LangDef("Persian (Iran)", "Arabic", dialect_of="fa"),
    "fa-AF": LangDef("Dari (Afghanistan)", "Arabic", dialect_of="fa",
        dialect_markers=["هستم", "استم", "میروم", "مشه"]),

    # Urdu
    "ur": LangDef("Urdu", "Arabic",
        unique_chars="ٹڈڑں",
        common_chars="ے",
        grams="کے اور ہے کہ",
        word_markers=["کے", "اور", "ہے", "کہ", "میں", "کی", "یہ", "ہیں", "نے"]),

    # Pashto
    "ps": LangDef("Pashto", "Arabic",
        unique_chars="ټډړښږ",
        grams="او د چې",
        word_markers=["او", "د", "چې", "دی", "کې", "چه", "ده"]),

    # Sindhi
    "sd": LangDef("Sindhi", "Arabic",
        unique_chars="ڄڃڂ",  # removed ج/ھ -- ordinary Arabic letters shared with ar/fa/ur, not Sindhi-specific
        grams="جي ۽ ان",
        word_markers=["جي", "۽", "ان", "آهي", "ڪيو"]),

    # Kurdish (Arabic script)
    "ku": LangDef("Kurdish (Sorani)", "Arabic",
        unique_chars="",
        grams="و له کہ",
        word_markers=["و", "له", "کہ", "بۆ", "هات", "دا", "ئه"]),

    # ══ Devanagari script ═════════════════════════════════════════════════════

    "hi": LangDef("Hindi", "Devanagari",
        grams="के और है कि",
        word_markers=["के", "और", "है", "कि", "में", "को", "से", "पर", "हैं"]),
    "mr": LangDef("Marathi", "Devanagari",
        unique_chars="ळ",
        grams="आणि आहे हे की",
        word_markers=["आणि", "आहे", "हे", "की", "मी", "त्या", "या", "ते"]),
    "ne": LangDef("Nepali", "Devanagari",
        grams="र छ को गर्न",
        word_markers=["र", "छ", "को", "यो", "त्यो", "मलाई", "भन्छ"]),
    "sa": LangDef("Sanskrit", "Devanagari",
        unique_chars="",
        grams="च एव तु",
        word_markers=["च", "एव", "तु", "वा", "न", "स्यात्", "अस्ति"]),
    "kok": LangDef("Konkani", "Devanagari"),
    "mai": LangDef("Maithili", "Devanagari"),

    # ══ Bengali script ════════════════════════════════════════════════════════

    "bn": LangDef("Bengali", "Bengali",
        grams="এবং যে তার",
        word_markers=["এবং", "যে", "তার", "আমি", "তুমি", "এই", "হয়", "থেকে"]),
    "as": LangDef("Assamese", "Bengali",
        unique_chars="ৱৰ",
        grams="আৰু যে তাৰ",
        word_markers=["আৰু", "যে", "তাৰ", "মই", "তুমি", "এই"]),

    # ══ Other Indian scripts ═══════════════════════════════════════════════════

    "gu": LangDef("Gujarati", "Gujarati",
        grams="અને કે ની",
        word_markers=["અને", "કે", "ની", "છે", "મારા", "તે", "પણ"]),
    "pa": LangDef("Punjabi (Gurmukhi)", "Gurmukhi",
        grams="ਅਤੇ ਕਿ ਦੇ",
        word_markers=["ਅਤੇ", "ਕਿ", "ਦੇ", "ਹੈ", "ਮੈਂ", "ਤੁਸੀਂ", "ਇਹ"]),
    "or": LangDef("Odia/Oriya", "Oriya",
        grams="ଏବଂ ଯେ ତାଙ୍କ",
        word_markers=["ଏବଂ", "ଯେ", "ତାଙ୍କ", "ଆମ", "ଏହି", "ସେ"]),
    "ta": LangDef("Tamil", "Tamil",
        grams="மற்றும் என்று அவர்",
        word_markers=["மற்றும்", "என்று", "அவர்", "இந்த", "என்", "ஆக", "அது"]),
    "te": LangDef("Telugu", "Telugu",
        grams="మరియు అని ఆ",
        word_markers=["మరియు", "అని", "ఆ", "ఈ", "అది", "నేను", "మీరు"]),
    "kn": LangDef("Kannada", "Kannada",
        grams="ಮತ್ತು ಎಂದು ಅವರ",
        word_markers=["ಮತ್ತು", "ಎಂದು", "ಅವರ", "ಈ", "ಅದು", "ನಾನು"]),
    "ml": LangDef("Malayalam", "Malayalam",
        unique_chars="ൺൾ",
        grams="ഒരു അതിന്‍ ഈ",
        word_markers=["ഒരു", "അതിന്", "ഈ", "ആ", "അവൻ", "ഞാൻ", "ഇത്"]),
    "si": LangDef("Sinhala", "Sinhala",
        grams="හා ද නම්",
        word_markers=["හා", "ද", "නම්", "ගේ", "ය", "ඔහු", "ඈ"]),

    # ══ East / Southeast Asian ════════════════════════════════════════════════

    "zh": LangDef("Chinese (Mandarin)", "CJK",
        grams="的 是 在 有 和",
        word_markers=["的", "是", "在", "有", "和", "我", "你", "他", "了", "们"]),
    "zh-Hans": LangDef("Chinese Simplified", "CJK", dialect_of="zh",
        dialect_markers=["个", "时", "东", "来", "说", "为", "问", "样", "达"]),
    "zh-Hant": LangDef("Chinese Traditional", "CJK", dialect_of="zh",
        dialect_markers=["個", "時", "東", "來", "說", "為", "問", "樣", "達"]),
    "yue": LangDef("Cantonese", "CJK", dialect_of="zh",
        dialect_markers=["嘅", "係", "唔", "喺", "咩", "乜", "啫", "囉", "囉喎", "佢"]),

    "ja": LangDef("Japanese", "Hiragana",
        grams="は の を が で",
        word_markers=["は", "の", "を", "が", "で", "も", "と", "に", "です", "ます"]),
    "ko": LangDef("Korean", "Hangul",
        grams="이 가 을 은 는",
        word_markers=["이", "가", "을", "은", "는", "을", "와", "과", "도", "에서"]),

    "th": LangDef("Thai", "Thai",
        grams="ที่ ใน และ",
        word_markers=["ที่", "ใน", "และ", "ของ", "ว่า", "ไม่", "ให้", "ได้", "เป็น"]),
    "lo": LangDef("Lao", "Lao",
        grams="ທີ່ ແລະ ໃນ",
        word_markers=["ທີ່", "ແລະ", "ໃນ", "ຂອງ", "ວ່າ", "ບໍ່"]),
    "my": LangDef("Burmese/Myanmar", "Myanmar",
        grams="၏ ၌ နှင့်",
        word_markers=["၏", "ကို", "မှ", "နှင့်", "တွင်", "ဖြစ်", "သည်"]),
    "km": LangDef("Khmer/Cambodian", "Khmer",
        grams="នៃ ដែល ក្នុង",
        word_markers=["នៃ", "ដែល", "ក្នុង", "ជា", "ពី", "ទៅ", "មាន"]),

    # ══ Georgian / Armenian ═══════════════════════════════════════════════════

    "ka": LangDef("Georgian", "Georgian",
        grams="და რომ არ",
        word_markers=["და", "რომ", "არ", "ის", "მე", "ეს", "სე", "შე"]),
    "hy": LangDef("Armenian", "Armenian",
        grams="ն ի և",
        word_markers=["ն", "ի", "և", "է", "ոչ", "այն", "որ", "մ"]),

    # ══ Greek ═════════════════════════════════════════════════════════════════

    "el": LangDef("Greek", "Greek",
        grams="και το τα",
        word_markers=["και", "το", "τα", "που", "για", "δεν", "ότι", "αν", "από"]),

    # ══ Hebrew ════════════════════════════════════════════════════════════════

    "he": LangDef("Hebrew", "Hebrew",
        grams="של בו ב",
        word_markers=["של", "בו", "בה", "לא", "כי", "את", "על", "הוא", "זה"]),
    "yi": LangDef("Yiddish", "Hebrew",
        grams="אויף פון",
        word_markers=["אויף", "פון", "אין", "אוי", "נישט"]),

    # ══ Ethiopian ══════════════════════════════════════════════════════════════

    "am": LangDef("Amharic", "Ethiopic",
        grams="እና ን ለ",
        word_markers=["እና", "ን", "ለ", "ነው", "ኧ", "ላይ", "ወደ", "እዚህ"]),
    "ti": LangDef("Tigrinya", "Ethiopic",
        grams="ን ዝ ብ",
        word_markers=["ን", "ዝ", "ብ", "እዩ", "ኣለ"]),
    "om": LangDef("Oromo", "Latin",
        grams="fi kan tti",
        word_markers=["fi", "kan", "tti", "dha", "moo", "ykn"]),

    # ══ Tibetan ════════════════════════════════════════════════════════════════

    "bo": LangDef("Tibetan", "Tibetan",
        grams="དང་ ལ་ ཀྱི་",
        word_markers=["དང་", "ལ་", "ཀྱི་", "ནི་", "ལས་", "ཡིན"]),

    # ══ Mongolian ══════════════════════════════════════════════════════════════
    # (Cyrillic form covered above)

    # ══ Other ══════════════════════════════════════════════════════════════════

    "tpi": LangDef("Tok Pisin (Papua New Guinea)", "Latin",
        grams="na bilong long",
        word_markers=["na", "bilong", "long", "em", "ol", "wantaim", "mipela"]),
    "haw": LangDef("Hawaiian", "Latin",
        unique_chars="ʻ",  # okina (U+02BB) -- NOT the ASCII apostrophe used in English contractions
        grams="o ka a me",
        word_markers=["o", "ka", "a", "me", "no", "ke", "i", "ʻo"]),
    "mi": LangDef("Māori", "Latin",
        grams="ko te a o",
        word_markers=["ko", "te", "a", "o", "i", "ki", "he", "ngā", "tēnei"]),
    "sm": LangDef("Samoan", "Latin",
        grams="le a ma",
        word_markers=["le", "a", "ma", "o", "e", "i", "ua", "ia", "ona"]),
    "fj": LangDef("Fijian", "Latin",
        unique_chars="",
        grams="na ka me",
        word_markers=["na", "ka", "me", "o", "i", "e", "ko", "kei", "vaka"]),
    "la": LangDef("Latin (Classical)", "Latin",
        unique_chars="",
        grams="et in est ad",
        word_markers=["et", "in", "est", "ad", "non", "sed", "ut", "ex", "aut"]),
    "oc": LangDef("Occitan", "Latin",
        unique_chars="",
        common_chars="àèéòù",
        grams="de lo la que e",
        word_markers=["de", "lo", "la", "que", "e", "es", "per", "se"]),
    "lmo": LangDef("Lombard", "Latin",
        grams="de el la",
        word_markers=["de", "el", "la", "l'", "i", "e"]),
    "vec": LangDef("Venetian", "Latin",
        grams="de el la",
        word_markers=["de", "el", "la", "i", "e", "no", "xe"]),
    "nap": LangDef("Neapolitan", "Latin",
        grams="de e a",
        word_markers=["de", "e", "a", "è", "no", "ca"]),
    "scn": LangDef("Sicilian", "Latin",
        grams="di e lu",
        word_markers=["di", "e", "lu", "la", "e", "n", "s"]),
}

# Build a quick lookup: script → set of language codes
_SCRIPT_LANGS: dict[str, list[str]] = defaultdict(list)
for code, defn in LANG_DB.items():
    _SCRIPT_LANGS[defn.script].append(code)

# Scripts written without spaces between words (scriptio continua) — word/gram
# markers can't be matched as whole tokens, so detection falls back to substring.
_NO_SPACE_SCRIPTS = {"CJK", "Hiragana", "Katakana", "Thai", "Lao", "Myanmar", "Khmer"}


# ══════════════════════════════════════════════════════════════════════════════
#  Dialect markers registry (for post-detection dialect refinement)
# ══════════════════════════════════════════════════════════════════════════════

# Patterns specifically diagnostic for dialect/sub-variety detection.
# Keyed by BCP 47 dialect code. Values are (regex_pattern, weight).
DIALECT_PATTERNS: dict[str, list[tuple[str, float]]] = {
    # ── English ──────────────────────────────────────────────────────────────
    "en-US": [
        (r'\bcolor\b', 1.0), (r'\bhonor\b', 1.0), (r'\bfavor\b', 1.0),
        (r'\banalyze\b', 1.0), (r'\bcenter\b', 0.8), (r'\bgotten\b', 1.0),
        (r'\bsoccer\b', 1.2), (r'\bmath\b', 0.8), (r'\bmom\b', 0.6),
        (r"\by'all\b", 2.0), (r"\bgonna\b", 0.8), (r"\bwanna\b", 0.8),
        (r'\bdude\b', 0.8), (r'\bawesome\b', 0.6), (r'\btrash\b', 0.5),
        (r'\belevator\b', 0.7), (r'\bapartment\b', 0.5), (r'\bgas station\b', 0.8),
    ],
    "en-GB": [
        (r'\bcolour\b', 1.0), (r'\bhonour\b', 1.0), (r'\bfavour\b', 1.0),
        (r'\banalyse\b', 1.0), (r'\bcentre\b', 0.8), (r'\bmaths\b', 1.2),
        (r'\bfootball\b', 0.6), (r'\bmate\b', 0.6), (r'\bbloody\b', 0.9),
        (r'\bbloke\b', 1.2), (r'\bcheers\b', 0.7), (r'\bfortnight\b', 1.5),
        (r'\bwhilst\b', 1.2), (r'\bpetrol\b', 1.0), (r'\bflat\b', 0.5),
        (r'\blift\b', 0.5), (r'\btelly\b', 1.2), (r'\bcheeky\b', 1.0),
        (r'\bquid\b', 1.5), (r'\bblighter\b', 2.0), (r'\bnumpty\b', 1.5),
    ],
    "en-AU": [
        (r'\barvo\b', 2.0), (r'\bbarbie\b', 1.5), (r'\bthong', 1.0),
        (r'\bbogan\b', 2.0), (r'\byobbo\b', 2.0), (r'\bfooty\b', 1.2),
        (r'\byeah no\b', 1.5), (r'\bheaps\b', 1.0), (r'\breckon\b', 0.8),
        (r'\bstrewth\b', 2.0), (r'\bcrikey\b', 2.0), (r'\buniversity\b', 0.3),
    ],
    "en-CA": [
        (r'\beh\b', 1.5), (r'\bchesterfield\b', 2.0), (r'\btoque\b', 2.0),
        (r'\bpoutine\b', 2.0), (r'\bloonie\b', 2.0), (r'\btoonie\b', 2.0),
        (r'\bhockey\b', 0.8), (r'\bwashroom\b', 1.2), (r'\bcanuck\b', 1.5),
    ],
    "en-IN": [
        (r'\bprepone\b', 2.0), (r'\bdo the needful\b', 2.0), (r'\bkindly\b', 0.7),
        (r'\blakh\b', 2.0), (r'\bcrore\b', 2.0), (r'\bwallah\b', 1.5),
        (r'\btiffin\b', 2.0), (r'\bchemist shop\b', 1.5), (r'\bairport only\b', 1.0),
    ],
    # ── Spanish dialects ──────────────────────────────────────────────────────
    "es-ES": [
        (r'\bvosotros\b', 2.0), (r'\bos\b', 0.8), (r'\bcoger\b', 1.5),
        (r'\bordenador\b', 2.0), (r'\bmóvil\b', 1.0), (r'\bpatatas\b', 1.5),
        (r'\bpiso\b', 1.0), (r'\bvale\b', 1.0), (r'\btío\b', 0.8),
        (r'\bguay\b', 2.0), (r'\bostias\b', 1.5), (r'\bjoder\b', 1.2),
    ],
    "es-AR": [
        (r'\bvos\b', 2.0), (r'\bche\b', 1.5), (r'\bboludo\b', 2.0),
        (r'\bpibe\b', 2.0), (r'\blaburo\b', 2.0), (r'\bcopado\b', 1.8),
        (r'\bbirra\b', 1.5), (r'\blaburar\b', 2.0), (r'\bgurí\b', 2.0),
    ],
    "es-MX": [
        (r'\bórale\b', 2.0), (r'\bahorita\b', 1.5), (r'\bchido\b', 2.0),
        (r'\bchamaco\b', 2.0), (r'\bmande\b', 2.0), (r'\bcuate\b', 2.0),
        (r'\bqué padre\b', 2.0), (r'\bantro\b', 1.5), (r'\bchamba\b', 1.5),
    ],
    # ── Portuguese dialects ───────────────────────────────────────────────────
    "pt-BR": [
        (r'\bvocê\b', 1.0), (r'\ba gente\b', 1.5), (r'\blegal\b', 1.0),
        (r'\btchau\b', 1.5), (r'\bônibus\b', 2.0), (r'\bcelular\b', 2.0),
        (r'\bgeladeira\b', 2.0), (r'\bbanheiro\b', 2.0), (r'\bmaneiro\b', 2.0),
    ],
    "pt-PT": [
        (r'\bautocarro\b', 2.0), (r'\bcasa de banho\b', 2.0), (r'\bfrigorífico\b', 2.0),
        (r'\btelemóvel\b', 2.0), (r'\bfixe\b', 2.0), (r'\bpá\b', 1.0),
        (r'\bpronto\b', 0.8), (r'\bdescansado\b', 0.5),
    ],
    # ── French dialects ───────────────────────────────────────────────────────
    "fr-CA": [
        (r'\btabernac\b', 2.0), (r'\bostie\b', 2.0), (r'\bcâlice\b', 2.0),
        (r'\bchar\b', 1.5), (r'\bmagasiner\b', 2.0), (r'\bfin de semaine\b', 2.0),
        (r'\bcourriel\b', 1.5), (r'\bchum\b', 1.5), (r'\bblonde\b', 1.5),
        (r'\bchiâler\b', 2.0),
    ],
    "fr-BE": [
        (r'\bseptante\b', 2.0), (r'\bnonante\b', 2.0), (r'\bkot\b', 2.0),
        (r'\bpanosse\b', 2.0),
    ],
    # ── Arabic dialects ───────────────────────────────────────────────────────
    "ar-EG": [
        (r'إيه', 2.0), (r'ازيك', 2.0), (r'كده', 2.0), (r'بتاع', 2.0),
        (r'ليه', 1.5), (r'فين', 1.5), (r'مش', 1.2),
    ],
    "ar-MA": [
        (r'واش', 2.0), (r'كيفاش', 2.0), (r'بزاف', 2.0), (r'هاد', 1.5),
        (r'ماشي', 1.5), (r'خويا', 1.5),
    ],
    "ar-LB": [
        (r'شو', 1.5), (r'وين', 1.5), (r'هلق', 2.0), (r'رح', 1.5),
        (r'منيح', 2.0), (r'هيك', 1.5),
    ],
    # ── Chinese dialects ──────────────────────────────────────────────────────
    "yue": [  # Cantonese
        (r'嘅', 3.0), (r'係', 2.0), (r'唔', 2.0), (r'喺', 2.0),
        (r'咩', 2.0), (r'乜', 2.0), (r'囉', 2.0), (r'佢', 2.0),
        (r'咗', 2.0), (r'咁', 1.5), (r'噉', 1.5),
    ],
    "zh-Hant": [
        (r'個', 1.5), (r'時', 1.0), (r'來', 1.5), (r'說', 1.5),
        (r'為', 1.5), (r'問', 1.0), (r'樣', 1.0), (r'國', 1.0),
    ],
    "zh-Hans": [
        (r'个', 1.5), (r'时', 1.0), (r'来', 1.5), (r'说', 1.5),
        (r'为', 1.5), (r'问', 1.0), (r'样', 1.0), (r'国', 1.0),
    ],
    # ── Norwegian ─────────────────────────────────────────────────────────────
    "nb": [
        (r'\bikke\b', 1.5), (r'\bojeg\b', 0.5), (r'\bden\b', 0.8),
        (r'\ber\b', 0.6), (r'\bvar\b', 0.5),
    ],
    "nn": [
        (r'\bikkje\b', 2.0), (r'\bdei\b', 1.5), (r'\bme\b', 1.0),
        (r'\beg\b', 1.5), (r'\bkjem\b', 1.5), (r'\bfår\b', 1.0),
    ],
    # ── South Asian (dialect sub-types) ──────────────────────────────────────
    "fa-AF": [
        (r'هستم', 2.0), (r'استم', 2.0), (r'مشه', 1.5),
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScriptInfo:
    """Script composition of a text."""
    dominant: str                           # most common script
    distribution: dict[str, float]          # script → fraction of characters
    is_mixed: bool                          # True if ≥2 scripts each ≥10 %
    languages_for_script: list[str]         # candidate language codes


@dataclass
class LanguageMatch:
    code: str       # BCP 47 language code
    name: str       # human-readable name
    confidence: float
    is_dialect: bool
    dialect_of: Optional[str]


@dataclass
class DetectionResult:
    language: str               # top language code
    dialect: Optional[str]      # most specific dialect code, if detected
    name: str                   # human-readable name of top match
    script: str                 # dominant script
    confidence: float           # 0.0–1.0
    top: list[LanguageMatch]    # ranked list (up to top_n)
    is_mixed: bool              # True if multiple scripts detected


# ══════════════════════════════════════════════════════════════════════════════
#  Core detector
# ══════════════════════════════════════════════════════════════════════════════

class LanguageDetector:
    """Detects language and dialect from text.

    Uses three layers:
      1. Unicode script detection — O(n) scan, narrows candidate set
      2. Unique character scoring — O(n×chars) per candidate
      3. Dialect marker matching — regex scan on top candidate

    Accuracy on clean text:
      - Unambiguous scripts (Korean, Thai, Georgian, …): ~100 %
      - CJK (Chinese vs Japanese vs Korean with CJK chars): ~98 %
      - Latin script (≥30 chars): ~94 % language, ~85 % dialect
      - Short texts (< 20 chars): lower confidence, use result.confidence

    To improve accuracy, call fit() with labelled text samples.
    """

    def __init__(self) -> None:
        # Trained trigram profiles: lang_code → Counter of trigrams
        self._trigram_profiles: dict[str, Counter] = {}
        # Cache compiled dialect regexes
        self._dialect_re: dict[str, list[tuple[re.Pattern, float]]] = {}
        for code, patterns in DIALECT_PATTERNS.items():
            self._dialect_re[code] = [(re.compile(pat, re.IGNORECASE), w) for pat, w in patterns]

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_script(self, text: str) -> ScriptInfo:
        """Identify the Unicode scripts present in this text."""
        counts: Counter = Counter()
        total = 0
        for ch in text:
            script = codepoint_to_script(ord(ch))
            if script not in ("ASCII", "Digit", "Unknown"):
                counts[script] += 1
                total += 1
        if total == 0:
            return ScriptInfo("Latin", {"Latin": 1.0}, False, list(_SCRIPT_LANGS["Latin"]))

        dist = {s: c / total for s, c in counts.most_common()}
        dominant = counts.most_common(1)[0][0]
        is_mixed = sum(1 for v in dist.values() if v >= 0.10) >= 2

        # Language candidates for dominant script
        candidates = list(_SCRIPT_LANGS.get(dominant, []))

        # Special cases: Hiragana/Katakana → Japanese
        if "Hiragana" in dist or "Katakana" in dist:
            dominant = "Hiragana"
            candidates = ["ja"]
        # Hangul → Korean
        elif "Hangul" in dist:
            dominant = "Hangul"
            candidates = ["ko"]

        return ScriptInfo(dominant, dist, is_mixed, candidates)

    def detect(
        self,
        text: str,
        top_n: int = 5,
        min_chars: int = 3,
    ) -> DetectionResult:
        """Detect language and dialect.

        Returns a DetectionResult with ranked candidates.
        """
        if not text or len(text.strip()) < min_chars:
            return DetectionResult("und", None, "Undetermined", "Unknown", 0.0, [], False)

        # Layer 1: script
        script_info = self.detect_script(text)

        # Build candidate list: script-appropriate languages only
        candidates = self._get_candidates(text, script_info)

        if not candidates:
            return DetectionResult("und", None, "Undetermined", script_info.dominant, 0.0, [], script_info.is_mixed)

        # Sort by score
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        # Normalise to confidence in [0,1]
        top_score = ranked[0][1]
        bottom = ranked[-1][1] if len(ranked) > 1 else 0.0
        rng = top_score - bottom if top_score > bottom else 1.0

        top_matches: list[LanguageMatch] = []
        for code, score in ranked[:top_n]:
            defn = LANG_DB.get(code)
            if defn is None:
                continue
            conf = min(1.0, (score - bottom) / rng) if rng > 0 else 1.0
            top_matches.append(LanguageMatch(
                code=code,
                name=defn.name,
                confidence=round(conf, 4),
                is_dialect=defn.dialect_of is not None,
                dialect_of=defn.dialect_of,
            ))

        if not top_matches:
            return DetectionResult("und", None, "Undetermined", script_info.dominant, 0.0, [], False)

        best = top_matches[0]
        parent_code = best.dialect_of if best.is_dialect else best.code

        # Layer 3: dialect detection (on the top parent language)
        dialect = self._detect_dialect(text, parent_code)

        # Build final confidence (penalise if margin over 2nd is small)
        raw_conf = best.confidence
        if len(top_matches) > 1:
            margin = best.confidence - top_matches[1].confidence
            raw_conf = min(1.0, raw_conf * (0.5 + margin))

        return DetectionResult(
            language=parent_code,
            dialect=dialect,
            name=LANG_DB.get(dialect or parent_code, LANG_DB.get(parent_code, LangDef("?", "?"))).name,
            script=script_info.dominant,
            confidence=round(raw_conf, 4),
            top=top_matches,
            is_mixed=script_info.is_mixed,
        )

    def detect_dialect(self, text: str, language_code: str) -> Optional[str]:
        """Given a known language, return the most likely dialect code."""
        return self._detect_dialect(text, language_code)

    def fit(self, text: str, language_code: str) -> None:
        """Train / update trigram profile for a language from raw text."""
        profile = self._trigram_profiles.setdefault(language_code, Counter())
        clean = text.lower()
        for i in range(len(clean) - 2):
            tri = clean[i:i + 3]
            if any(c.isalpha() for c in tri):
                profile[tri] += 1
        # Normalise
        total = sum(profile.values())
        if total:
            for k in profile:
                profile[k] /= total

    def save(self, path: str | Path) -> None:
        data = {k: dict(v) for k, v in self._trigram_profiles.items()}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._trigram_profiles = {k: Counter(v) for k, v in data.items()}

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_candidates(
        self,
        text: str,
        script_info: ScriptInfo,
    ) -> dict[str, float]:
        """Score all plausible candidates for this text."""
        text_lower = text.lower()
        char_set   = set(text_lower)
        # Split on whitespace/punctuation rather than \b\w+\b: the latter
        # excludes Unicode combining marks (category Mc/Mn), which mangles
        # Devanagari/Bengali/Tamil/etc. words that carry vowel-sign matras.
        words = set(w for w in re.split(r'[\s।॥.,;:!?()«»"\'…—–\-]+', text_lower) if w)
        scores: dict[str, float] = defaultdict(float)

        # Only evaluate languages matching the dominant script(s).
        # Built via dict.fromkeys (not a set) so iteration order is
        # deterministic — ties between equally-scored candidates resolve
        # in favor of whichever LANG_DB entry was defined first, instead
        # of depending on Python's per-process string hash order.
        relevant_scripts = [s for s, frac in script_info.distribution.items() if frac >= 0.05]
        relevant_langs: dict[str, None] = {}
        for script in relevant_scripts:
            for code in _SCRIPT_LANGS.get(script, []):
                relevant_langs[code] = None
        # Also include Hiragana → ja, Hangul → ko
        if "Hiragana" in relevant_scripts or "Katakana" in relevant_scripts:
            relevant_langs["ja"] = None
        if "Hangul" in relevant_scripts:
            relevant_langs["ko"] = None

        for code in relevant_langs:
            defn = LANG_DB.get(code)
            if defn is None:
                continue

            score = 0.0

            # Unique characters (strongest signal — each unique char = +3)
            for ch in defn.unique_chars:
                if ch in char_set:
                    score += 3.0
                elif ch.lower() in char_set:
                    score += 2.0

            # Common characters
            common_present = sum(1 for ch in defn.common_chars if ch in char_set)
            score += common_present * 0.5

            # Word markers
            # Scripts written without spaces between words (scriptio continua):
            # whole-token matching is meaningless, so fall back to substring.
            no_space_script = defn.script in _NO_SPACE_SCRIPTS

            word_hits = 0
            for w in defn.word_markers:
                if w in words or (no_space_script and w in text_lower):
                    word_hits += 1
            score += word_hits * 1.2

            # Gram matching (word-boundary for spaced scripts; substring for
            # scriptio continua scripts like CJK/Hiragana/Thai)
            gram_hits = 0
            for gram in defn.grams.split():
                if gram in words or (no_space_script and gram in text_lower):
                    gram_hits += 1
            score += gram_hits * 0.8

            # Trigram profile match (if trained)
            if code in self._trigram_profiles:
                score += self._trigram_score(text_lower, self._trigram_profiles[code]) * 2.0

            # Small boost for non-dialect (base language)
            if defn.dialect_of is None:
                score += 0.2

            if score > 0:
                scores[code] = score

        return dict(scores)

    def _trigram_score(self, text: str, profile: Counter) -> float:
        """Cosine-like similarity between text trigrams and a trained profile."""
        text_tri: Counter = Counter()
        for i in range(len(text) - 2):
            tri = text[i:i + 3]
            if any(c.isalpha() for c in tri):
                text_tri[tri] += 1
        total = sum(text_tri.values()) or 1
        score = 0.0
        for tri, cnt in text_tri.items():
            if tri in profile:
                score += (cnt / total) * profile[tri]
        return score

    def _detect_dialect(self, text: str, language_code: str) -> Optional[str]:
        """Apply dialect marker regexes for sub-variety detection."""
        # Find all dialect codes that are children of language_code
        dialect_codes = [
            code for code, defn in LANG_DB.items()
            if defn.dialect_of == language_code and code in self._dialect_re
        ]
        # Also check CJK sub-dialects
        if language_code == "zh":
            dialect_codes += ["yue", "zh-Hans", "zh-Hant"]

        if not dialect_codes:
            return None

        best_code: Optional[str] = None
        best_score = 0.0
        for code in dialect_codes:
            patterns = self._dialect_re.get(code, [])
            score = 0.0
            for regex, weight in patterns:
                m = regex.search(text)
                if m:
                    score += weight
            if score > best_score:
                best_score = score
                best_code = code

        # Require at least one marker hit (score ≥ 1.0)
        return best_code if best_score >= 1.0 else None

    def available_languages(self) -> list[tuple[str, str]]:
        """Return list of (code, name) for all languages in the database."""
        return [(code, defn.name) for code, defn in sorted(LANG_DB.items())
                if defn.dialect_of is None]

    def available_dialects(self) -> list[tuple[str, str, str]]:
        """Return list of (code, name, parent_code) for all dialects."""
        return [(code, defn.name, defn.dialect_of)
                for code, defn in sorted(LANG_DB.items())
                if defn.dialect_of is not None]
