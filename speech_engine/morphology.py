"""Word analysis: lemmas, features and POS tags from corpus resources, plus BPE subwords.

With CoNLL-U resources loaded, analyses come from attested forms and a bigram
HMM tagger chooses POS in context. Without them, English falls back to affix
stripping and other languages to simple POS rules. BPE subwords are reported
separately and are not treated as morphemes. Confidence is None unless
calibrated.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import unicodedata
import math

from speech_engine.linguistic_resources import CorpusLexicon, language_code, normalized_form
from speech_engine.unknown_words import UnknownWordModel
from speech_engine.sentence_boundaries import sentence_ranges, validate_ranges
from speech_engine.pos_uncertainty import (
    forward_backward,
    POSCalibration,
    pos_review_recommended,
    pos_model_fingerprint,
)


#  POS tag set (Universal Dependencies style, language-agnostic)


class POS:
    NOUN = "NOUN"  # common noun
    PROPN = "PROPN"  # proper noun
    VERB = "VERB"  # verb (any form)
    AUX = "AUX"  # auxiliary verb
    ADJ = "ADJ"  # adjective
    ADV = "ADV"  # adverb
    PRON = "PRON"  # pronoun
    DET = "DET"  # determiner / article
    PREP = "PREP"  # preposition / postposition
    CONJ = "CONJ"  # coordinating conjunction
    SCONJ = "SCONJ"  # subordinating conjunction
    NUM = "NUM"  # numeral
    PART = "PART"  # particle
    INTJ = "INTJ"  # interjection
    PUNCT = "PUNCT"  # punctuation
    SYM = "SYM"  # symbol
    X = "X"  # other / unknown


#  Morphological feature tags


@dataclass
class MorphFeatures:
    """Morphological features for one word token."""

    number: Optional[str] = None  # "Sing", "Plur", "Dual"
    gender: Optional[str] = None  # "Masc", "Fem", "Neut"
    case: Optional[str] = None  # "Nom", "Acc", "Gen", "Dat", "Loc", "Inst", ...
    tense: Optional[str] = None  # "Pres", "Past", "Fut", "Perf", "Imperf"
    person: Optional[str] = None  # "1", "2", "3"
    mood: Optional[str] = None  # "Ind", "Imp", "Sub", "Cond"
    voice: Optional[str] = None  # "Act", "Pass", "Mid"
    aspect: Optional[str] = None  # "Perf", "Imp" (perfective / imperfective)
    degree: Optional[str] = None  # "Pos", "Comp", "Sup"
    derivation: Optional[str] = None  # "negation", "diminutive", "augmentative", etc.

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class MorphAnalysis:
    """Full morphological analysis of one word."""

    surface: str  # original word form
    root: str  # heuristic stem or surface fallback
    morphemes: list[str]  # proposed segments; empty when unavailable
    pos: str  # POS tag
    features: MorphFeatures  # morphological features
    lang: str  # language code
    # None means uncalibrated; rule matches are not certainty estimates.
    confidence: Optional[float] = None
    method: str = "unavailable"
    subwords: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    ud_features: dict[str, str] = field(default_factory=dict)
    span: Optional[tuple[int, int]] = None
    selected_candidate_indices: list[int] = field(default_factory=list)
    pos_method: str = "heuristic_rules"
    model_source: Optional[dict] = None
    pos_uncertainty: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "root": self.root,
            "pos": self.pos,
            "morphemes": self.morphemes,
            "features": self.features.to_dict(),
            "lang": self.lang,
            "confidence": self.confidence,
            "pos_uncertainty": self.pos_uncertainty,
            "pos_review_recommended": pos_review_recommended(self.pos_uncertainty),
            "method": self.method,
            "subwords": self.subwords,
            "root_status": (
                "corpus_attested"
                if self.candidates and len({c["lemma"] for c in self.candidates}) == 1
                else "predicted"
                if self.method == "corpus_context"
                and len({self.candidates[i]["lemma"] for i in self.selected_candidate_indices}) == 1
                else "heuristic"
                if self.method == "english_affix_rules"
                else "surface"
            ),
            "pos_status": (
                "corpus_attested"
                if self.candidates and len({c["pos"] for c in self.candidates}) == 1
                else "predicted"
                if self.method == "corpus_context"
                else "ambiguous"
                if self.method == "corpus_ambiguous"
                else "structural"
                if self.method == "structural"
                else "predicted"
                if self.pos_method == "bigram_hmm"
                else "heuristic"
            ),
            "candidates": self.candidates,
            "ud_features": self.ud_features,
            "span": list(self.span) if self.span is not None else None,
            "span_unit": "unicode_codepoints",
            "selected_candidate_indices": self.selected_candidate_indices,
            "pos_method": self.pos_method,
            "model_source": self.model_source,
        }


@dataclass
class TokenAnalysis:
    """Analysis of a full tokenised sentence."""

    tokens: list[str]
    analyses: list[MorphAnalysis]
    lang: str
    text: Optional[str] = None  # Original input; absent for supplied tokens.
    sentence_ranges: list = field(default_factory=list)
    sentence_boundary_method: str = "supplied_token_boundaries"


#  Closed-class word lists per language (POS: DET / PREP / CONJ / PRON)

CLOSED_CLASS: dict[str, dict[str, list[str]]] = {
    "en": {
        POS.DET: [
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "some",
            "any",
            "no",
            "each",
            "every",
            "either",
            "neither",
            "all",
            "both",
            "half",
        ],
        POS.PREP: [
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "to",
            "into",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "around",
            "of",
            "near",
            "behind",
            "beside",
            "beyond",
            "within",
            "without",
        ],
        POS.CONJ: ["and", "but", "or", "nor", "for", "yet", "so"],
        POS.SCONJ: [
            "although",
            "because",
            "since",
            "unless",
            "until",
            "when",
            "where",
            "while",
            "if",
            "though",
            "whereas",
            "whether",
            "as",
            "that",
        ],
        POS.PRON: [
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "you",
            "your",
            "yours",
            "yourself",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "who",
            "whom",
            "whose",
            "which",
            "what",
            "that",
            "this",
            "these",
            "those",
        ],
        POS.AUX: [
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "can",
            "could",
            "ought",
        ],
    },
    "es": {
        POS.DET: [
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "unos",
            "unas",
            "este",
            "esta",
            "estos",
            "estas",
            "ese",
            "esa",
            "esos",
            "esas",
            "aquel",
            "aquella",
            "aquellos",
            "aquellas",
            "mi",
            "tu",
            "su",
            "nuestro",
        ],
        POS.PREP: [
            "a",
            "ante",
            "bajo",
            "con",
            "contra",
            "de",
            "desde",
            "durante",
            "en",
            "entre",
            "hacia",
            "hasta",
            "mediante",
            "para",
            "por",
            "según",
            "sin",
            "sobre",
            "tras",
        ],
        POS.CONJ: ["y", "e", "o", "u", "ni", "pero", "sino", "mas"],
        POS.SCONJ: [
            "si",
            "aunque",
            "porque",
            "cuando",
            "como",
            "mientras",
            "donde",
            "que",
            "para",
            "sino",
            "hasta",
        ],
        POS.PRON: [
            "yo",
            "tú",
            "él",
            "ella",
            "nosotros",
            "vosotros",
            "ellos",
            "ellas",
            "me",
            "te",
            "se",
            "lo",
            "la",
            "le",
            "nos",
            "os",
            "les",
            "que",
            "quien",
            "cuál",
            "qué",
        ],
        POS.AUX: [
            "he",
            "has",
            "ha",
            "hemos",
            "habéis",
            "han",
            "había",
            "habías",
            "habíamos",
            "habían",
            "estoy",
            "estás",
            "está",
            "estamos",
            "estáis",
            "están",
            "estaba",
            "estabas",
            "estábamos",
            "estaban",
            "soy",
            "eres",
            "es",
            "somos",
            "sois",
            "son",
            "era",
            "eras",
            "éramos",
            "eran",
            "fue",
            "fueron",
            "puedo",
            "puedes",
            "puede",
            "podemos",
            "podéis",
            "pueden",
            "debo",
            "debes",
            "debe",
            "debemos",
            "debéis",
            "deben",
        ],
    },
    "fr": {
        POS.DET: [
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "son",
            "sa",
            "ses",
            "notre",
            "nos",
            "votre",
            "vos",
            "leur",
            "leurs",
            "ce",
            "cet",
            "cette",
            "ces",
        ],
        POS.PREP: [
            "à",
            "de",
            "en",
            "dans",
            "sur",
            "sous",
            "avec",
            "sans",
            "pour",
            "par",
            "entre",
            "vers",
            "chez",
            "avant",
            "après",
            "depuis",
            "pendant",
            "contre",
            "selon",
            "malgré",
        ],
        POS.CONJ: ["et", "ou", "mais", "donc", "or", "ni", "car"],
        POS.PRON: [
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "me",
            "te",
            "se",
            "le",
            "la",
            "les",
            "lui",
            "leur",
            "y",
            "en",
            "qui",
            "que",
            "quoi",
            "dont",
            "où",
        ],
        POS.AUX: [
            "ai",
            "as",
            "a",
            "avons",
            "avez",
            "ont",
            "avait",
            "avaient",
            "avais",
            "avions",
            "aviez",
            "suis",
            "es",
            "est",
            "sommes",
            "êtes",
            "sont",
            "étais",
            "était",
            "étions",
            "étiez",
            "étaient",
            "peux",
            "peut",
            "pouvons",
            "pouvez",
            "peuvent",
            "dois",
            "doit",
            "devons",
            "devez",
            "doivent",
        ],
    },
    "de": {
        POS.DET: [
            "der",
            "die",
            "das",
            "dem",
            "den",
            "des",
            "ein",
            "eine",
            "einen",
            "einem",
            "eines",
            "einer",
            "kein",
            "keine",
            "mein",
            "dein",
            "sein",
        ],
        POS.PREP: [
            "an",
            "auf",
            "aus",
            "bei",
            "durch",
            "für",
            "gegen",
            "in",
            "mit",
            "nach",
            "neben",
            "über",
            "um",
            "unter",
            "von",
            "vor",
            "zu",
            "zwischen",
        ],
        POS.CONJ: ["und", "aber", "oder", "denn", "sondern"],
        POS.PRON: [
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "mich",
            "mir",
            "dich",
            "dir",
            "ihm",
            "ihn",
            "uns",
            "euch",
            "ihnen",
            "wer",
            "was",
            "welche",
        ],
        POS.AUX: [
            "habe",
            "hast",
            "hat",
            "haben",
            "habt",
            "hatte",
            "hatten",
            "bin",
            "bist",
            "ist",
            "sind",
            "seid",
            "war",
            "waren",
            "werde",
            "wirst",
            "wird",
            "werdet",
            "wurde",
            "wurden",
            "kann",
            "kannst",
            "können",
            "könnt",
            "muss",
            "musst",
            "müssen",
            "müsst",
            "soll",
            "sollst",
            "sollen",
            "sollt",
            "will",
            "willst",
            "wollen",
            "wollt",
            "darf",
            "darfst",
            "dürfen",
            "dürft",
            "mag",
            "magst",
            "mögen",
            "mögt",
        ],
    },
    "ru": {
        POS.PREP: [
            "в",
            "на",
            "с",
            "к",
            "от",
            "до",
            "за",
            "по",
            "из",
            "под",
            "над",
            "при",
            "без",
            "через",
            "между",
            "около",
            "вместо",
        ],
        POS.CONJ: ["и", "а", "но", "или", "ни", "да", "однако", "же"],
        POS.PRON: [
            "я",
            "ты",
            "он",
            "она",
            "оно",
            "мы",
            "вы",
            "они",
            "меня",
            "тебя",
            "его",
            "её",
            "нас",
            "вас",
            "их",
            "кто",
            "что",
            "который",
        ],
        POS.AUX: [
            "был",
            "была",
            "было",
            "были",
            "будет",
            "будут",
            "есть",
            "может",
            "можно",
            "нужно",
            "надо",
            "должен",
            "должна",
            "должны",
        ],
    },
    "ar": {
        POS.PREP: [
            "في",
            "على",
            "إلى",
            "من",
            "عن",
            "مع",
            "بين",
            "بعد",
            "قبل",
            "حول",
            "ضد",
            "خلال",
            "تحت",
            "فوق",
        ],
        POS.CONJ: ["و", "أو", "لكن", "ثم", "بل", "لا"],
        POS.PRON: ["أنا", "أنت", "هو", "هي", "نحن", "أنتم", "هم"],
        POS.AUX: ["كان", "كانت", "كانوا", "يكون", "تكون", "يجب", "يمكن"],
    },
    "tr": {
        # Turkish is agglutinative — these are base forms
        POS.CONJ: ["ve", "veya", "ama", "fakat", "lakin", "ya da", "hem"],
        POS.PRON: [
            "ben",
            "sen",
            "o",
            "biz",
            "siz",
            "onlar",
            "bu",
            "şu",
            "kim",
            "ne",
            "hangi",
            "kendim",
            "kendisi",
        ],
    },
    "hi": {
        POS.PREP: ["में", "पर", "को", "से", "के", "की", "का", "ने", "द्वारा"],
        POS.CONJ: ["और", "या", "लेकिन", "परंतु", "क्योंकि", "इसलिए"],
        POS.PRON: ["मैं", "तुम", "आप", "वह", "वो", "हम", "यह", "ये", "जो", "कौन"],
    },
    "zh": {
        POS.PREP: ["在", "从", "到", "向", "对", "给", "为", "比", "把", "被", "让"],
        POS.CONJ: ["和", "与", "或", "但", "而", "因为", "所以", "如果", "虽然"],
        POS.PRON: ["我", "你", "他", "她", "它", "我们", "你们", "他们", "谁", "什么"],
    },
    "ja": {
        POS.PART: [
            "は",
            "が",
            "を",
            "に",
            "で",
            "と",
            "も",
            "の",
            "へ",
            "から",
            "まで",
            "より",
            "だけ",
            "でも",
            "しか",
            "ほど",
            "ごろ",
            "ながら",
        ],
        POS.CONJ: ["そして", "しかし", "また", "だが", "けれど", "なのに", "だから"],
        POS.PRON: [
            "私",
            "僕",
            "俺",
            "あなた",
            "彼",
            "彼女",
            "私たち",
            "彼ら",
            "これ",
            "それ",
            "あれ",
            "ここ",
            "そこ",
            "あそこ",
            "だれ",
            "何",
        ],
    },
    "ko": {
        POS.PART: [
            "이",
            "가",
            "은",
            "는",
            "을",
            "를",
            "에",
            "에서",
            "으로",
            "로",
            "의",
            "도",
            "만",
            "와",
            "과",
            "이나",
            "나",
        ],
        POS.CONJ: ["그리고", "하지만", "또는", "또한", "그러나", "따라서", "그래서"],
        POS.PRON: [
            "나",
            "저",
            "너",
            "당신",
            "그",
            "그녀",
            "우리",
            "여러분",
            "이것",
            "그것",
            "저것",
            "누구",
            "무엇",
        ],
    },
}


#  Suffix rules for rule-based POS tagging

# Format: (pattern, pos_tag, features_dict, language_filter)
# language_filter=None means applies to all Latin-script languages
SUFFIX_RULES: list[tuple[re.Pattern, str, dict, Optional[str]]] = [
    # English
    (
        re.compile(r"tion$|sion$|ment$|ness$|ity$|ance$|ence$|ism$|ist$|er$|or$|ure$"),
        POS.NOUN,
        {},
        "en",
    ),
    (re.compile(r"ful$|less$|ous$|ive$|able$|ible$|al$|ary$|ory$|ic$"), POS.ADJ, {}, "en"),
    (re.compile(r"ly$"), POS.ADV, {"derivation": "adverb-forming"}, "en"),
    (re.compile(r"ing$"), POS.VERB, {"tense": "Pres", "aspect": "Prog"}, "en"),
    (re.compile(r"ed$"), POS.VERB, {"tense": "Past"}, "en"),
    (re.compile(r"en$"), POS.VERB, {"tense": "Past", "aspect": "Perf"}, "en"),
    (re.compile(r"ize$|ise$|ify$|ate$"), POS.VERB, {}, "en"),
    (re.compile(r"s$"), POS.NOUN, {"number": "Plur"}, "en"),  # weak signal
    # Spanish
    (re.compile(r"ción$|sión$|miento$|dad$|tad$|eza$|ura$"), POS.NOUN, {}, "es"),
    (re.compile(r"oso$|osa$|able$|ible$|al$|ar$"), POS.ADJ, {}, "es"),
    (re.compile(r"mente$"), POS.ADV, {"derivation": "adverb-forming"}, "es"),
    (re.compile(r"ar$|er$|ir$"), POS.VERB, {}, "es"),
    (re.compile(r"ando$|iendo$"), POS.VERB, {"tense": "Pres", "aspect": "Prog"}, "es"),
    (re.compile(r"ado$|ido$|ada$|ida$"), POS.VERB, {"tense": "Past", "aspect": "Perf"}, "es"),
    # Spanish conditional mood: hablaría, comerías, viviríamos
    # The conditional is formed by attaching endings directly to the infinitive,
    # so stripping these recovers the infinitive form.
    (
        re.compile(
            r"aría$|arías$|aríamos$|arían$|ería$|erías$|eríamos$|erían$|iría$|irías$|iríamos$|irían$"
        ),
        POS.VERB,
        {"tense": "Fut", "mood": "Cond"},
        "es",
    ),
    # Spanish past subjunctive: hablara, comieras, vivieran
    # Used in hypothetical and counterfactual clauses ("si pudieras…")
    (
        re.compile(
            r"ara$|aras$|áramos$|aran$|iera$|ieras$|iéramos$|ieran$|ase$|ases$|ásemos$|asen$|iese$|ieses$|iésemos$|iesen$"
        ),
        POS.VERB,
        {"tense": "Past", "mood": "Subj"},
        "es",
    ),
    # Spanish present subjunctive: hable, comas, vivan
    (re.compile(r"emos$|éis$"), POS.VERB, {"tense": "Pres", "mood": "Subj"}, "es"),
    # Spanish finite past-tense conjugations (preterite/imperfect).
    # Note: ía$ is tightened to [a-z]{3,}ía$ so it does NOT match "día" (noun,
    # only 1 char before "ía") while still catching "comía", "vivía", "habría".
    (
        re.compile(r"ó$|aron$|ieron$|aba$|ábamos$|aban$|amos$|asteis$|aste$|íamos$|ían$"),
        POS.VERB,
        {"tense": "Past"},
        "es",
    ),
    (re.compile(r"[a-z]{2,}ía$"), POS.VERB, {"tense": "Past"}, "es"),
    # French
    (re.compile(r"tion$|sion$|ment$|ité$|eur$|esse$|age$"), POS.NOUN, {}, "fr"),
    (re.compile(r"eux$|euse$|ive$|al$|able$|ible$"), POS.ADJ, {}, "fr"),
    (re.compile(r"ment$"), POS.ADV, {"derivation": "adverb-forming"}, "fr"),
    (re.compile(r"er$|ir$|re$|oir$"), POS.VERB, {}, "fr"),
    (re.compile(r"ant$"), POS.VERB, {"tense": "Pres", "aspect": "Prog"}, "fr"),
    # French conditional & imperfect: voudrais, parlerait, finirions
    (re.compile(r"ais$|ait$|aient$|ions$|iez$"), POS.VERB, {"tense": "Past", "mood": "Cond"}, "fr"),
    # French present indicative conjugations: parle, parles, parlent, finis, finit
    (re.compile(r"ons$|ez$|ent$"), POS.VERB, {"tense": "Pres"}, "fr"),
    # French past participles (e.g. "exploré", "abandonnée"). Weak signal:
    # collides with nouns that happen to end in -é/-ée (été, idée, santé) —
    # accepted approximation, same class of trade-off as the rules above.
    (re.compile(r"é$|ée$|és$|ées$"), POS.VERB, {"tense": "Past", "aspect": "Perf"}, "fr"),
    # German
    (re.compile(r"ung$|keit$|heit$|schaft$|nis$|tum$"), POS.NOUN, {"gender": "Fem"}, "de"),
    (re.compile(r"lich$|ig$|sam$|bar$|los$|voll$|haft$"), POS.ADJ, {}, "de"),
    (re.compile(r"lich$"), POS.ADV, {}, "de"),
    (re.compile(r"en$|eln$|ern$"), POS.VERB, {}, "de"),
    (re.compile(r"end$"), POS.VERB, {"tense": "Pres", "aspect": "Prog"}, "de"),
    # Turkish (agglutinative suffixes)
    (re.compile(r"lık$|lik$|luk$|lük$"), POS.NOUN, {"derivation": "nominalization"}, "tr"),
    (re.compile(r"cı$|ci$|cu$|cü$|çı$|çi$|çu$|çü$"), POS.NOUN, {"derivation": "agent"}, "tr"),
    (re.compile(r"lı$|li$|lu$|lü$"), POS.ADJ, {"derivation": "possession"}, "tr"),
    (re.compile(r"sız$|siz$|suz$|süz$"), POS.ADJ, {"derivation": "privative"}, "tr"),
    (re.compile(r"ca$|ce$"), POS.ADV, {}, "tr"),
    (re.compile(r"mak$|mek$"), POS.VERB, {"mood": "Inf"}, "tr"),
    (re.compile(r"yor$"), POS.VERB, {"tense": "Pres", "aspect": "Prog"}, "tr"),
    (re.compile(r"dı$|di$|du$|dü$|tı$|ti$|tu$|tü$"), POS.VERB, {"tense": "Past"}, "tr"),
    (re.compile(r"acak$|ecek$"), POS.VERB, {"tense": "Fut"}, "tr"),
    # Finnish (agglutinative)
    (re.compile(r"nen$|nen$"), POS.ADJ, {}, "fi"),
    (re.compile(r"sti$"), POS.ADV, {}, "fi"),
    (re.compile(r"ssa$|ssä$"), POS.NOUN, {"case": "Ine"}, "fi"),
    (re.compile(r"sta$|stä$"), POS.NOUN, {"case": "Ela"}, "fi"),
    (re.compile(r"lla$|llä$"), POS.NOUN, {"case": "Ade"}, "fi"),
    (re.compile(r"lta$|ltä$"), POS.NOUN, {"case": "Abl"}, "fi"),
    # Russian / Slavic endings
    (re.compile(r"ость$|ение$|ание$|ация$|ство$|ник$"), POS.NOUN, {}, "ru"),
    (re.compile(r"ный$|ная$|ное$|ные$|ской$|ская$"), POS.ADJ, {}, "ru"),
    (re.compile(r"но$|ly$"), POS.ADV, {}, "ru"),  # adverb '-но'
    # Arabic (imperfect, cannot easily apply rules to Arabic here, skip)
    # Italian
    (re.compile(r"zione$|mento$|tà$|tore$|trice$"), POS.NOUN, {}, "it"),
    (re.compile(r"oso$|osa$|ale$|ivo$|iva$"), POS.ADJ, {}, "it"),
    (re.compile(r"mente$"), POS.ADV, {"derivation": "adverb-forming"}, "it"),
    (re.compile(r"are$|ere$|ire$"), POS.VERB, {}, "it"),
    # Portuguese
    (re.compile(r"ção$|mento$|dade$|tude$"), POS.NOUN, {}, "pt"),
    (re.compile(r"oso$|osa$|al$|ável$|ível$"), POS.ADJ, {}, "pt"),
    (re.compile(r"mente$"), POS.ADV, {"derivation": "adverb-forming"}, "pt"),
    (re.compile(r"ar$|er$|ir$"), POS.VERB, {}, "pt"),
    # Common across Latin-script languages (weak signals, low priority)
    (re.compile(r"ist$|ism$|ology$"), POS.NOUN, {}, None),
]

# Prefix rules for derivational morphology
PREFIX_RULES: list[tuple[re.Pattern, str, dict, Optional[str]]] = [
    (re.compile(r"^un"), POS.X, {"derivation": "negation"}, "en"),
    (re.compile(r"^in(?=[aeiou])"), POS.X, {"derivation": "negation"}, "en"),
    (re.compile(r"^dis"), POS.X, {"derivation": "reversal"}, "en"),
    (re.compile(r"^re"), POS.X, {"derivation": "repetition"}, "en"),
    (re.compile(r"^pre"), POS.X, {"derivation": "before"}, "en"),
    (re.compile(r"^mis"), POS.X, {"derivation": "error"}, "en"),
    (re.compile(r"^over"), POS.X, {"derivation": "excess"}, "en"),
    (re.compile(r"^under"), POS.X, {"derivation": "deficiency"}, "en"),
    (re.compile(r"^super"), POS.X, {"derivation": "above"}, "en"),
    (re.compile(r"^inter"), POS.X, {"derivation": "between"}, "en"),
    (re.compile(r"^trans"), POS.X, {"derivation": "across"}, "en"),
    (re.compile(r"^anti"), POS.X, {"derivation": "against"}, "en"),
    (re.compile(r"^auto"), POS.X, {"derivation": "self"}, "en"),
    (re.compile(r"^non"), POS.X, {"derivation": "negation"}, "en"),
    (re.compile(r"^im(?=[bmp])"), POS.X, {"derivation": "negation"}, "en"),
    (re.compile(r"^il(?=l)"), POS.X, {"derivation": "negation"}, "en"),
    (re.compile(r"^ir(?=r)"), POS.X, {"derivation": "negation"}, "en"),
    # Spanish
    (re.compile(r"^des"), POS.X, {"derivation": "reversal"}, "es"),
    (re.compile(r"^in"), POS.X, {"derivation": "negation"}, "es"),
    (re.compile(r"^re"), POS.X, {"derivation": "repetition"}, "es"),
    # Turkish
    (re.compile(r"^ön"), POS.X, {"derivation": "before"}, "tr"),
]


#  Inflection suffix tables — for lexicon lookup recovery
#
#  Each entry is (inflected_suffix, [infinitive_endings_to_try]).
#  When an unknown word ends with the inflected suffix, strip it, then append
#  each candidate ending in order until one is found in the translation lexicon.
#
#  Ordered longest-first within each language so specific forms match before
#  short ones (e.g. "arías" before "ías" before "ia").
#
#  Minimum stem length is enforced by the caller (≥ 2 chars) to prevent short
#  words like "dia" matching the "ia" rule and producing spurious stems.
#
#  This table is the canonical source of truth — lexical_fallback.py reads
#  from here rather than maintaining its own language-specific copy.

INFLECTION_SUFFIXES: dict[str, tuple[tuple[str, list[str]], ...]] = {
    # Spanish
    # Conditional (hablaría, comerías, viviríamos) and past subjunctive
    # (hablara/hablase, comieras/comieses) all map back to -ar/-er/-ir infinitive.
    # Unaccented variants are included for users who type without diacritics.
    "es": (
        # Accented conditional / past-subjunctive (from corpus)
        ("aríamos", ["ar"]),
        ("eríamos", ["er"]),
        ("iríamos", ["ir"]),
        ("arían", ["ar"]),
        ("erían", ["er"]),
        ("irían", ["ir"]),
        ("arías", ["ar"]),
        ("erías", ["er"]),
        ("irías", ["ir"]),
        ("aría", ["ar"]),
        ("ería", ["er"]),
        ("iría", ["ir"]),
        ("iéramos", ["er", "ir"]),
        ("ieras", ["er", "ir"]),
        ("ieses", ["er", "ir"]),
        ("iera", ["er", "ir"]),
        ("iese", ["er", "ir"]),
        ("ías", ["ar", "er", "ir", ""]),
        ("ía", ["ar", "er", "ir", ""]),
        # Unaccented variants (user input without diacritics)
        ("ariamos", ["ar"]),
        ("eriamos", ["er"]),
        ("iriamos", ["ir"]),
        ("arian", ["ar"]),
        ("erian", ["er"]),
        ("irian", ["ir"]),
        ("arias", ["ar"]),
        ("erias", ["er"]),
        ("irias", ["ir"]),
        ("aria", ["ar"]),
        ("eria", ["er"]),
        ("iria", ["ir"]),
        ("ieramos", ["er", "ir"]),
        ("ias", ["ar", "er", "ir", ""]),
        ("ia", ["ar", "er", "ir", ""]),
    ),
    # French
    # Conditional (parlerais, finirais, prendrais) and imperfect (parlais, finissais).
    # Strip suffix → stem + infinitive ending recovers the infinitive.
    "fr": (
        ("erions", ["er"]),
        ("iriez", ["ir"]),
        ("rions", ["r", "re"]),
        ("erais", ["er"]),
        ("irais", ["ir"]),
        ("rais", ["r", "re"]),
        ("erait", ["er"]),
        ("irait", ["ir"]),
        ("rait", ["r", "re"]),
        ("eraient", ["er"]),
        ("iraient", ["ir"]),
        ("raient", ["r", "re"]),
        ("issais", ["ir"]),
        ("issait", ["ir"]),
        ("issaient", ["ir"]),
        ("ais", ["er", "ir", "re"]),
        ("ait", ["er", "ir", "re"]),
    ),
    # German
    # Konjunktiv II (hätte, wäre, käme) — these are suppletive/irregular so
    # coverage is limited; regular verbs use würde + infinitive in practice.
    # Konjunktiv I (gehe, komme) maps cleanly to infinitive stem.
    "de": (
        ("ten", ["ten", "t"]),  # schritten → schritten (weak signal)
        ("test", ["ten"]),  # gingest → gingen (irregular, low coverage)
        ("te", ["ten", "t"]),  # liebte  → lieben
        ("st", ["en"]),  # gehst   → gehen (2nd sg present)
    ),
    # Portuguese
    "pt": (
        ("aríamos", ["ar"]),
        ("eríamos", ["er"]),
        ("iríamos", ["ir"]),
        ("ariam", ["ar"]),
        ("eriam", ["er"]),
        ("iriam", ["ir"]),
        ("arias", ["ar"]),
        ("erias", ["er"]),
        ("irias", ["ir"]),
        ("aria", ["ar"]),
        ("eria", ["er"]),
        ("iria", ["ir"]),
        ("asse", ["ar"]),
        ("esse", ["er", "ir"]),
        ("ias", ["ar", "er", "ir"]),
        ("ia", ["ar", "er", "ir"]),
    ),
    # Italian
    "it": (
        ("eremmo", ["ere"]),
        ("iremmo", ["ire"]),
        ("remmo", ["re"]),
        ("ereste", ["ere"]),
        ("ireste", ["ire"]),
        ("reste", ["re"]),
        ("erebbe", ["ere"]),
        ("irebbe", ["ire"]),
        ("rebbe", ["re"]),
        ("erebbero", ["ere"]),
        ("irebbero", ["ire"]),
        ("rebbero", ["re"]),
        ("erei", ["ere"]),
        ("irei", ["ire"]),
        ("rei", ["re"]),
        ("assi", ["are"]),
        ("essi", ["ere", "ire"]),
        ("asse", ["are"]),
        ("esse", ["ere", "ire"]),
    ),
}


#  BPE Segmenter


class BPESegmenter:
    """Statistical subword segmenter trained using adjacent-pair frequencies.

    BPE pieces are computational units, not verified linguistic morphemes.
    """

    def __init__(self, vocab_size: int = 4000) -> None:
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []  # ordered merge rules
        self.vocab: set[str] = set()

    def fit(self, sentences: list[str], min_freq: int = 2) -> None:
        """Learn BPE merge rules from a list of sentences."""
        # Initialize: split each word into characters + end-of-word marker
        word_freqs: Counter = Counter()
        for sent in sentences:
            for word in re.findall(r"\b\w+\b", sent.lower()):
                word_freqs[word] += 1

        # Represent each word as a tuple of characters (with </w> end marker)
        vocab: dict[tuple, int] = {}
        for word, freq in word_freqs.items():
            if freq >= min_freq:
                chars = tuple(list(word) + ["</w>"])
                vocab[chars] = freq

        # Collect initial character vocabulary
        all_symbols: set[str] = set()
        for word_tuple in vocab:
            all_symbols.update(word_tuple)
        self.vocab = all_symbols

        target = self.vocab_size - len(all_symbols)

        for _ in range(max(0, target)):
            pairs = self._get_pair_freqs(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=lambda p: pairs[p])
            if pairs[best_pair] < min_freq:
                break
            vocab = self._merge_pair(vocab, best_pair)
            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])

    def segment(self, word: str) -> list[str]:
        """Segment a word into BPE pieces."""
        if not self.merges:
            return list(word)

        tokens = list(word) + ["</w>"]

        for merge_a, merge_b in self.merges:
            i = 0
            new_tokens: list[str] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge_a and tokens[i + 1] == merge_b:
                    new_tokens.append(merge_a + merge_b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Remove end-of-word marker
        return [t.replace("</w>", "") for t in tokens if t != "</w>"]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges, "vocab": list(self.vocab)}))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab = set(data["vocab"])

    @staticmethod
    def _get_pair_freqs(vocab: dict) -> Counter:
        pairs: Counter = Counter()
        for word_tuple, freq in vocab.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(vocab: dict, pair: tuple) -> dict:
        a, b = pair
        new_vocab: dict = {}
        for word_tuple, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == a and word_tuple[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab


#  POS Tagger


class POSTagger:
    """Rule-based + statistical POS tagger.

    Uses a language-specific bigram HMM when trained counts are available.
    Otherwise falls back to closed-class, suffix, capitalization, and noun
    heuristics. Neither path supplies calibrated confidence.
    """

    def __init__(self) -> None:
        # Bigram transition counts: (prev_pos, pos) → count
        self._transitions: Counter = Counter()
        self._emission: dict[str, Counter] = defaultdict(Counter)
        self._models = {"en": (self._transitions, self._emission)}
        self._decoding = {}
        self._unknown_models = {}
        self._calibrations = {}
        self._model_sources = {}
        self._model_fingerprints = {}

    def has_model(self, lang: str) -> bool:
        counts = self._models.get(language_code(lang))
        return bool(counts and counts[0] and counts[1])

    def load_corpus_model(self, lexicon: CorpusLexicon) -> None:
        self._models[lexicon.lang] = lexicon.pos_statistics()
        self._decoding.pop(lexicon.lang, None)
        self._unknown_models.pop(lexicon.lang, None)
        self._calibrations.pop(lexicon.lang, None)
        self._model_sources[lexicon.lang] = lexicon.source["sha256"]
        self._model_fingerprints.pop(lexicon.lang, None)

    def model_fingerprint(self, lang):
        code = language_code(lang)
        if code not in self._model_fingerprints:
            self._model_fingerprints[code] = pos_model_fingerprint(*self._models[code])
        return self._model_fingerprints[code]

    def add_calibration(self, calibration):
        if (
            self._model_sources.get(calibration.language) != calibration.training_sha256
            or self.model_fingerprint(calibration.language) != calibration.model_sha256
        ):
            raise ValueError("Calibration does not match the loaded training model")
        self._calibrations[calibration.language] = calibration

    def load_calibrations(self, directory):
        directory = Path(directory)
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise ValueError("Calibration path must be a directory")
        loaded = {}
        for path in sorted(directory.glob("*.json")):
            calibration = POSCalibration.load(path)
            if (
                calibration.language in loaded
                or self._model_sources.get(calibration.language) != calibration.training_sha256
                or self.model_fingerprint(calibration.language) != calibration.model_sha256
            ):
                raise ValueError("Duplicate or mismatched POS calibration")
            loaded[calibration.language] = calibration
        self._calibrations.update(loaded)
        return sorted(loaded)

    def tag(self, tokens: list[str], lang: str = "en", ranges=None) -> list[tuple[str, str]]:
        """Tag a token sequence. Returns list of (token, pos_tag)."""
        if tokens and self.has_model(lang):
            # Reset context at sentence-final punctuation rather than carrying
            # an unrelated previous sentence into the next prediction.
            ranges = (
                validate_ranges(ranges, len(tokens))
                if ranges is not None
                else sentence_ranges(tokens)
            )
            result = []
            for left, right in ranges:
                sentence = tokens[left:right]
                result.extend(zip(sentence, self._decode(sentence, lang)))
            return result
        result: list[tuple[str, str]] = []
        prev_pos = "BOS"
        base_lang = lang.lower().split("-")[0]
        closed = CLOSED_CLASS.get(base_lang, {})

        for i, token in enumerate(tokens):
            lower = token.lower()

            # Layer 1: closed class lookup
            pos = self._lookup_closed(lower, closed)

            # Layer 2: suffix/prefix rules
            if pos is None:
                pos = self._apply_suffix_rules(lower, base_lang)

            # Layer 4: capitalisation heuristic (proper noun)
            if pos is None and i > 0 and token and token[0].isupper():
                pos = POS.PROPN

            # Layer 5: default
            if pos is None:
                pos = POS.NOUN

            result.append((token, pos))
            prev_pos = pos

        return result

    def fit(self, tagged_sentences: list[list[tuple[str, str]]], lang: str = "en") -> None:
        """Train bigram transition and emission from tagged data."""
        code = language_code(lang)
        self._decoding.pop(code, None)
        self._unknown_models.pop(code, None)
        self._calibrations.pop(code, None)
        self._model_sources.pop(code, None)
        self._model_fingerprints.pop(code, None)
        if code not in self._models:
            self._models[code] = (Counter(), defaultdict(Counter))
        transitions, emissions = self._models[code]
        for sent in tagged_sentences:
            if not sent:
                continue
            prev = "BOS"
            for token, pos in sent:
                transitions[(prev, pos)] += 1
                emissions[normalized_form(token)][pos] += 1
                prev = pos
            transitions[(prev, "EOS")] += 1

    def tag_with_uncertainty(self, tokens: list[str], lang: str = "en", ranges=None):
        tagged = self.tag(tokens, lang, ranges)
        if not tokens or not self.has_model(lang):
            return [(token, pos, None) for token, pos in tagged]
        distributions = []
        ranges = (
            validate_ranges(ranges, len(tokens)) if ranges is not None else sentence_ranges(tokens)
        )
        for start, end in ranges:
            lattice, transitions = self._lattice(tokens[start:end], lang)
            distributions.extend(forward_backward(lattice, transitions))
        emissions = self._models[language_code(lang)][1]
        result = [
            (
                token,
                pos,
                {
                    "method": "forward_backward",
                    "status": "uncalibrated",
                    "selected_pos": pos,
                    "model_probability": probabilities[pos],
                    "known_form": normalized_form(token) in emissions,
                    "candidate_count": len(probabilities),
                    "distribution": probabilities,
                    "entropy_nats": -sum(p * math.log(p) for p in probabilities.values() if p > 0),
                },
            )
            for (token, pos), probabilities in zip(tagged, distributions)
        ]
        calibration = self._calibrations.get(language_code(lang))
        if calibration:
            for _, _, evidence in result:
                estimate = calibration.estimate(
                    evidence["model_probability"], evidence["known_form"]
                )
                if estimate is not None:
                    evidence["calibration"] = estimate
                    evidence["status"] = "development_calibrated"
        return result

    def _lattice(self, tokens: list[str], lang: str):
        """Build the shared log-score lattice for Viterbi and forward–backward.

        Observed forms restrict candidate POS tags; unknown forms may take any
        learned tag. Scores are model predictions, not calibrated confidence.
        """
        code = language_code(lang)
        transitions, emissions = self._models[code]
        if code not in self._decoding:
            totals = Counter()
            outgoing = Counter()
            for counts in emissions.values():
                totals.update(counts)
            for (left, _), count in transitions.items():
                outgoing[left] += count
            tags = sorted(totals)
            vocabulary = len(emissions) + 1
            transition_log = {
                (left, right): math.log(
                    (transitions[(left, right)] + 1) / (outgoing[left] + len(tags) + 1)
                )
                for left in ["BOS", *tags]
                for right in [*tags, "EOS"]
            }
            self._decoding[code] = (totals, tags, vocabulary, transition_log)
            self._unknown_models[code] = UnknownWordModel(emissions, language=code)
        totals, tags, vocabulary, transition_log = self._decoding[code]
        lattice = []
        for token in tokens:
            observed = emissions.get(normalized_form(token), {})
            if observed:
                choices = sorted(observed)
                hint = None
            elif any(char.isalpha() for char in token):
                # Unseen lexical words must not become punctuation just because
                # punctuation has a favorable smoothed emission denominator.
                choices = [tag for tag in tags if tag not in {POS.PUNCT, POS.SYM}]
                choices = choices or tags
                hint = None
            else:
                structural = {POS.NUM} if token.isdecimal() else {POS.PUNCT, POS.SYM}
                choices = [tag for tag in tags if tag in structural]
                choices = choices or tags
                hint = None
            spelling = (
                self._unknown_models[code].scores(token, choices)
                if not observed and any(c.isalpha() for c in token)
                else {}
            )
            lattice.append(
                {
                    tag: (
                        math.log((observed[tag] + 1) / (totals[tag] + vocabulary))
                        if observed
                        else -math.log(vocabulary) + spelling.get(tag, 0.0)
                    )
                    for tag in choices
                }
            )
        return lattice, transition_log

    def _decode(self, tokens: list[str], lang: str) -> list[str]:
        lattice, transition_log = self._lattice(tokens, lang)
        scores = {"BOS": 0.0}
        backpointers = []
        for emissions in lattice:
            next_scores = {}
            previous_tags = {}
            for tag, emission in emissions.items():
                prior = max(scores, key=lambda prev: scores[prev] + transition_log[(prev, tag)])
                next_scores[tag] = scores[prior] + transition_log[(prior, tag)] + emission
                previous_tags[tag] = prior
            backpointers.append(previous_tags)
            scores = next_scores
        best = max(scores, key=lambda tag: scores[tag] + transition_log[(tag, "EOS")])
        sequence = []
        for previous_tags in reversed(backpointers):
            sequence.append(best)
            best = previous_tags[best]
        return list(reversed(sequence))

    @staticmethod
    def _lookup_closed(token: str, closed: dict) -> Optional[str]:
        for pos_tag, words in closed.items():
            if token in words:
                return pos_tag
        return None

    @staticmethod
    def _apply_suffix_rules(token: str, lang: str) -> Optional[str]:
        if len(token) < 3:
            return None
        for pattern, pos_tag, _, lang_filter in SUFFIX_RULES:
            if lang_filter is not None and lang_filter != lang:
                continue
            if pattern.search(token):
                return pos_tag
        return None


#  Morpheme extraction (English-focused, extensible)

# English prefix → (stripped_length, derivation_label)
EN_PREFIXES: list[tuple[str, str]] = [
    ("un", "negation"),
    ("in", "negation"),
    ("im", "negation"),
    ("il", "negation"),
    ("ir", "negation"),
    ("dis", "reversal"),
    ("mis", "error"),
    ("re", "repetition"),
    ("pre", "before"),
    ("pro", "for"),
    ("anti", "against"),
    ("over", "excess"),
    ("under", "deficiency"),
    ("super", "above"),
    ("trans", "across"),
    ("inter", "between"),
    ("sub", "under"),
    ("auto", "self"),
    ("non", "negation"),
    ("co", "together"),
    ("counter", "against"),
    ("de", "reversal"),
    ("ex", "former"),
    ("out", "surpass"),
    ("semi", "half"),
    ("multi", "many"),
    ("micro", "small"),
    ("macro", "large"),
    ("post", "after"),
    ("neo", "new"),
]

# English suffix → (stripped_length, pos_tag, derivation_label)
EN_SUFFIXES: list[tuple[str, str, str]] = [
    # Noun-forming
    ("tion", POS.NOUN, "nominalization"),
    ("sion", POS.NOUN, "nominalization"),
    ("ment", POS.NOUN, "nominalization"),
    ("ness", POS.NOUN, "nominalization"),
    ("ity", POS.NOUN, "nominalization"),
    ("ance", POS.NOUN, "nominalization"),
    ("ence", POS.NOUN, "nominalization"),
    ("ism", POS.NOUN, "ideology"),
    ("ist", POS.NOUN, "agent"),
    ("er", POS.NOUN, "agent"),
    ("or", POS.NOUN, "agent"),
    ("ure", POS.NOUN, "nominalization"),
    ("age", POS.NOUN, "nominalization"),
    ("ry", POS.NOUN, "collection"),
    ("ery", POS.NOUN, "place/state"),
    ("ship", POS.NOUN, "state"),
    ("hood", POS.NOUN, "state"),
    ("dom", POS.NOUN, "domain"),
    # Adjective-forming
    ("ful", POS.ADJ, "having"),
    ("less", POS.ADJ, "without"),
    ("ous", POS.ADJ, "having"),
    ("ive", POS.ADJ, "tending-to"),
    ("able", POS.ADJ, "capable-of"),
    ("ible", POS.ADJ, "capable-of"),
    ("al", POS.ADJ, "relating-to"),
    ("ary", POS.ADJ, "relating-to"),
    ("ic", POS.ADJ, "relating-to"),
    ("ical", POS.ADJ, "relating-to"),
    ("ish", POS.ADJ, "somewhat"),
    ("esque", POS.ADJ, "like"),
    ("like", POS.ADJ, "resembling"),
    ("ward", POS.ADJ, "direction"),
    ("en", POS.ADJ, "made-of"),
    # Adverb-forming
    ("ly", POS.ADV, "adverb-forming"),
    ("wise", POS.ADV, "manner"),
    ("wards", POS.ADV, "direction"),
    # Verb-forming
    ("ize", POS.VERB, "causative"),
    ("ise", POS.VERB, "causative"),
    ("ify", POS.VERB, "causative"),
    ("ate", POS.VERB, "causative"),
    ("en", POS.VERB, "causative"),
    # Tense markers
    ("ing", POS.VERB, "progressive"),
    ("ed", POS.VERB, "past"),
    # Comparative/superlative
    ("er", POS.ADJ, "comparative"),
    ("est", POS.ADJ, "superlative"),
    # Plural
    ("s", POS.NOUN, "plural"),
    ("es", POS.NOUN, "plural"),
]

# Words where a prefix/suffix pattern superficially matches but isn't
# actually a real morpheme boundary synchronically — mostly Latinate
# borrowings that read as simplex words in modern English even though
# their etymology decomposes (e.g. "explore" is historically "ex-" +
# Latin "plorare", but "plore" isn't an English word, so blind
# prefix-stripping mangles it into nonsense: "explored" -> "plor").
# Checked before any stripping is attempted; these words are returned
# undecomposed. This is a curated stop-list rather than a general
# stem-plausibility check (no dictionary is available to test "is this
# remaining stem a real word") — extend it as new false positives turn up.
EN_OPAQUE_WORDS: frozenset[str] = frozenset(
    {
        # false "ex-" prefix splits
        "explore",
        "explored",
        "exploring",
        "explorer",
        "explorers",
        "exist",
        "existed",
        "existing",
        "exists",
        "exact",
        "exactly",
        "example",
        "examples",
        "exercise",
        "exercised",
        "exercising",
        "exercises",
        "expand",
        "expanded",
        "expanding",
        "expands",
        "experience",
        "experienced",
        "experiencing",
        "experiences",
        "express",
        "expressed",
        "expressing",
        "expresses",
        "explain",
        "explained",
        "explaining",
        "explains",
        "expensive",
        "extreme",
        "extremely",
        "excellent",
        # false "-ous" suffix splits (Latinate adjectives with no free root)
        "curious",
        "serious",
        "obvious",
        "previous",
        "various",
        "anxious",
        "nervous",
        "famous",
        "jealous",
        "generous",
        "dangerous",
        "enormous",
        "tremendous",
        "conscious",
        "gracious",
        "spacious",
        "religious",
        "ridiculous",
        "courageous",
        "outrageous",
        "marvelous",
        "glorious",
        "mysterious",
        "numerous",
        "fabulous",
        "hideous",
        "hilarious",
        "delicious",
        "suspicious",
        "ambitious",
        "spontaneous",
        "simultaneous",
        "courteous",
        "gorgeous",
        "righteous",
    }
)


def _extract_en_morphemes(word: str) -> tuple[str, list[str], MorphFeatures]:
    """Extract English morphemes using ordered prefix + suffix stripping."""
    lower = word.lower()
    if lower in EN_OPAQUE_WORDS:
        return lower, [lower], MorphFeatures()
    root = lower
    parts = []
    feats = MorphFeatures()

    # Strip known prefix
    prefix_stripped = None
    for pref, deriv in sorted(EN_PREFIXES, key=lambda x: -len(x[0])):
        if lower.startswith(pref) and len(lower) > len(pref) + 2:
            prefix_stripped = pref
            root = lower[len(pref) :]
            parts.append(pref)
            feats.derivation = deriv
            break

    # Strip known suffix (longest match first)
    suffix_stripped = None
    remaining = root
    for suf, pos_tag, deriv in sorted(EN_SUFFIXES, key=lambda x: -len(x[0])):
        if remaining.endswith(suf) and len(remaining) > len(suf) + 1:
            suffix_stripped = suf
            stem = remaining[: -len(suf)]
            # Handle common spelling changes (e.g., "believ" → "believe")
            root = _restore_stem(stem)
            if feats.derivation is None:
                feats.derivation = deriv
            break

    # Build morpheme list
    if suffix_stripped:
        parts.append(root)
        parts.append(suffix_stripped)
    else:
        parts = parts + [remaining]

    return root, parts if parts else [lower], feats


def _restore_stem(stem: str) -> str:
    """Attempt to restore the original root from a stripped stem."""
    # Common English stem repairs
    if stem.endswith("i"):
        return stem[:-1] + "y"  # happily → happi → happy
    if re.search(r"([bcdfghjklmnpqrstvwxyz])\1$", stem):
        return stem[:-1]  # running → runn → run
    return stem


#  Language-type classifiers

# Languages by morphological typology
AGGLUTINATIVE = {
    "tr",
    "fi",
    "hu",
    "et",
    "lv",
    "lt",
    "ko",
    "sw",
    "eu",
    "uz",
    "kk",
    "az",
    "ky",
    "tg",
    "my",
    "ka",
    "am",
    "yo",
    "si",
    "km",
    "lo",
}
FUSIONAL = {
    "ru",
    "uk",
    "be",
    "pl",
    "cs",
    "sk",
    "sl",
    "hr",
    "sr",
    "bg",
    "mk",
    "de",
    "nl",
    "sv",
    "da",
    "nb",
    "nn",
    "el",
    "la",
    "ar",
    "he",
    "es",
    "fr",
    "it",
    "pt",
    "ro",
    "ca",
}
ISOLATING = {"zh", "vi", "th", "lo", "ja", "id", "ms", "tl"}
ANALYTIC = {"en", "af"}  # English has mostly lost inflection


def morphological_typology(lang: str) -> str:
    """Return the morphological type of a language."""
    code = lang.lower().split("-")[0]
    if code in AGGLUTINATIVE:
        return "agglutinative"
    if code in FUSIONAL:
        return "fusional"
    if code in ISOLATING:
        return "isolating"
    if code in ANALYTIC:
        return "analytic"
    return "unknown"


#  MorphologyEngine


class MorphologyEngine:
    """Full morphological analysis pipeline.

    - Tokenises text
    - Applies POS tagging
    - Proposes English morphemes using heuristic rules; reports BPE separately
    - Extracts morphological features
    - Reports language typology

    Usage::

        engine = MorphologyEngine()
        result = engine.analyze("The cats are running quickly.", lang="en")
        for a in result.analyses:
            print(a.surface, a.root, a.morphemes, a.pos)
    """

    def __init__(self) -> None:
        self.tagger = POSTagger()
        self._segmenters: dict[str, BPESegmenter] = {}
        self._lexicons: dict[str, CorpusLexicon] = {}
        self.dependency_parsers = {}
        self._sentence_abbreviations = {}

    def analyze(self, text: str, lang: str = "en") -> TokenAnalysis:
        """Analyze all words in a text."""
        pieces = self._tokenize_with_spans(text, lang)
        tokens = [surface for surface, _ in pieces]
        ranges = sentence_ranges(
            tokens,
            [span for _, span in pieces],
            text,
            self._sentence_abbreviations.get(language_code(lang), ()),
        )
        result = self.analyze_tokens(tokens, lang, ranges=ranges)
        result.text = text
        result.sentence_boundary_method = "source_heuristics"
        for analysis, (_, span) in zip(result.analyses, pieces):
            analysis.span = span
            if analysis.pos_uncertainty is not None:
                analysis.pos_uncertainty["sentence_boundary_method"] = (
                    result.sentence_boundary_method
                )
        return result

    def analyze_tokens(self, tokens: list[str], lang: str = "en", ranges=None) -> TokenAnalysis:
        """Analyze one upstream sentence; explicit ranges support multiple.

        Respect supplied sentence boundaries, including internal punctuation.
        Original-text offsets are unavailable for these supplied tokens.
        """
        if any(not token or not token.strip() for token in tokens):
            raise ValueError("tokens must contain non-whitespace forms")
        ranges = (
            ([(0, len(tokens))] if tokens else [])
            if ranges is None
            else validate_ranges(ranges, len(tokens))
        )
        tagged = self.tagger.tag_with_uncertainty(tokens, lang, ranges)
        contextual = len(tokens) > 1 and self.tagger.has_model(lang)
        analyses = []
        for tok, pos, evidence in tagged:
            analysis = self._analyze_token(tok, pos, lang, contextual)
            if evidence is not None:
                evidence["applies_to_returned_pos"] = analysis.pos == pos
                analysis.pos_uncertainty = evidence
            analyses.append(analysis)
        return TokenAnalysis(
            tokens=list(tokens), analyses=analyses, lang=lang, sentence_ranges=list(ranges)
        )

    def analyze_word(self, word: str, lang: str = "en") -> MorphAnalysis:
        """Analyze a single word."""
        if not word or not word.strip():
            raise ValueError("word must contain a non-whitespace token")
        _, pos, evidence = self.tagger.tag_with_uncertainty([word], lang)[0]
        result = self._analyze_token(word, pos, lang)
        if evidence is not None:
            evidence["applies_to_returned_pos"] = result.pos == pos
            result.pos_uncertainty = evidence
        result.span = (0, len(word))
        return result

    def get_typology(self, lang: str) -> str:
        """Return morphological typology for a language."""
        return morphological_typology(lang)

    def add_segmenter(self, lang: str, segmenter: BPESegmenter) -> None:
        """Register a trained BPE segmenter for a language."""
        self._segmenters[lang.lower().split("-")[0]] = segmenter

    def add_lexicon(self, lexicon: CorpusLexicon) -> None:
        """Register an attested form index; unknown forms retain explicit fallback."""
        self._lexicons[lexicon.lang] = lexicon
        self.dependency_parsers.pop(lexicon.lang, None)
        self.tagger.load_corpus_model(lexicon)
        self._sentence_abbreviations[lexicon.lang] = self._abbreviations(lexicon)

    def load_resources(self, directory: str | Path) -> list[dict]:
        """Load local compiled resources atomically; report corrupt/duplicate data."""
        directory = Path(directory)
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise ValueError("Linguistic resource path must be a directory")
        loaded = {}
        for path in sorted(directory.glob("*.json")):
            try:
                lexicon = CorpusLexicon.load(path)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid linguistic resource {path.name}: {error}") from error
            if lexicon.lang in loaded:
                raise ValueError(f"Duplicate resource language: {lexicon.lang}")
            loaded[lexicon.lang] = lexicon
        self._lexicons.update(loaded)
        for lexicon in loaded.values():
            self.dependency_parsers.pop(lexicon.lang, None)
            self.tagger.load_corpus_model(lexicon)
            self._sentence_abbreviations[lexicon.lang] = self._abbreviations(lexicon)
        return [
            {"language": lexicon.lang, "source": lexicon.source, "forms": lexicon.summary["forms"]}
            for lexicon in loaded.values()
        ]

    @staticmethod
    def _abbreviations(lexicon):
        _, emissions = lexicon.pos_statistics()
        return {
            form[:-1].lower()
            for form, counts in emissions.items()
            if form.endswith(".")
            and 1 < len(form) <= 8
            and form[:-1].isalpha()
            and any(tag not in {"PUNCT", "SYM"} for tag in counts)
        }

    def capabilities(self, lang: str) -> dict:
        """Report which analysis methods are available for each loaded language."""
        base_lang = lang.lower().split("-")[0]
        return {
            "morphology": (
                "corpus_lookup"
                if base_lang in self._lexicons
                else "english_affix_rules"
                if base_lang == "en"
                else "unavailable"
            ),
            "pos": (
                "corpus_lookup_and_heuristic"
                if base_lang in self._lexicons
                else "heuristic"
                if base_lang in CLOSED_CLASS
                else "default_heuristic"
            ),
            "subwords": "bpe" if base_lang in self._segmenters else "unavailable",
            "tokenization": "characters"
            if base_lang in ("zh", "ja")
            else "english_clitic_rules"
            if base_lang == "en"
            else "punctuation_split",
            "corpus_forms": self._lexicons[base_lang].summary["forms"]
            if base_lang in self._lexicons
            else 0,
            "context": "bigram_hmm" if self.tagger.has_model(lang) else "unavailable",
            "unknown_word_pos": "locally_trained_case_variants_suffix_backoff"
            if self.tagger.has_model(lang)
            else "heuristic",
        }

    # Internal

    def _tokenize(self, text: str, lang: str) -> list[str]:
        return [surface for surface, _ in self._tokenize_with_spans(text, lang)]

    def tokenize_with_spans(self, text: str, lang: str) -> list[tuple[str, tuple[int, int]]]:
        """Reuse production text boundaries without invoking POS/morphology models."""
        return self._tokenize_with_spans(text, lang)

    def _tokenize_with_spans(self, text: str, lang: str) -> list[tuple[str, tuple[int, int]]]:
        """Preserve combining marks and exact original-text offsets.

        CJK remains character-oriented (base character plus marks), not a
        claimed word segmenter. Other scripts group letters, numbers, and marks.
        """
        cjk = lang.lower().split("-")[0] in ("zh", "ja")
        result = []
        start = None
        for i, char in enumerate(text):
            category = unicodedata.category(char)
            mark = category.startswith("M")
            word = category[0] in "LN" or category == "Pc"
            # Keep an internal apostrophe, but not a surrounding quotation mark.
            apostrophe = (
                char in "'’"
                and start is not None
                and i + 1 < len(text)
                and unicodedata.category(text[i + 1])[0] in "LN"
            )
            joiner = (
                char in "\u200c\u200d"
                and start is not None
                and i + 1 < len(text)
                and unicodedata.category(text[i + 1])[0] in "LM"
            )
            if start is not None and (mark or (not cjk and (word or apostrophe or joiner))):
                continue
            if start is not None:
                result.append((text[start:i], (start, i)))
                start = None
            if char.isspace():
                continue
            if word or mark:
                start = i
            else:
                result.append((char, (i, i + 1)))
        if start is not None:
            result.append((text[start:], (start, len(text))))
        if language_code(lang) == "en":
            # English syntactic-word convention, not universal morphology.
            # Keep each piece as an exact surface slice; do not expand can't
            # into invented text such as 'can not'. Meanings remain ambiguous.
            pieces = []
            for surface, (left, right) in result:
                remaining = surface
                suffixes = []
                while True:
                    match = re.search(
                        r"(?:n['’]t|['’](?:s|re|ve|ll|d|m))$", remaining, re.IGNORECASE
                    )
                    if match is None or match.start() == 0:
                        break
                    suffixes.append(
                        (remaining[match.start() :], (left + match.start(), left + len(remaining)))
                    )
                    remaining = remaining[: match.start()]
                pieces.append((remaining, (left, left + len(remaining))))
                pieces.extend(reversed(suffixes))
            result = pieces
        return result

    def _analyze_token(
        self, surface: str, pos: str, lang: str, contextual: bool = False
    ) -> MorphAnalysis:
        base_lang = lang.lower().split("-")[0]

        subwords = (
            self._segmenters[base_lang].segment(surface.lower())
            if base_lang in self._segmenters
            else []
        )
        lexicon = self._lexicons.get(base_lang)
        pos_method = "bigram_hmm" if self.tagger.has_model(lang) else "heuristic_rules"
        model_source = dict(lexicon.source) if lexicon and pos_method == "bigram_hmm" else None
        candidates = lexicon.lookup(surface) if lexicon else []
        if candidates:
            selected = (
                [i for i, c in enumerate(candidates) if c["pos"] == pos] if contextual else []
            )
            narrowed = bool(selected and len(selected) < len(candidates))
            if not selected:
                selected = list(range(len(candidates)))
            active = [candidates[i] for i in selected]
            lemmas = {c["lemma"] for c in active}
            tags = {c["pos"] for c in active}
            # Preserve field-level consensus even when the complete analysis is
            # ambiguous. Never fill a disputed field from the frequency winner.
            shared_features = {
                key: value
                for key, value in active[0]["features"].items()
                if all(c["features"].get(key) == value for c in active)
            }
            features = MorphFeatures(
                **{
                    key.lower(): value
                    for key, value in shared_features.items()
                    if key.lower() in MorphFeatures.__dataclass_fields__
                }
            )
            return MorphAnalysis(
                surface,
                next(iter(lemmas)) if len(lemmas) == 1 else surface,
                [],
                next(iter(tags)) if len(tags) == 1 else POS.X,
                features,
                lang,
                method="corpus_context"
                if narrowed
                else "corpus_lookup"
                if len(candidates) == 1
                else "corpus_ambiguous",
                candidates=candidates,
                ud_features=shared_features,
                subwords=subwords,
                selected_candidate_indices=selected,
                pos_method="bigram_hmm"
                if narrowed
                else "corpus_lookup"
                if len(tags) == 1
                else "unresolved",
                model_source=model_source,
            )

        # Punctuation / numbers — no morpheme analysis needed
        if not any(char.isalpha() for char in surface):
            return MorphAnalysis(
                surface=surface,
                root=surface,
                morphemes=[surface],
                pos=POS.NUM if surface[0].isdigit() else POS.PUNCT,
                features=MorphFeatures(),
                lang=lang,
                method="structural",
                pos_method="structural",
            )

        # Statistical subwords are independent of linguistic morphology.

        # Rule-based for English
        if base_lang == "en" and lexicon is None:
            root, morphemes, feats = _extract_en_morphemes(surface)
            return MorphAnalysis(
                surface=surface,
                root=root,
                morphemes=morphemes,
                pos=pos,
                features=feats,
                lang=lang,
                method="english_affix_rules",
                subwords=subwords,
                pos_method=pos_method,
                model_source=model_source,
            )

        # No analyzer for this language, so no morpheme boundaries are reported.
        return MorphAnalysis(
            surface=surface,
            root=surface,
            morphemes=[],
            pos=pos,
            features=MorphFeatures(),
            lang=lang,
            subwords=subwords,
            pos_method=pos_method,
            model_source=model_source,
        )
