"""
speech_engine/grammar.py — Grammar checking and sentence structure analysis.

Built from scratch: no language tool libraries, no grammarly APIs.

Architecture:
  - Language-agnostic rule framework (add rules per language)
  - English rule set: 25+ rules covering the most common error categories
  - Extensible: subclass GrammarRule for any language
  - Sentence structure labelling (basic phrase chunking)

Rule categories for English:
  - Subject-verb agreement (he run → he runs)
  - Article usage (a apple → an apple, a/the ambiguity)
  - Double negation (I don't have nothing → I don't have anything)
  - Common confusions (their/there/they're, your/you're, its/it's,
                       affect/effect, then/than, to/too/two)
  - Comma splices and run-on detection
  - Redundant phrases (end result → result, past history → history)
  - Wordy constructions (in order to → to, due to the fact that → because)
  - Spelling patterns (common misspellings)
  - Punctuation (missing period, multiple exclamation marks)

Usage::

    checker = GrammarChecker()
    result = checker.check("He don't have nothing to eat.", lang="en")
    for err in result.errors:
        print(err.message, "→", err.suggestion)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Error severity levels
# ══════════════════════════════════════════════════════════════════════════════

class Severity:
    ERROR   = "error"    # almost certainly wrong
    WARNING = "warning"  # likely wrong but context-dependent
    STYLE   = "style"    # not incorrect, but could be clearer/shorter


# ══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GrammarError:
    message:    str
    suggestion: Optional[str]
    span:       tuple[int, int]   # character offsets in original text
    severity:   str               # Severity.*
    rule_id:    str               # machine-readable rule identifier
    category:   str               # "agreement", "confusion", "style", etc.


@dataclass
class GrammarResult:
    text:      str
    lang:      str
    errors:    list[GrammarError]
    score:     Optional[float]  # Rule penalty; None when a check failed.
    n_words:   int
    checked_rules: list[str] = field(default_factory=list)
    failed_rules: list[dict] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.failed_rules and not any(e.severity == Severity.ERROR for e in self.errors)

    def summary(self) -> str:
        if not self.errors:
            return ("No findings from successful checks; result is incomplete."
                    if self.failed_rules else "No issues found by the available rules.")
        score_note = f"rule score {self.score:.2f}" if self.score is not None else "incomplete checks"
        lines = [f"Found {len(self.errors)} issue(s) in {self.n_words} words ({score_note}):"]
        for e in self.errors:
            sug = f"  → {e.suggestion}" if e.suggestion else ""
            lines.append(f"  [{e.severity.upper():7}] {e.message}{sug}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Abstract rule base
# ══════════════════════════════════════════════════════════════════════════════

class GrammarRule(ABC):
    """Base class for a grammar rule.

    Subclass this to add rules for any language.  Each rule checks one
    specific pattern and returns zero or more GrammarError objects.
    """

    @property
    @abstractmethod
    def rule_id(self) -> str: ...

    @property
    @abstractmethod
    def category(self) -> str: ...

    @property
    @abstractmethod
    def languages(self) -> list[str]:
        """List of language codes this rule applies to.  ["*"] = all languages."""
        ...

    @abstractmethod
    def check(self, text: str) -> list[GrammarError]: ...

    def _make_error(
        self,
        message: str,
        span: tuple[int, int],
        suggestion: Optional[str] = None,
        severity: str = Severity.ERROR,
    ) -> GrammarError:
        return GrammarError(
            message=message,
            suggestion=suggestion,
            span=span,
            severity=severity,
            rule_id=self.rule_id,
            category=self.category,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  English grammar rules
# ══════════════════════════════════════════════════════════════════════════════

class ArticleChoiceRule(GrammarRule):
    """'a' vs 'an': 'a' before consonant sounds, 'an' before vowel sounds."""

    rule_id  = "EN_ARTICLE_AN"
    category = "agreement"
    languages = ["en"]

    # Words that start with a vowel letter but consonant sound
    _consonant_sound_words = frozenset([
        "one", "once", "unit", "uniform", "unicorn", "university", "union",
        "unique", "used", "usual", "user", "university", "useful", "usage",
        "utopia", "utter",
    ])
    # Words that start with a consonant letter but vowel sound
    _vowel_sound_words = frozenset([
        "hour", "honest", "honour", "honor", "herb", "heir", "mba", "nba", "fbi", "nfl",
    ])
    _a_before_vowel   = re.compile(r'\ba\s+([aeiouAEIOU]\w*)', re.IGNORECASE)
    _an_before_cons   = re.compile(r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w*)', re.IGNORECASE)

    def check(self, text: str) -> list[GrammarError]:
        errors: list[GrammarError] = []
        for m in self._a_before_vowel.finditer(text):
            word = m.group(1).lower()
            if word in self._consonant_sound_words:
                continue  # "a university" is correct
            errors.append(self._make_error(
                f"Use 'an' before words starting with a vowel sound: 'an {m.group(1)}'",
                (m.start(), m.end()),
                suggestion=f"an {m.group(1)}",
            ))
        for m in self._an_before_cons.finditer(text):
            word = m.group(1).lower()
            if word in self._vowel_sound_words:
                continue  # "an hour" is correct
            errors.append(self._make_error(
                f"Use 'a' before words starting with a consonant sound: 'a {m.group(1)}'",
                (m.start(), m.end()),
                suggestion=f"a {m.group(1)}",
            ))
        return errors


class SubjectVerbAgreementRule(GrammarRule):
    """Common subject-verb agreement errors in English."""

    rule_id   = "EN_SVA"
    category  = "agreement"
    languages = ["en"]

    _patterns: list[tuple[re.Pattern, str, str]] = [
        # he/she/it + bare verb
        (re.compile(r'\b(he|she|it)\s+(don\'t|dont)\b', re.I), "he/she/it don't", "doesn't"),
        (re.compile(r'\b(he|she|it)\s+(have)\b(?!\s+been)', re.I), "he/she/it have", "has"),
        (re.compile(r'\b(he|she|it)\s+(go)\b', re.I), "he/she/it go", "goes"),
        (re.compile(r'\b(he|she|it)\s+(do)\b(?!\s+not)', re.I), "he/she/it do", "does"),
        (re.compile(r'\b(he|she|it)\s+(run)\b', re.I), "he/she/it run", "runs"),
        (re.compile(r'\b(he|she|it)\s+(make)\b', re.I), "he/she/it make", "makes"),
        (re.compile(r'\b(he|she|it)\s+(want)\b', re.I), "he/she/it want", "wants"),
        # I/you/we/they + singular verb form
        (re.compile(r'\b(i|you|we|they)\s+(doesn\'t|doesnt)\b', re.I), "I/you/we/they doesn't", "don't"),
        (re.compile(r'\b(i|you|we|they)\s+(has)\b(?!\s+been)', re.I), "I/you/we/they has", "have"),
        # "There is" + plural subject
        (re.compile(r'\bthere\s+is\s+\w+\s+and\s+\w+\b', re.I), "there is X and Y (plural)", "there are"),
    ]

    # A bare verb after these words is grammatical: questions and negation
    # ("does she want"), causatives/perception ("let it go", "made him run")
    # and infinitives ("to have it go").
    _licensing = {"do", "does", "did", "don't", "doesn't", "didn't", "will", "would", "can",
                  "could", "shall", "should", "may", "might", "must", "won't", "wouldn't",
                  "can't", "couldn't", "shouldn't", "let", "lets", "letting", "make", "makes",
                  "made", "making", "have", "has", "had", "help", "helps", "helped", "watch",
                  "watched", "see", "saw", "seen", "hear", "heard", "feel", "felt", "to", "bid"}

    def check(self, text: str) -> list[GrammarError]:
        errors: list[GrammarError] = []
        for pattern, desc, suggestion in self._patterns:
            for m in pattern.finditer(text):
                before = re.findall(r"[\w']+", text[:m.start()].lower())
                if before and before[-1] in self._licensing:
                    continue
                errors.append(self._make_error(
                    f"Subject-verb agreement: '{m.group()}' ({desc})",
                    (m.start(), m.end()),
                    suggestion=suggestion,
                ))
        return errors


class DoubleNegativeRule(GrammarRule):
    """Double negatives in English (nonstandard)."""

    rule_id   = "EN_DOUBLE_NEG"
    category  = "agreement"
    languages = ["en"]

    _pattern = re.compile(
        r"\b(don't|doesn't|didn't|can't|won't|wouldn't|shouldn't|couldn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
        r".{0,30}"
        r"\b(nothing|nobody|nowhere|no one|no\s+\w+|never|neither|nor)\b",
        re.IGNORECASE | re.DOTALL,
    )

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        for m in self._pattern.finditer(text):
            errors.append(self._make_error(
                f"Negative concord: '{m.group().strip()}' — grammatical in many English "
                f"varieties; formal written English usually uses one negative",
                (m.start(), m.end()),
                suggestion="For formal register: 'don't have nothing' → 'don't have anything'",
                severity=Severity.STYLE,
            ))
        return errors


class CommonConfusionsRule(GrammarRule):
    """Commonly confused words in English."""

    rule_id   = "EN_CONFUSION"
    category  = "confusion"
    languages = ["en"]

    _confusions: list[tuple[re.Pattern, str]] = [
        # their / there / they're
        (re.compile(r'\bthere\s+(book|car|house|dog|cat|friend|idea|plan|job|own)\b', re.I),
         "Did you mean 'their' (possessive)?"),
        (re.compile(r'\btheir\s+(is|are|was|were|has|have|will|would|can|could)\b', re.I),
         "Did you mean 'there' (place/existence) or 'they're' (they are)?"),
        # your / you're
        (re.compile(r"\byour\s+(going|coming|welcome|right|wrong|sure|kidding|joking|lying|saying)\b", re.I),
         "Did you mean 'you're' (you are)?"),
        (re.compile(r"\byou're\s+(job|house|car|dog|cat|friend|name|life|work|plan)\b", re.I),
         "Did you mean 'your' (possessive)?"),
        # its / it's
        (re.compile(r"\bits\s+(a|an|the|going|not|been|time|ok|fine|true|false|done)\b", re.I),
         "Did you mean 'it's' (it is / it has)?"),
        (re.compile(r"\bit's\s+(\w+\s+)?(tail|body|head|color|shape|surface|name|size|weight)\b", re.I),
         "Did you mean 'its' (possessive)?"),
        # affect / effect
        (re.compile(r'\bthe\s+affect\s+of\b', re.I),
         "Did you mean 'effect' (noun: the effect of)?"),
        # then / than
        (re.compile(r'\bbetter\s+then\b', re.I),  "Did you mean 'than' (comparison)?"),
        (re.compile(r'\bgreater\s+then\b', re.I), "Did you mean 'than' (comparison)?"),
        (re.compile(r'\bmore\s+then\b', re.I),    "Did you mean 'than' (comparison)?"),
        (re.compile(r'\bless\s+then\b', re.I),    "Did you mean 'than' (comparison)?"),
        (re.compile(r'\brather\s+then\b', re.I),  "Did you mean 'than' (comparison)?"),
        # to / too / two
        # Only before punctuation/end: "to many people", "to long-standing" are fine.
        (re.compile(r'\bto\s+(much|many|late|early|far|soon|long|fast|slow)\b(?=\s*(?:[.,!?;:]|$))', re.I),
         "Did you mean 'too' (also / excessively)?"),
        # loose / lose
        (re.compile(r'\b(going\s+to|gonna|will)\s+loose\b', re.I),
         "Did you mean 'lose' (to be defeated / misplace)?"),
        # who's / whose
        (re.compile(r"\bwhose\s+(going|coming|there|here|been|got|done)\b", re.I),
         "Did you mean 'who's' (who is / who has)?"),
        # should of / could of / would of
        # 'may' excluded: "in May of last year" is correct.
        (re.compile(r'\b(should|could|would|must|might)\s+of\b', re.I),
         "Did you mean 'have'? 'of' is incorrect after modal verbs."),
        # irregardless
        (re.compile(r'\birregardless\b', re.I),
         "'Irregardless' is nonstandard. Use 'regardless'."),
        # alot
        (re.compile(r'\balot\b', re.I),
         "'Alot' should be two words: 'a lot'."),
        # everyday / every day
        (re.compile(r'\beveryday\s+(i|we|he|she|they|you)\b', re.I),
         "Did you mean 'every day' (each day)?"),
    ]

    # Nouns, not modals: "the might of his arm". Relative possessive: "of whose coming".
    _determiners = {"the", "a", "an", "his", "her", "their", "its", "our", "my", "your", "this", "that", "whose"}
    _prepositions = {"of", "to", "in", "for", "by", "with", "from", "on", "at", "about", "upon", "without"}

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        for pattern, message in self._confusions:
            for m in pattern.finditer(text):
                before = re.findall(r"[\w']+", text[:m.start()].lower())
                first = m.group().split()[0].lower()
                if before and ((first in {"might", "must"} and before[-1] in self._determiners)
                               or (first == "whose" and before[-1] in self._prepositions)):
                    continue
                errors.append(self._make_error(
                    f"Possible confusion: '{m.group()}' — {message}",
                    (m.start(), m.end()),
                    severity=Severity.WARNING,
                ))
        return errors


class RedundancyRule(GrammarRule):
    """Redundant phrases and pleonasms."""

    rule_id   = "EN_REDUNDANCY"
    category  = "style"
    languages = ["en"]

    _redundancies: list[tuple[str, str]] = [
        ("past history",        "history"),
        ("future plans",        "plans"),
        ("end result",          "result"),
        ("final outcome",       "outcome"),
        ("free gift",           "gift"),
        ("past memories",       "memories"),
        ("close proximity",     "proximity"),
        ("exact same",          "same"),
        ("added bonus",         "bonus"),
        ("completely surrounded by", "surrounded by"),
        ("unexpected surprise", "surprise"),
        ("advance planning",    "planning"),
        ("adequate enough",     "adequate"),
        ("at this point in time", "now"),
        ("due to the fact that", "because"),
        ("in order to",         "to"),
        ("in spite of the fact that", "although"),
        ("in the event that",   "if"),
        ("a total of",          ""),
        ("12 noon",             "noon"),
        ("12 midnight",         "midnight"),
        ("each and every",      "each / every"),
        ("first and foremost",  "first"),
        ("null and void",       "null"),
        ("various different",   "various"),
        ("return back",         "return"),
        ("revert back",         "revert"),
        ("repeat again",        "repeat"),
        ("join together",       "join"),
        ("merge together",      "merge"),
        ("collaborate together","collaborate"),
        ("biography of his life", "biography"),
    ]

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        lower = text.lower()
        for phrase, suggestion in self._redundancies:
            idx = lower.find(phrase)
            while idx != -1:
                # Make sure we have word boundaries
                before_ok = idx == 0 or not lower[idx - 1].isalpha()
                after_ok  = (idx + len(phrase) == len(lower)
                             or not lower[idx + len(phrase)].isalpha())
                if before_ok and after_ok:
                    original = text[idx: idx + len(phrase)]
                    sug = f"'{suggestion}'" if suggestion else "(omit)"
                    errors.append(self._make_error(
                        f"Redundant phrase: '{original}' — prefer {sug}",
                        (idx, idx + len(phrase)),
                        suggestion=suggestion or None,
                        severity=Severity.STYLE,
                    ))
                idx = lower.find(phrase, idx + 1)
        return errors


class PunctuationRule(GrammarRule):
    """Basic punctuation hygiene."""

    rule_id   = "EN_PUNCTUATION"
    category  = "punctuation"
    languages = ["en", "de", "es", "it", "pt"]  # not fr: French typography requires a space before ; : ! ?

    _double_space     = re.compile(r'  +')
    _multiple_excl    = re.compile(r'!{3,}')
    _multiple_quest   = re.compile(r'\?{3,}')
    _space_before_punct = re.compile(r'\s([,;:!?.])')
    _missing_space_after = re.compile(r'([,;!?])([A-Za-z])')
    _comma_splice     = re.compile(r'[a-z],\s+[A-Z][a-z]')

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        for m in self._double_space.finditer(text):
            errors.append(self._make_error(
                "Extra whitespace", (m.start(), m.end()),
                suggestion=" ", severity=Severity.STYLE,
            ))
        for m in self._multiple_excl.finditer(text):
            errors.append(self._make_error(
                f"Excessive exclamation marks: '{m.group()}'",
                (m.start(), m.end()),
                suggestion="!", severity=Severity.STYLE,
            ))
        for m in self._space_before_punct.finditer(text):
            errors.append(self._make_error(
                f"Space before punctuation: ' {m.group(1)}'",
                (m.start(), m.end()),
                suggestion=m.group(1), severity=Severity.STYLE,
            ))
        for m in self._missing_space_after.finditer(text):
            errors.append(self._make_error(
                f"Missing space after punctuation: '{m.group()}'",
                (m.start(), m.end()),
                suggestion=f"{m.group(1)} {m.group(2)}", severity=Severity.STYLE,
            ))
        return errors


class SpellingPatternRule(GrammarRule):
    """Common English misspellings detectable by pattern."""

    rule_id   = "EN_SPELLING"
    category  = "spelling"
    languages = ["en"]

    _misspellings: list[tuple[re.Pattern, str, str]] = [
        (re.compile(r'\baccommodation\b', re.I),   "", ""),        # placeholder — correct
        (re.compile(r'\baccommodatoin\b', re.I),   "accomodatoin", "accommodation"),
        (re.compile(r'\brecieve\b', re.I),          "recieve",      "receive"),
        (re.compile(r'\bwierd\b', re.I),            "wierd",        "weird"),
        (re.compile(r'\boccured\b', re.I),          "occured",      "occurred"),
        (re.compile(r'\bseperate\b', re.I),         "seperate",     "separate"),
        (re.compile(r'\bdefinately\b', re.I),       "definately",   "definitely"),
        (re.compile(r'\bneccessary\b', re.I),       "neccessary",   "necessary"),
        (re.compile(r'\bexistance\b', re.I),        "existance",    "existence"),
        (re.compile(r'\bcommittment\b', re.I),      "committment",  "commitment"),
        (re.compile(r'\boccasionally\b', re.I),     "", ""),        # correct
        (re.compile(r'\boccasionly\b', re.I),       "occasionly",   "occasionally"),
        (re.compile(r'\bperseverance\b', re.I),     "", ""),        # correct
        (re.compile(r'\bperserverance\b', re.I),    "perserverance","perseverance"),
        (re.compile(r'\buntil\b', re.I),            "", ""),        # correct
        (re.compile(r'\buntill\b', re.I),           "untill",       "until"),
        (re.compile(r'\bwether\b', re.I),           "wether",       "whether"),
        (re.compile(r'\bbeggining\b', re.I),        "beggining",    "beginning"),
        (re.compile(r'\bproffesional\b', re.I),     "proffesional", "professional"),
        (re.compile(r'\bprivelege\b', re.I),        "privelege",    "privilege"),
        (re.compile(r'\bgoverment\b', re.I),        "goverment",    "government"),
        (re.compile(r'\bjudgement\b', re.I),        "", ""),        # both spellings valid
        (re.compile(r'\baccomodate\b', re.I),       "accomodate",   "accommodate"),
        (re.compile(r'\benviroment\b', re.I),       "enviroment",   "environment"),
        (re.compile(r'\bgrammer\b', re.I),          "grammer",      "grammar"),
        (re.compile(r'\bliason\b', re.I),           "liason",       "liaison"),
        (re.compile(r'\bnucleus\b', re.I),          "", ""),        # correct
        (re.compile(r'\bnoticeable\b', re.I),       "", ""),        # correct
        (re.compile(r'\bnoticiable\b', re.I),       "noticiable",   "noticeable"),
    ]

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        for pattern, wrong, correct in self._misspellings:
            if not wrong:
                continue  # skip placeholder "correct" entries
            for m in pattern.finditer(text):
                errors.append(self._make_error(
                    f"Possible misspelling: '{m.group()}' → '{correct}'",
                    (m.start(), m.end()),
                    suggestion=correct,
                    severity=Severity.WARNING,
                ))
        return errors


class PassiveVoiceRule(GrammarRule):
    """Flags passive voice constructions (style suggestion)."""

    rule_id   = "EN_PASSIVE"
    category  = "style"
    languages = ["en"]

    _passive = re.compile(
        r'\b(is|are|was|were|been|being|be)\s+'
        r'([\w]+ed|written|done|made|given|taken|found|known|seen|said|'
        r'told|shown|built|sent|brought|bought|sold|kept|held|left|lost|'
        r'broken|stolen|chosen|driven|eaten|fallen|forgotten|frozen|'
        r'grown|hidden|ridden|risen|run|spoken|thrown|worn)\b',
        re.IGNORECASE,
    )

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        for m in self._passive.finditer(text):
            errors.append(self._make_error(
                f"Passive voice: '{m.group()}' — consider active voice for clarity",
                (m.start(), m.end()),
                severity=Severity.STYLE,
            ))
        return errors[:3]  # cap at 3 to avoid noise on heavily passive texts


class SentenceLengthRule(GrammarRule):
    """Flags very long sentences."""

    rule_id   = "SENT_LENGTH"
    category  = "style"
    languages = ["*"]

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        pos = 0
        for sent in sentences:
            words = len(sent.split())
            if words > 50:
                errors.append(self._make_error(
                    f"Very long sentence ({words} words) — consider splitting it",
                    (pos, pos + len(sent)),
                    severity=Severity.STYLE,
                ))
            pos += len(sent) + 1
        return errors


# ── Spanish rules ─────────────────────────────────────────────────────────────

class SpanishAccentRule(GrammarRule):
    """Missing accents on common Spanish question words."""

    rule_id   = "ES_ACCENT"
    category  = "spelling"
    languages = ["es"]

    _unaccented_interrogatives = re.compile(
        r'\b(como|donde|cuando|cuanto|cuanta|cuantos|cuantas|que|cual|cuales|quien|quienes)\b'
        r'(?=\s+\w)',
        re.IGNORECASE,
    )

    _accent_map = {
        "como": "cómo", "donde": "dónde", "cuando": "cuándo",
        "cuanto": "cuánto", "cuanta": "cuánta", "que": "qué",
        "cual": "cuál", "cuales": "cuáles", "quien": "quién",
        "quienes": "quiénes", "cuantos": "cuántos", "cuantas": "cuántas",
    }

    # Question word positions: right after ¿ (optionally after a preposition).
    _question_start = re.compile(
        r'¿\s*(?:(?:de|a|por|para|con|en|desde|hasta|hacia|sin|sobre)\s+)?'
        r'(como|donde|cuando|cuanto|cuanta|cuantos|cuantas|que|cual|cuales|quien|quienes)\b',
        re.IGNORECASE,
    )

    def check(self, text: str) -> list[GrammarError]:
        # Only the word that opens a ¿…? question: in "¿Sabes que vienes?" 'que' is a
        # conjunction and needs no accent. Embedded questions are left unflagged (abstain).
        errors = []
        for m in self._question_start.finditer(text):
            word = m.group(1).lower()
            correct = self._accent_map.get(word)
            if correct:
                errors.append(self._make_error(
                    f"Missing accent in interrogative: '{m.group(1)}' → '{correct}'",
                    (m.start(1), m.end(1)),
                    suggestion=correct,
                    severity=Severity.WARNING,
                ))
        return errors


# ── French rules ──────────────────────────────────────────────────────────────

class FrenchNegationRule(GrammarRule):
    """Ne ... pas negation — missing 'ne' in formal register."""

    rule_id   = "FR_NEGATION"
    category  = "agreement"
    languages = ["fr"]

    # Subject pronoun + verb + pas, with no ne/n' before the verb ("je sais pas").
    _missing_ne = re.compile(
        r"\b(je|tu|il|elle|on|nous|vous|ils|elles)\s+(?!ne\b|n['’])([^\W\d_]+)\s+(pas)\b", re.I)

    def check(self, text: str) -> list[GrammarError]:
        errors = []
        for m in self._missing_ne.finditer(text):
            pronoun, verb = m.group(1), m.group(2)
            ne = "n'" if verb[0].lower() in "aeiouyhâàéèêîôû" else "ne "
            errors.append(self._make_error(
                f"Negation without 'ne': '{m.group()}' — usual in speech; formal French uses 'ne … pas'",
                (m.start(), m.end()),
                suggestion=f"{pronoun} {ne}{verb} pas",
                severity=Severity.STYLE,
            ))
        return errors


# ══════════════════════════════════════════════════════════════════════════════
#  GrammarChecker — applies all applicable rules
# ══════════════════════════════════════════════════════════════════════════════

# Default rules registered at module load
_DEFAULT_RULES: list[GrammarRule] = [
    ArticleChoiceRule(),
    SubjectVerbAgreementRule(),
    DoubleNegativeRule(),
    CommonConfusionsRule(),
    RedundancyRule(),
    PunctuationRule(),
    SpellingPatternRule(),
    PassiveVoiceRule(),
    SentenceLengthRule(),
    SpanishAccentRule(),
    FrenchNegationRule(),
]


class GrammarChecker:
    """Apply grammar rules to a text and return annotated errors.

    Usage::

        checker = GrammarChecker()
        result = checker.check("He don't have nothing to eat.", lang="en")
        print(result.summary())

    Add custom rules::

        checker.add_rule(MyCustomRule())
    """

    def __init__(self, rules: Optional[list[GrammarRule]] = None) -> None:
        self._rules: list[GrammarRule] = list(_DEFAULT_RULES) if rules is None else rules

    def add_rule(self, rule: GrammarRule) -> None:
        """Register an additional rule."""
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> None:
        """Deregister a rule by ID."""
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def check(
        self,
        text: str,
        lang: str = "en",
        severities: Optional[list[str]] = None,
    ) -> GrammarResult:
        """Check text for grammar issues.

        Args:
            text:        input text
            lang:        BCP 47 language code (e.g., "en", "es-MX")
            severities:  filter to specific severities; None = all

        Returns:
            GrammarResult with errors, score, and summary.
        """
        base_lang = lang.split("-")[0]
        all_errors: list[GrammarError] = []
        checked_rules = []
        failed_rules = []

        for rule in self._rules:
            applies = ("*" in rule.languages
                       or lang in rule.languages
                       or base_lang in rule.languages)
            if not applies:
                continue
            try:
                errors = rule.check(text)
                if (not isinstance(errors, list)
                        or any(not isinstance(error, GrammarError)
                               or not isinstance(error.span, tuple) or len(error.span) != 2
                               or any(type(offset) is not int for offset in error.span)
                               or not 0 <= error.span[0] <= error.span[1] <= len(text)
                               for error in errors)):
                    raise ValueError('Rule returned invalid findings or original-text offsets')
                all_errors.extend(errors)
                checked_rules.append(rule.rule_id)
            except Exception as error:
                # Retain successful findings without falsely declaring a failed
                # check clean. Exception types suffice; no internal trace leaks.
                failed_rules.append({'rule_id': rule.rule_id, 'error_type': type(error).__name__})

        # Filter by severity if requested
        if severities:
            all_errors = [e for e in all_errors if e.severity in severities]

        # Sort by position
        all_errors.sort(key=lambda e: e.span[0])

        # Score: 1.0 − (weighted error count / word count)
        n_words = len(text.split())
        # Style hints (passive voice, sentence length, register notes) are not errors.
        weights  = {Severity.ERROR: 3.0, Severity.WARNING: 1.5, Severity.STYLE: 0.0}
        total_w  = sum(weights.get(e.severity, 1.0) for e in all_errors)
        score    = max(0.0, 1.0 - (total_w / max(1, n_words * 3)))

        return GrammarResult(
            text=text,
            lang=lang,
            errors=all_errors,
            score=round(score, 3) if not failed_rules else None,
            n_words=n_words,
            checked_rules=checked_rules,
            failed_rules=failed_rules,
        )

    def available_rules(self) -> list[tuple[str, str, list[str]]]:
        """Return [(rule_id, category, languages)] for all registered rules."""
        return [(r.rule_id, r.category, r.languages) for r in self._rules]
