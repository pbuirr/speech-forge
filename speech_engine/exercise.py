"""Exercise generation from flashcards.

Three template-based exercise types: fill-in-the-blank on the card's example
sentence, multiple choice with same-POS distractors from the deck, and word
order. Word order checks against the one reference sentence, so other valid
orders are marked wrong.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from speech_engine.flashcard import Flashcard, FlashcardDeck
from speech_engine.text_matching import answer_key
from speech_engine.linguistic_resources import language_code


@dataclass
class ExerciseResult:
    correct: bool
    expected: str
    given: str


@dataclass
class FillInBlankExercise:
    card_id: str
    prompt: str  # sentence with the target word replaced by "____"
    answer: str
    lang: str = "en"
    source: str = "definition"
    span: Optional[tuple[int, int]] = None
    case_sensitive: bool = False

    def grade(self, given: str) -> ExerciseResult:
        ok = answer_key(given, self.lang, case_sensitive=self.case_sensitive) == answer_key(
            self.answer, self.lang, case_sensitive=self.case_sensitive
        )
        return ExerciseResult(correct=ok, expected=self.answer, given=given)


@dataclass
class MultipleChoiceExercise:
    card_id: str
    prompt: str
    choices: list[str]
    answer_index: int

    def grade(self, choice_index: int) -> ExerciseResult:
        ok = choice_index == self.answer_index
        return ExerciseResult(
            correct=ok,
            expected=self.choices[self.answer_index],
            given=self.choices[choice_index] if 0 <= choice_index < len(self.choices) else "?",
        )


@dataclass
class WordOrderExercise:
    card_id: str
    scrambled: list[str]
    answer: str  # original sentence
    lang: str = "en"

    def grade(self, given: str) -> ExerciseResult:
        ok = _normalize_sentence(given, self.lang) == _normalize_sentence(self.answer, self.lang)
        return ExerciseResult(correct=ok, expected=self.answer, given=given)


def _normalize_sentence(s: str, lang: str = "en") -> str:
    return " ".join(answer_key(s, lang).rstrip(".!?。！？؟।").split())


#  Generators


def generate_fill_in_blank(card: Flashcard, morph_engine=None) -> FillInBlankExercise:
    """Blank out `card.front` inside its example sentence, if one exists
    and contains the word; otherwise fall back to a definition prompt.
    """
    if card.example and card.front.strip():
        if morph_engine is None:
            from speech_engine.morphology import MorphologyEngine

            morph_engine = MorphologyEngine()
        pieces = morph_engine.tokenize_with_spans(card.example, card.lang)
        starts = {offsets[0] for _, offsets in pieces}
        ends = {offsets[1] for _, offsets in pieces}
        evidence = card.analysis if isinstance(card.analysis, dict) else {}
        span = evidence.get("span")
        valid = (
            evidence.get("span_unit") == "unicode_codepoints"
            and isinstance(span, (list, tuple))
            and len(span) == 2
            and all(type(v) is int for v in span)
            and 0 <= span[0] < span[1] <= len(card.example)
            and span[0] in starts
            and span[1] in ends
            and answer_key(card.example[span[0] : span[1]], card.lang)
            == answer_key(card.front, card.lang)
        )
        if valid:
            span = tuple(span)
            source = "analysis_span"
        else:
            # Legacy/manual cards use the same production tokenizer. Never
            # blank a partial Latin word merely because it contains the target.
            span = next(
                (
                    offsets
                    for surface, offsets in pieces
                    if answer_key(surface, card.lang) == answer_key(card.front, card.lang)
                ),
                None,
            )
            source = "token_match"
        if span is not None:
            left, right = span
            return FillInBlankExercise(
                card.id,
                card.example[:left] + "____" + card.example[right:],
                card.example[left:right],
                card.lang,
                source,
                span,
            )
    # No usable example — drill from the gloss instead.
    return FillInBlankExercise(
        card_id=card.id, prompt=f"____ — {card.back}", answer=card.front, lang=card.lang
    )


def generate_multiple_choice(
    card: Flashcard,
    deck: FlashcardDeck,
    n_distractors: int = 3,
    rng: Optional[random.Random] = None,
) -> Optional[MultipleChoiceExercise]:
    """Build a multiple-choice exercise: `card.back` as the prompt (the
    known side), `card.front` as the correct choice, distractors pulled
    from other cards in the deck (same POS preferred — guessing "cat"
    vs. "ran" by POS alone shouldn't be possible).

    Returns None if no distinct same-language distractor is available. Exact
    duplicate glosses are excluded; semantic equivalence is not established.
    """
    rng = rng or random.Random()
    if type(n_distractors) is not int or n_distractors < 1:
        raise ValueError("n_distractors must be a positive integer")
    seen = {answer_key(card.front, card.lang)}
    pool = []
    for candidate in deck.all():
        if candidate.id == card.id or language_code(candidate.lang) != language_code(card.lang):
            continue
        key = answer_key(candidate.front, card.lang)
        if (
            not key
            or key in seen
            or answer_key(candidate.back, card.lang) == answer_key(card.back, card.lang)
        ):
            continue  # an identical gloss cannot distinguish the choices
        seen.add(key)
        pool.append(candidate)
    if not pool:
        return None  # one choice is not a meaningful recognition exercise
    same_pos = [c for c in pool if card.pos and c.pos == card.pos]
    distractor_pool = same_pos if len(same_pos) >= n_distractors else pool

    distractors = rng.sample(distractor_pool, k=min(n_distractors, len(distractor_pool)))
    choices = [card.front] + [d.front for d in distractors]
    rng.shuffle(choices)
    return MultipleChoiceExercise(
        card_id=card.id,
        prompt=card.back,
        choices=choices,
        answer_index=choices.index(card.front),
    )


def generate_word_order(
    card: Flashcard, rng: Optional[random.Random] = None
) -> Optional[WordOrderExercise]:
    """Scramble the example sentence's words. Returns None if the card
    has no usable space-delimited example or no distinguishable permutation.
    """
    if not card.example:
        return None
    words = card.example.rstrip(".!?").split()
    if len(words) < 3 or len(set(words)) < 2:
        return None
    rng = rng or random.Random()
    scrambled = words[:]
    rng.shuffle(scrambled)
    if scrambled == words:
        # A bounded fallback also works with a no-op or unlucky RNG.
        other = next(i for i, word in enumerate(words) if word != words[0])
        scrambled[0], scrambled[other] = scrambled[other], scrambled[0]
    return WordOrderExercise(
        card_id=card.id, scrambled=scrambled, answer=card.example, lang=card.lang
    )
