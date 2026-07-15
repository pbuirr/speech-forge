"""
speech_engine/exercise.py — LinguaForge Phase 6: Exercise Generation

Generates drill exercises from Flashcards. Three kinds, all rule-based
(no generative model needed — these are template transforms over data
you already have):

  - FillInBlank   — blank out the target word in its example sentence
  - MultipleChoice — target word + distractors pulled from the same deck,
                      preferring same-POS distractors (so "cat" isn't
                      trivially distinguishable from "ran")
  - WordOrder     — scramble an example sentence's tokens, ask the learner
                      to put them back; a cheap way to drill word-order
                      typology (see morphology.morphological_typology)

Pronunciation-style exercises (compare spoken attempt vs. target) are
intentionally not implemented here — that needs the ASR decoder
(speech_engine/decode.py) wired to a trained checkpoint, which is a
runtime/integration concern, not exercise *generation*. The interface
below (`Exercise` / `ExerciseResult`) is shaped so a future
`PronunciationExercise` slots in the same way once that's wired up.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from speech_engine.flashcard import Flashcard, FlashcardDeck


@dataclass
class ExerciseResult:
    correct:  bool
    expected: str
    given:    str


@dataclass
class FillInBlankExercise:
    card_id:  str
    prompt:   str   # sentence with the target word replaced by "____"
    answer:   str

    def grade(self, given: str) -> ExerciseResult:
        ok = given.strip().lower() == self.answer.strip().lower()
        return ExerciseResult(correct=ok, expected=self.answer, given=given)


@dataclass
class MultipleChoiceExercise:
    card_id:  str
    prompt:   str
    choices:  list[str]
    answer_index: int

    def grade(self, choice_index: int) -> ExerciseResult:
        ok = choice_index == self.answer_index
        return ExerciseResult(correct=ok, expected=self.choices[self.answer_index],
                               given=self.choices[choice_index] if 0 <= choice_index < len(self.choices) else "?")


@dataclass
class WordOrderExercise:
    card_id:   str
    scrambled: list[str]
    answer:    str   # original sentence

    def grade(self, given: str) -> ExerciseResult:
        ok = _normalize_sentence(given) == _normalize_sentence(self.answer)
        return ExerciseResult(correct=ok, expected=self.answer, given=given)


def _normalize_sentence(s: str) -> str:
    return " ".join(s.strip().lower().rstrip(".!?").split())


# ══════════════════════════════════════════════════════════════════════════════
#  Generators
# ══════════════════════════════════════════════════════════════════════════════

def generate_fill_in_blank(card: Flashcard) -> FillInBlankExercise:
    """Blank out `card.front` inside its example sentence, if one exists
    and contains the word; otherwise fall back to a definition prompt.
    """
    if card.example:
        # Whole-word, case-insensitive replace of the first occurrence.
        words = card.example.split()
        for i, w in enumerate(words):
            stripped = w.strip(".,;:!?\"'()")
            if stripped.lower() == card.front.lower():
                words[i] = w.replace(stripped, "____")
                return FillInBlankExercise(card_id=card.id, prompt=" ".join(words), answer=card.front)
    # No usable example — drill from the gloss instead.
    return FillInBlankExercise(card_id=card.id, prompt=f"____ — {card.back}", answer=card.front)


def generate_multiple_choice(
    card: Flashcard,
    deck: FlashcardDeck,
    n_distractors: int = 3,
    rng: Optional[random.Random] = None,
) -> MultipleChoiceExercise:
    """Build a multiple-choice exercise: `card.back` as the prompt (the
    known side), `card.front` as the correct choice, distractors pulled
    from other cards in the deck (same POS preferred — guessing "cat"
    vs. "ran" by POS alone shouldn't be possible).
    """
    rng = rng or random.Random()
    pool = [c for c in deck.all() if c.id != card.id and c.front.lower() != card.front.lower()]
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


def generate_word_order(card: Flashcard, rng: Optional[random.Random] = None) -> Optional[WordOrderExercise]:
    """Scramble the example sentence's words. Returns None if the card
    has no example or the example is a single word (nothing to reorder).
    """
    if not card.example:
        return None
    words = card.example.rstrip(".!?").split()
    if len(words) < 3:
        return None
    rng = rng or random.Random()
    scrambled = words[:]
    while scrambled == words:           # don't hand back the unscrambled order
        rng.shuffle(scrambled)
    return WordOrderExercise(card_id=card.id, scrambled=scrambled, answer=card.example)
