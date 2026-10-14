"""
scripts/demos/flashcards.py — LinguaForge Phase 6: Learning Engine

Demonstrates:
  1. SM-2 spaced-repetition scheduling — interval/ease progression over time
  2. Flashcard decks — manual cards + auto-extraction from real text
  3. Exercise generation — fill-in-blank, multiple-choice, word-order
  4. A simulated multi-day review session
  5. JSON persistence round-trip

Usage:
    python scripts/demos/flashcards.py
    python scripts/demos/flashcards.py --add "casa=house" --lang es --deck data/decks/default.json
    python scripts/demos/flashcards.py --review --deck data/decks/default.json
    python scripts/demos/flashcards.py --stats --deck data/decks/default.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.exercise import (
    generate_fill_in_blank,
    generate_multiple_choice,
    generate_word_order,
)
from speech_engine.flashcard import FlashcardDeck
from speech_engine.morphology import MorphologyEngine
from speech_engine.srs import CardState, Grade, sm2_update

# ── ANSI colours (matches phase5_demo.py) ───────────────────────────────────
_G = "\033[92m"
_B = "\033[94m"
_C = "\033[96m"
_Y = "\033[93m"
_D = "\033[2m"
_W = "\033[1m"
_R = "\033[0m"


def _sec(title: str) -> None:
    print(f"\n{_C}── {title} {'─' * max(0, 56 - len(title))}{_R}")


def _ok(label: str, value: str) -> None:
    print(f"  {_B}{label:<22}{_R} {value}")


DEFAULT_DECK_PATH = PROJECT_ROOT / "data" / "decks" / "default.json"


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1: SM-2 scheduling
# ══════════════════════════════════════════════════════════════════════════════


def demo_sm2(today: date) -> None:
    _sec("SM-2 Spaced Repetition — interval/ease progression")
    print(f"\n  {'Day':<5}  {'Grade':<7}  {'Rep':<4}  {'Interval':<9}  {'Ease':<6}  {'Due'}")
    print(f"  {'─' * 5}  {'─' * 7}  {'─' * 4}  {'─' * 9}  {'─' * 6}  {'─' * 12}")

    state = CardState(due_date=today)
    # Simulate a learner who mostly does well but fumbles once (day 3).
    grades = [
        Grade.GOOD,
        Grade.GOOD,
        Grade.INCORRECT,
        Grade.HESITANT,
        Grade.GOOD,
        Grade.PERFECT,
        Grade.PERFECT,
    ]
    label = {
        Grade.INCORRECT: "again",
        Grade.HESITANT: "hard",
        Grade.GOOD: "good",
        Grade.PERFECT: "easy",
    }
    day = today
    for i, g in enumerate(grades):
        state = sm2_update(state, g, today=day)
        print(
            f"  {i:<5}  {label[g]:<7}  {state.repetitions:<4}  "
            f"{state.interval:<9}  {state.ease_factor:<6}  {state.due_date}"
        )
        day = state.due_date  # jump straight to when it's next due

    print(
        f"\n  {_D}Note the lapse on day 2 ('again'): repetitions reset to 0 and the\n"
        f"  interval drops back to 1 day, but the ease factor keeps falling rather\n"
        f'  than resetting — a card that\'s been forgotten once stays "harder"\n'
        f"  even after it's relearned.{_R}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2: Decks + auto-extraction from text
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_SENTENCES = {
    "en": "The curious cat quietly explored the old, abandoned house.",
    "es": "El gato curioso exploró silenciosamente la casa vieja y abandonada.",
    "fr": "Le chat curieux a exploré silencieusement la vieille maison abandonnée.",
}


def demo_extraction(engine: MorphologyEngine) -> FlashcardDeck:
    _sec("Flashcard auto-extraction from real sentences")
    deck = FlashcardDeck(name="demo")

    # A couple of manually-authored cards (real translations — this is what
    # extraction *can't* produce on its own, see flashcard.py docstring).
    deck.add(front="hola", back="hello", lang="es", tags=["manual"])
    deck.add(front="gracias", back="thank you", lang="es", tags=["manual"])

    for lang, sentence in SAMPLE_SENTENCES.items():
        new_cards = deck.extract_from_text(sentence, lang, engine)
        print(f"\n  [{lang}] {_D}{sentence}{_R}")
        for c in new_cards:
            print(f"    {c.front:<12}  →  {c.back}")

    print(
        f"\n  {_D}Total cards in deck: {len(deck)} (2 manual + {len(deck) - 2} auto-extracted){_R}"
    )
    return deck


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3: Exercise generation
# ══════════════════════════════════════════════════════════════════════════════


def demo_exercises(deck: FlashcardDeck) -> None:
    _sec("Exercise generation")
    cards = [c for c in deck.all() if c.example]  # need an example sentence
    if not cards:
        print(f"  {_Y}No auto-extracted cards with examples to drill.{_R}")
        return
    card = cards[0]

    fib = generate_fill_in_blank(card)
    print(f"\n  Fill-in-the-blank  (card: {card.front})")
    print(f"    {fib.prompt}")
    print(f"    {_D}answer: {fib.answer}{_R}")

    mcq = generate_multiple_choice(card, deck)
    if mcq:
        print(f"\n  Multiple choice  (card: {card.front})")
        print(f'    What word means: "{mcq.prompt}"?')
        for i, choice in enumerate(mcq.choices):
            print(f"      {i + 1}. {choice}")
        print(f"    {_D}answer: {mcq.choices[mcq.answer_index]}{_R}")

    word_order = generate_word_order(card)
    if word_order:
        print(f"\n  Word order  (card: {card.front})")
        print(f"    Scrambled: {' '.join(word_order.scrambled)}")
        print(f"    {_D}answer: {word_order.answer}{_R}")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4: Simulated multi-day review session
# ══════════════════════════════════════════════════════════════════════════════


def demo_review_session(deck: FlashcardDeck, today: date) -> None:
    _sec("Simulated 14-day review session")
    import random

    rng = random.Random(7)

    print(
        f"\n  {'Day':<5}  {'Due':<5}  {'Reviewed':<10}  {'New avg ease':<13}  {'New avg interval'}"
    )
    print(f"  {'─' * 5}  {'─' * 5}  {'─' * 10}  {'─' * 13}  {'─' * 16}")
    for day_offset in range(14):
        day = today + timedelta(days=day_offset)
        due = deck.due(today=day)
        for card in due:
            # Fake a learner who's right ~75% of the time.
            grade = rng.choice([Grade.GOOD, Grade.GOOD, Grade.PERFECT, Grade.INCORRECT])
            deck.review(card.id, grade, today=day)
        stats = deck.stats()
        print(
            f"  {day_offset:<5}  {len(due):<5}  {len(due):<10}  "
            f"{stats['avg_ease']:<13}  {stats['avg_interval']}"
        )

    stats = deck.stats()
    print(
        f"\n  Final deck stats: {stats['cards']} cards, "
        f"{stats['lapses']} total lapses, avg ease {stats['avg_ease']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5: Persistence round-trip
# ══════════════════════════════════════════════════════════════════════════════


def demo_persistence(deck: FlashcardDeck) -> None:
    _sec("JSON persistence round-trip")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "deck.json"
        deck.save(path)
        reloaded = FlashcardDeck.load(path)
        ok = len(reloaded) == len(deck) and all(
            reloaded.get(c.id).front == c.front for c in deck.all()
        )
        _ok("Saved to:", str(path))
        _ok("Reloaded cards:", str(len(reloaded)))
        _ok("Round-trip OK:", f"{_G}yes{_R}" if ok else f"{_Y}no{_R}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


def _load_or_new_deck(path: Path) -> FlashcardDeck:
    if path.exists():
        return FlashcardDeck.load(path)
    return FlashcardDeck(name=path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="LinguaForge Phase 6: Learning Engine")
    parser.add_argument("--add", default=None, help='Add a card: "front=back"')
    parser.add_argument("--lang", default="en", help="Language code for --add")
    parser.add_argument("--deck", default=str(DEFAULT_DECK_PATH), help="Deck JSON file path")
    parser.add_argument("--review", action="store_true", help="Run an interactive review session")
    parser.add_argument("--stats", action="store_true", help="Show deck stats")
    args = parser.parse_args()

    deck_path = Path(args.deck)

    if args.add:
        if "=" not in args.add:
            print('  --add expects "front=back"')
            return
        front, back = args.add.split("=", 1)
        deck = _load_or_new_deck(deck_path)
        card = deck.add(front=front.strip(), back=back.strip(), lang=args.lang, tags=["manual"])
        deck_path.parent.mkdir(parents=True, exist_ok=True)
        deck.save(deck_path)
        print(f"  Added {card.front} → {card.back}  (id={card.id})")
        print(f"  Deck saved to {deck_path} ({len(deck)} cards total)")
        return

    if args.stats:
        if not deck_path.exists():
            print(f"  No deck found at {deck_path}")
            return
        deck = FlashcardDeck.load(deck_path)
        s = deck.stats()
        print(f"  Deck: {deck.name}  ({deck_path})")
        for k, v in s.items():
            _ok(k, str(v))
        return

    if args.review:
        if not deck_path.exists():
            print(f"  No deck found at {deck_path}. Add cards first with --add.")
            return
        deck = FlashcardDeck.load(deck_path)
        due = deck.due()
        if not due:
            print("  Nothing due right now.")
            return
        print(
            f"  {len(due)} card(s) due. Press Enter to reveal the answer, "
            f"then grade: (1) again  (2) hard  (3) good  (4) easy\n"
        )
        button_for = {"1": "again", "2": "hard", "3": "good", "4": "easy"}
        for card in due:
            print(f"  {card.front}")
            input("  [Enter to reveal] ")
            print(f"  → {card.back}")
            choice = input("  Grade (1-4): ").strip()
            button = button_for.get(choice, "good")
            deck.review_button(card.id, button)
            print()
        deck.save(deck_path)
        print(f"  Session saved to {deck_path}")
        return

    # ── Full demo ────────────────────────────────────────────────────────────
    print(f"\n{_W}{'═' * 66}{_R}")
    print(f"{_W}  LinguaForge — Phase 6: Learning Engine{_R}")
    print(f"{_W}{'═' * 66}{_R}")

    today = date.today()
    engine = MorphologyEngine()

    demo_sm2(today)
    deck = demo_extraction(engine)
    demo_exercises(deck)
    demo_review_session(deck, today)
    demo_persistence(deck)

    print(f"\n{_W}{'─' * 66}{_R}")
    print(f"{_G}Phase 6 complete.{_R}  Learning engine built from scratch.\n")
    print("What you built:")
    print("  • SM-2 spaced repetition scheduler                          (srs.py)")
    print("  • Flashcard decks + JSON persistence                        (flashcard.py)")
    print("  • Auto-extraction of vocabulary cards from real text        (flashcard.py)")
    print("  • Fill-in-blank / multiple-choice / word-order exercises    (exercise.py)")
    print("""
Next: Phase 7 — Machine Translation
  Trains the Phase 4 Transformer encoder-decoder as a from-scratch NMT
  model (own data, no NLLB/Marian weights) — needed before auto-extracted
  flashcards can get a real translated `back` instead of a morphological
  gloss. See ROADMAP.md for the full plan.
""")


if __name__ == "__main__":
    main()
