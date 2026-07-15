"""
speech_engine/flashcard.py — LinguaForge Phase 6: Flashcard System

A flashcard is just a (front, back) pair plus metadata, wired up to an
SM-2 scheduling state (see srs.py). The interesting bit isn't the data
structure — it's `FlashcardDeck.extract_from_text`, which builds cards
*automatically* from real sentences using the Phase 5 morphology engine:
content words (nouns/verbs/adjectives/adverbs) become cards, function
words (the/a/and/in/...) are skipped, and each card keeps the original
sentence as a usage example.

Honest limitation: without a trained translation model (a later phase —
see ROADMAP.md), there's no way to auto-generate a real L1 gloss for the
`back` of an auto-extracted card. Auto-extraction fills `back` with the
lemma + POS + morphological features instead, which is useful for
grammar/morphology drilling but is *not* a substitute for a translation.
Pass an explicit `back=` to `deck.add(...)` for real vocabulary cards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from speech_engine.srs import CardState, Grade, sm2_update

# Content (= "open class") POS tags worth turning into vocabulary cards.
# Function words (DET/PREP/CONJ/PRON/AUX/PART/PUNCT) are skipped — they're
# already covered by the grammar checker, not by vocab drilling.
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@dataclass
class Flashcard:
    id:       str
    front:    str
    back:     str
    lang:     str               = "en"
    tags:     list[str]         = field(default_factory=list)
    example:  Optional[str]     = None     # source sentence, if auto-extracted
    pos:      Optional[str]     = None     # part of speech, if known
    created:  date              = field(default_factory=date.today)
    srs:      CardState         = field(default_factory=CardState)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "front": self.front, "back": self.back,
            "lang": self.lang, "tags": self.tags, "example": self.example,
            "pos": self.pos, "created": self.created.isoformat(),
            "srs": self.srs.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Flashcard":
        return cls(
            id=d["id"], front=d["front"], back=d["back"],
            lang=d.get("lang", "en"), tags=d.get("tags", []),
            example=d.get("example"), pos=d.get("pos"),
            created=date.fromisoformat(d["created"]) if d.get("created") else date.today(),
            srs=CardState.from_dict(d["srs"]) if d.get("srs") else CardState(),
        )


class FlashcardDeck:
    """An ordered collection of Flashcards with SM-2 review scheduling
    and JSON persistence. No database — this is a single-user local tool,
    a JSON file on disk is the right amount of infrastructure.
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._cards: dict[str, Flashcard] = {}
        self._next_id = 1

    # ── Card management ─────────────────────────────────────────────────────

    def add(
        self,
        front: str,
        back: str,
        lang: str = "en",
        tags: Optional[list[str]] = None,
        example: Optional[str] = None,
        pos: Optional[str] = None,
    ) -> Flashcard:
        card_id = f"{self.name}-{self._next_id}"
        self._next_id += 1
        card = Flashcard(id=card_id, front=front, back=back, lang=lang,
                          tags=tags or [], example=example, pos=pos)
        self._cards[card_id] = card
        return card

    def remove(self, card_id: str) -> None:
        self._cards.pop(card_id, None)

    def get(self, card_id: str) -> Optional[Flashcard]:
        return self._cards.get(card_id)

    def all(self) -> list[Flashcard]:
        return list(self._cards.values())

    def __len__(self) -> int:
        return len(self._cards)

    # ── Auto-extraction from real text (uses Phase 5 morphology) ───────────

    def extract_from_text(self, text: str, lang: str, morph_engine,
                           tags: Optional[list[str]] = None) -> list[Flashcard]:
        """Turn the content words of `text` into draft flashcards.

        `morph_engine` is a `speech_engine.morphology.MorphologyEngine`
        instance (not imported at module level to avoid a hard dependency
        for callers who only want manual decks).
        """
        analysis = morph_engine.analyze(text, lang=lang)
        seen_roots: set[str] = set()
        new_cards: list[Flashcard] = []
        for a in analysis.analyses:
            if a.pos not in _CONTENT_POS:
                continue
            root = a.root.lower()
            if root in seen_roots:
                continue
            seen_roots.add(root)
            features = ", ".join(f"{k}={v}" for k, v in a.features.to_dict().items())
            back = f"{a.root} ({a.pos.lower()}{', ' + features if features else ''})"
            card = self.add(front=a.surface, back=back, lang=lang,
                             tags=(tags or []) + ["auto"], example=text, pos=a.pos)
            new_cards.append(card)
        return new_cards

    # ── Review (SM-2) ───────────────────────────────────────────────────────

    def due(self, today: Optional[date] = None) -> list[Flashcard]:
        today = today or date.today()
        return sorted(
            (c for c in self._cards.values() if c.srs.is_due(today)),
            key=lambda c: c.srs.due_date,
        )

    def review(self, card_id: str, grade: int, today: Optional[date] = None) -> Flashcard:
        card = self._cards.get(card_id)
        if card is None:
            raise KeyError(f"No such card: {card_id}")
        card.srs = sm2_update(card.srs, grade, today)
        return card

    def review_button(self, card_id: str, button: str, today: Optional[date] = None) -> Flashcard:
        return self.review(card_id, Grade.from_button(button), today)

    def stats(self) -> dict:
        if not self._cards:
            return {"cards": 0, "due": 0, "avg_ease": 0.0, "avg_interval": 0.0, "lapses": 0}
        cards = self._cards.values()
        n = len(cards)
        return {
            "cards": n,
            "due": len(self.due()),
            "avg_ease": round(sum(c.srs.ease_factor for c in cards) / n, 3),
            "avg_interval": round(sum(c.srs.interval for c in cards) / n, 2),
            "lapses": sum(c.srs.lapses for c in cards),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "next_id": self._next_id,
            "cards": [c.to_dict() for c in self._cards.values()],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FlashcardDeck":
        data = json.loads(Path(path).read_text())
        deck = cls(name=data.get("name", "default"))
        deck._next_id = data.get("next_id", 1)
        for cd in data.get("cards", []):
            card = Flashcard.from_dict(cd)
            deck._cards[card.id] = card
        return deck
