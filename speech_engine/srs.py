"""SM-2 spaced-repetition scheduler (Wozniak, 1987).

Each card has an ease factor (starts at 2.5, minimum 1.3), a repetition count
and an interval in days. A grade below 3 resets the card; otherwise the
interval goes 1 day, 6 days, then multiplies by the ease factor, and the ease
factor moves up or down with the grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


#  Grades


class Grade:
    """SM-2's original 0-5 recall-quality scale.

    Most flashcard UIs only expose four buttons (Again/Hard/Good/Easy) —
    `Grade.from_button` maps those onto the 0-5 scale SM-2 expects.
    """

    BLACKOUT = 0  # complete blank
    INCORRECT = 1  # wrong, but recognised the answer once shown
    INCORRECT_EASY = 2  # wrong, but felt close
    HESITANT = 3  # correct, with serious difficulty
    GOOD = 4  # correct, with some hesitation
    PERFECT = 5  # correct, instant recall

    PASS_THRESHOLD = 3  # grades below this reset the card

    _BUTTON_MAP = {
        "again": INCORRECT,
        "hard": HESITANT,
        "good": GOOD,
        "easy": PERFECT,
    }

    @classmethod
    def from_button(cls, button: str) -> int:
        try:
            return cls._BUTTON_MAP[button.strip().lower()]
        except KeyError:
            raise ValueError(f"Unknown button {button!r}; expected one of {list(cls._BUTTON_MAP)}")


#  Card scheduling state

MIN_EASE_FACTOR = 1.3
INITIAL_EASE_FACTOR = 2.5


@dataclass
class CardState:
    """Scheduling state for one flashcard. Pure data — no behaviour."""

    repetitions: int = 0
    interval: int = 0  # days
    ease_factor: float = INITIAL_EASE_FACTOR
    due_date: date = field(default_factory=date.today)
    last_reviewed: Optional[date] = None
    lapses: int = 0  # total times grade < PASS_THRESHOLD
    total_reviews: int = 0

    def is_due(self, today: Optional[date] = None) -> bool:
        return self.due_date <= (today or date.today())

    def to_dict(self) -> dict:
        return {
            "repetitions": self.repetitions,
            "interval": self.interval,
            "ease_factor": self.ease_factor,
            "due_date": self.due_date.isoformat(),
            "last_reviewed": self.last_reviewed.isoformat() if self.last_reviewed else None,
            "lapses": self.lapses,
            "total_reviews": self.total_reviews,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CardState":
        return cls(
            repetitions=d.get("repetitions", 0),
            interval=d.get("interval", 0),
            ease_factor=d.get("ease_factor", INITIAL_EASE_FACTOR),
            due_date=date.fromisoformat(d["due_date"]) if d.get("due_date") else date.today(),
            last_reviewed=date.fromisoformat(d["last_reviewed"])
            if d.get("last_reviewed")
            else None,
            lapses=d.get("lapses", 0),
            total_reviews=d.get("total_reviews", 0),
        )


def sm2_update(state: CardState, grade: int, today: Optional[date] = None) -> CardState:
    """Apply one SM-2 review step and return the new scheduling state.

    `state` is not mutated — a fresh CardState is returned so callers can
    keep history if they want to (e.g. for a "review log" / forgetting-curve
    chart).
    """
    if not 0 <= grade <= 5:
        raise ValueError(f"grade must be 0-5, got {grade}")
    today = today or date.today()

    repetitions = state.repetitions
    interval = state.interval
    ef = state.ease_factor
    lapses = state.lapses

    if grade < Grade.PASS_THRESHOLD:
        # Forgot it — back to the start of the learning curve, but the ease
        # factor is *not* reset (SM-2: a hard card stays hard, it doesn't
        # un-learn its history).
        repetitions = 0
        interval = 1
        lapses += 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ef)
        repetitions += 1

    # Ease factor adjustment (SM-2's original formula) — always applied,
    # even on a lapse, so a card that's repeatedly hard keeps getting
    # shorter intervals relative to an easy one.
    ef = ef + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ef = max(MIN_EASE_FACTOR, ef)

    return CardState(
        repetitions=repetitions,
        interval=interval,
        ease_factor=round(ef, 4),
        due_date=today + timedelta(days=interval),
        last_reviewed=today,
        lapses=lapses,
        total_reviews=state.total_reviews + 1,
    )


#  Scheduler — convenience wrapper over a collection of (id -> CardState)


class SRSScheduler:
    """Tracks SM-2 state for a set of card ids. Card *content* (front/back
    text etc.) lives in flashcard.py — this class only knows about scheduling.
    """

    def __init__(self) -> None:
        self._states: dict[str, CardState] = {}

    def register(self, card_id: str, state: Optional[CardState] = None) -> None:
        self._states.setdefault(card_id, state or CardState())

    def review(self, card_id: str, grade: int, today: Optional[date] = None) -> CardState:
        state = self._states.get(card_id) or CardState()
        new_state = sm2_update(state, grade, today)
        self._states[card_id] = new_state
        return new_state

    def state(self, card_id: str) -> CardState:
        return self._states.setdefault(card_id, CardState())

    def due_ids(self, today: Optional[date] = None) -> list[str]:
        today = today or date.today()
        return [cid for cid, s in self._states.items() if s.is_due(today)]

    def stats(self) -> dict:
        if not self._states:
            return {"cards": 0, "due": 0, "avg_ease": 0.0, "avg_interval": 0.0, "lapses": 0}
        n = len(self._states)
        return {
            "cards": n,
            "due": len(self.due_ids()),
            "avg_ease": round(sum(s.ease_factor for s in self._states.values()) / n, 3),
            "avg_interval": round(sum(s.interval for s in self._states.values()) / n, 2),
            "lapses": sum(s.lapses for s in self._states.values()),
        }
