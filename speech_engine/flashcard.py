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
surface + predicted POS and heuristic analysis instead, which is useful for
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
from speech_engine.text_matching import answer_key

# Content (= "open class") POS tags worth turning into vocabulary cards.
# Function words (DET/PREP/CONJ/PRON/AUX/PART/PUNCT) are skipped — they're
# already covered by the grammar checker, not by vocab drilling.
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@dataclass
class Flashcard:
    id: str
    front: str
    back: str
    lang: str = "en"
    tags: list[str] = field(default_factory=list)
    example: Optional[str] = None  # source sentence, if auto-extracted
    pos: Optional[str] = None  # part of speech, if known
    created: date = field(default_factory=date.today)
    srs: CardState = field(default_factory=CardState)
    analysis: Optional[dict] = None  # original analysis, alternatives, and corpus evidence

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "front": self.front,
            "back": self.back,
            "lang": self.lang,
            "tags": self.tags,
            "example": self.example,
            "pos": self.pos,
            "created": self.created.isoformat(),
            "srs": self.srs.to_dict(),
            "analysis": self.analysis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Flashcard":
        return cls(
            id=d["id"],
            front=d["front"],
            back=d["back"],
            lang=d.get("lang", "en"),
            tags=d.get("tags", []),
            example=d.get("example"),
            pos=d.get("pos"),
            created=date.fromisoformat(d["created"]) if d.get("created") else date.today(),
            srs=CardState.from_dict(d["srs"]) if d.get("srs") else CardState(),
            analysis=d.get("analysis"),
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
        analysis: Optional[dict] = None,
    ) -> Flashcard:
        card_id = f"{self.name}-{self._next_id}"
        self._next_id += 1
        card = Flashcard(
            id=card_id,
            front=front,
            back=back,
            lang=lang,
            tags=tags or [],
            example=example,
            pos=pos,
            analysis=analysis,
        )
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

    def extract_from_text(
        self, text: str, lang: str, morph_engine, tags: Optional[list[str]] = None
    ) -> list[Flashcard]:
        """Turn the content words of `text` into draft flashcards.

        `morph_engine` is a `speech_engine.morphology.MorphologyEngine`
        instance (not imported at module level to avoid a hard dependency
        for callers who only want manual decks).
        """
        analysis = morph_engine.analyze(text, lang=lang)
        return self.extract_from_analysis(analysis, tags=tags)

    def extract_from_analysis(self, analysis, tags: Optional[list[str]] = None) -> list[Flashcard]:
        """Reuse a pipeline's analysis and evidence without rerunning a model."""
        text, lang = analysis.text, analysis.lang
        if text is None:
            raise ValueError("Card extraction requires an analysis of original text")
        seen_forms = {}
        new_cards: list[Flashcard] = []
        for token_index, a in enumerate(analysis.analyses):
            if a.pos not in _CONTENT_POS:
                continue
            # Heuristic stems must not collapse distinct vocabulary entries.
            form = answer_key(a.surface, lang)
            status = a.to_dict()
            # A spelling is not a single lexical analysis: keep homographs
            # apart when POS/lemma evidence distinguishes their uses.
            lemma = a.root if status["root_status"] in {"corpus_attested", "predicted"} else None
            contextual_pos = a.pos if a.pos_method in {"bigram_hmm", "corpus_lookup"} else None
            form = (form, contextual_pos, lemma, status["root_status"])
            occurrence = {
                "token_index": token_index,
                "surface": a.surface,
                "span": list(a.span) if a.span is not None else None,
                "analysis": dict(status),
            }
            if form in seen_forms:
                seen_forms[form].analysis["occurrences"].append(occurrence)
                if (
                    status["pos_review_recommended"]
                    and "pos-uncertain" not in seen_forms[form].tags
                ):
                    seen_forms[form].tags.append("pos-uncertain")
                continue
            features = ", ".join(f"{k}={v}" for k, v in a.features.to_dict().items())
            back = f"{a.surface} (predicted {a.pos.lower()})"
            status["occurrences"] = [occurrence]
            if status["root_status"] == "corpus_attested":
                pos_label = (
                    "corpus-attested" if status["pos_status"] == "corpus_attested" else "predicted"
                )
                back = f"{a.root} ({pos_label} {a.pos.lower()})"
                if a.ud_features:
                    label = (
                        "predicted features: "
                        if a.method == "corpus_context"
                        else "attested features: "
                    )
                    back += (
                        "; "
                        + label
                        + ", ".join(f"{key}={value}" for key, value in a.ud_features.items())
                    )
            elif status["root_status"] == "predicted":
                back += f"; predicted lemma: {a.root}"
            if getattr(a, "method", "unavailable") == "english_affix_rules":
                back += f"; heuristic stem: {a.root}"
                if features:
                    back += f"; predicted features: {features}"
            card = self.add(
                front=a.surface,
                back=back,
                lang=lang,
                tags=(tags or [])
                + ["auto", "needs-review"]
                + (["pos-uncertain"] if status["pos_review_recommended"] else []),
                example=text,
                pos=a.pos,
                analysis=status,
            )
            new_cards.append(card)
            seen_forms[form] = card
        return new_cards

    def extract_from_understanding(self, result, morph_engine, grammar_checker, tags=None):
        """Connect draft learning items to the shared source and rule findings.

        Findings remain heuristics; links do not establish sentence correctness,
        word senses, or dependency structure. No component is run a second time.
        """
        from hashlib import sha256

        if result.input_evidence is not None:
            raise ValueError(
                "Automatic learning cards are unavailable for model-generated input; add manually checked material"
            )
        payload = result.to_dict(morph_engine, grammar_checker)
        cards = self.extract_from_analysis(result.morphology, tags=tags)
        source_hash = sha256(result.text.encode("utf-8")).hexdigest()
        for card in cards:
            issue_indices = set()
            relationship_indices = set()
            for occurrence in card.analysis["occurrences"]:
                indices = payload["morphology"]["tokens"][occurrence["token_index"]][
                    "grammar_issue_indices"
                ]
                occurrence["grammar_issue_indices"] = list(indices)
                issue_indices.update(indices)
                links = payload["morphology"]["tokens"][occurrence["token_index"]][
                    "relationship_indices"
                ]
                occurrence["relationship_indices"] = list(links)
                relationship_indices.update(links)
            card.analysis["context"] = {
                "schema_version": 1,
                "text_sha256": source_hash,
                "language": result.lang,
                "language_selection": result.language_selection,
                "span_unit": "unicode_codepoints",
                "grammar_status": payload["grammar"]["status"],
                "grammar_findings": [
                    {"issue_index": i, **payload["grammar"]["errors"][i]}
                    for i in sorted(issue_indices)
                ],
                "failed_rules": payload["grammar"]["failed_rules"],
                "form_relationships": [
                    payload["morphology"]["relationships"][i] for i in sorted(relationship_indices)
                ],
            }
            syntax = payload.get("syntax") or {}
            indices = {o["token_index"] for o in card.analysis["occurrences"]}
            card.analysis["context"]["syntax"] = {
                "status": syntax.get("status", "unavailable"),
                "method": syntax.get("method"),
                "source": syntax.get("source"),
                "confidence": None,
                "note": syntax.get("note"),
                "arcs": [
                    {
                        **arc,
                        "surface": payload["morphology"]["tokens"][arc["token_index"]]["surface"],
                        "head_surface": payload["morphology"]["tokens"][arc["head_token_index"]][
                            "surface"
                        ]
                        if arc["head_token_index"] is not None
                        else None,
                    }
                    for arc in syntax.get("arcs", [])
                    if arc["token_index"] in indices or arc["head_token_index"] in indices
                ],
            }
            sentence_indices = {
                payload["morphology"]["tokens"][i].get("sentence_index") for i in indices
            }
            card.analysis["context"]["sentences"] = [
                s for s in payload["sentences"] if s["sentence_index"] in sentence_indices
            ]
            if issue_indices:
                card.tags.append("source-grammar-finding")
        return cards

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
        # Encode before touching the destination; failed writes must preserve it.
        import os
        import tempfile

        destination = Path(path)
        content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "FlashcardDeck":
        try:
            content = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            import locale

            content = Path(path).read_text(encoding=locale.getencoding())
        data = json.loads(content)
        deck = cls(name=data.get("name", "default"))
        deck._next_id = data.get("next_id", 1)
        for cd in data.get("cards", []):
            card = Flashcard.from_dict(cd)
            deck._cards[card.id] = card
        return deck
