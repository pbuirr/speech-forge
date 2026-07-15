"""
webapp/server.py — LinguaForge local web demo

A small hand-rolled HTTP server — stdlib `http.server` only, no Flask/
FastAPI/Django — exposing the speech_engine's language detector, grammar
checker, morphology engine, and flashcard/SRS system as a JSON API, with
a single static HTML page as the frontend (vanilla JS, no build step,
no frontend framework).

This keeps the "from scratch" philosophy consistent through the web
layer too: every line here is either stdlib or a call into the engines
already built in earlier phases.

Run:
    .venv/bin/python3 webapp/server.py
Then open http://127.0.0.1:8765 in a browser.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.exercise import (
    generate_fill_in_blank,
    generate_multiple_choice,
    generate_word_order,
)
from speech_engine.flashcard import FlashcardDeck
from speech_engine.grammar import GrammarChecker
from speech_engine.language_detector import LanguageDetector
from speech_engine.morphology import MorphologyEngine
from speech_engine.conjugation import conjugate, lookup_form, TENSE_NAMES, PERSONS, EN_PERSONS, known_verbs
from speech_engine.nmt import load_checkpoint
from speech_engine.preprocessing import normalize_es_input, postprocess_en_output, detect_translation_failure
from speech_engine.lexical_fallback import LexicalFallback
from speech_engine.vad import VoiceActivityDetector

STATIC_DIR = Path(__file__).resolve().parent / "static"
DECK_PATH = PROJECT_ROOT / "data" / "decks" / "web.json"
PORT = 8765

# Parallel corpora for statistical lexical fallback — one TSV per language pair.
# IBM Model 1 is trained on whichever corpus is present at startup.
CORPUS_PATHS: dict[str, Path] = {
    "es→en": PROJECT_ROOT / "data" / "corpora" / "es_en_real.tsv",
    "fr→en": PROJECT_ROOT / "data" / "corpora" / "fr_en_real.tsv",
    "de→en": PROJECT_ROOT / "data" / "corpora" / "de_en_real.tsv",
}
# Curated corpus supplements the real one when available
_CURATED: dict[str, Path] = {
    "es→en": PROJECT_ROOT / "data" / "corpora" / "es_en.tsv",
}

NMT_CHECKPOINTS = {
    "es→en": PROJECT_ROOT / "data" / "models" / "nmt_es_en.pt",
    "fr→en": PROJECT_ROOT / "data" / "models" / "nmt_fr_en.pt",
}

# Engines are stateless / cheap to construct — one shared instance per process.
detector = LanguageDetector()
checker  = GrammarChecker()
morph    = MorphologyEngine()
vad      = VoiceActivityDetector()

# Lexical fallback cache — trained per language pair on first request.
# Training IBM Model 1 on 200k pairs takes ~10s; cached thereafter.
_lex_cache: dict[str, object] = {}


def _get_fallback(direction: str):
    """Return (or build) the IBM Model 1 fallback for a language pair."""
    if direction not in _lex_cache:
        src, tgt = direction.split("→")
        # Combine all available corpora: large real corpus + curated pairs
        # IBM Model 1 is trained on the merged set so curated pairs anchor
        # rare/specific forms that would drown in the noisy real corpus alone.
        corpora = []
        real = CORPUS_PATHS.get(direction)
        if real and real.exists():
            corpora.append(real)
        curated = _CURATED.get(direction)
        if curated and curated.exists():
            corpora.append(curated)
        _lex_cache[direction] = LexicalFallback.from_corpora(
            corpora, src_lang=src, tgt_lang=tgt
        )
    return _lex_cache[direction]

# NMT models — loaded lazily on first /api/translate call.
_nmt_cache: dict[str, tuple] = {}  # direction → (model, src_vocab, tgt_vocab)


def _get_nmt(direction: str):
    if direction not in _nmt_cache:
        ckpt_path = NMT_CHECKPOINTS.get(direction)
        if ckpt_path is None or not ckpt_path.exists():
            return None
        model, src_vocab, tgt_vocab, _ = load_checkpoint(ckpt_path)
        _nmt_cache[direction] = (model, src_vocab, tgt_vocab)
    return _nmt_cache.get(direction)


def _load_deck() -> FlashcardDeck:
    if DECK_PATH.exists():
        return FlashcardDeck.load(DECK_PATH)
    DECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    return FlashcardDeck(name="web")


_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    # ── helpers ──────────────────────────────────────────────────────────────

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        path = (STATIC_DIR / rel_path.lstrip("/")).resolve()
        if STATIC_DIR not in path.parents and path != STATIC_DIR or not path.exists():
            return self._send_json({"error": "not found"}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _STATIC_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def log_message(self, fmt, *args) -> None:  # quiet the default access log
        pass

    # ── routing ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self._send_static("index.html")
        if path == "/api/deck":
            deck = _load_deck()
            due_ids = {c.id for c in deck.due()}
            cards = [dict(c.to_dict(), is_due=c.id in due_ids) for c in deck.all()]
            return self._send_json({"name": deck.name, "cards": cards, "stats": deck.stats()})
        if path == "/api/languages":
            from speech_engine.language_detector import LANG_DB
            _MORPH_DEEP = {"en", "es", "fr", "de", "ru", "ar"}
            _GRAMMAR    = {"en", "es", "fr"}
            _nmt_src = {d.split("→")[0] for d, p in NMT_CHECKPOINTS.items() if p.exists()}
            _nmt_tgt = {d.split("→")[1] for d, p in NMT_CHECKPOINTS.items() if p.exists()}
            langs = [
                {
                    "code": code,
                    "name": defn.name,
                    "script": defn.script,
                    "dialect_of": defn.dialect_of,
                    "caps": {
                        "detection": True,
                        "morphology": code in _MORPH_DEEP,
                        "grammar": code in _GRAMMAR,
                        "nmt_src": code in _nmt_src,
                        "nmt_tgt": code in _nmt_tgt,
                    },
                }
                for code, defn in LANG_DB.items()
            ]
            return self._send_json({"languages": langs, "total": len(langs)})
        self._send_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            data = self._read_json()
            handler = _ROUTES.get(path)
            if handler is None:
                return self._send_json({"error": f"no such endpoint: {path}"}, 404)
            return handler(self, data)
        except Exception as e:  # noqa: BLE001 — surface the error to the page, don't crash the server
            return self._send_json({"error": str(e)}, 500)


# ── POST endpoint implementations ────────────────────────────────────────────

def _detect(self: Handler, data: dict) -> None:
    text = data.get("text", "")
    result = detector.detect(text)
    self._send_json({
        "language": result.language,
        "name": result.name,
        "dialect": result.dialect,
        "script": result.script,
        "confidence": result.confidence,
        "is_mixed": result.is_mixed,
        "top": [{"code": m.code, "name": m.name, "confidence": m.confidence} for m in result.top],
    })


def _grammar(self: Handler, data: dict) -> None:
    text = data.get("text", "")
    lang = data.get("lang", "en")
    r = checker.check(text, lang=lang)
    self._send_json({
        "is_clean": r.is_clean,
        "score": r.score,
        "n_words": r.n_words,
        "errors": [
            {"message": e.message, "suggestion": e.suggestion, "severity": e.severity,
             "category": e.category, "rule_id": e.rule_id, "span": list(e.span)}
            for e in r.errors
        ],
    })


_MORPH_DEEP_LANGS = {"en", "es", "fr", "de", "ru", "ar"}


def _morphology(self: Handler, data: dict) -> None:
    text = data.get("text", "")
    lang = data.get("lang", "en")
    r = morph.analyze(text, lang=lang)
    limited = lang not in _MORPH_DEEP_LANGS
    self._send_json({
        "typology": morph.get_typology(lang),
        "limited": limited,
        "limited_note": (
            f"Full morphological analysis is not yet available for '{lang}'. "
            "Showing basic tokenization (punctuation-split). "
            "POS tags, morpheme breakdown, and root extraction require a dedicated "
            "language profile — contribute lexical data or a tokenizer to unlock it."
        ) if limited else None,
        "tokens": [
            {"surface": a.surface, "root": a.root, "pos": a.pos,
             "morphemes": a.morphemes, "features": a.features.to_dict()}
            for a in r.analyses
        ],
    })


def _extract(self: Handler, data: dict) -> None:
    text = data.get("text", "")
    lang = data.get("lang", "en")
    deck = _load_deck()
    new_cards = deck.extract_from_text(text, lang, morph)
    deck.save(DECK_PATH)
    self._send_json({"added": len(new_cards), "cards": [c.to_dict() for c in new_cards]})


def _add_card(self: Handler, data: dict) -> None:
    deck = _load_deck()
    card = deck.add(front=data["front"], back=data["back"], lang=data.get("lang", "en"), tags=["manual"])
    deck.save(DECK_PATH)
    self._send_json({"card": card.to_dict()})


def _review(self: Handler, data: dict) -> None:
    deck = _load_deck()
    card = deck.review_button(data["card_id"], data["button"])
    deck.save(DECK_PATH)
    self._send_json({"card": card.to_dict()})


def _exercise(self: Handler, data: dict) -> None:
    deck = _load_deck()
    card = deck.get(data["card_id"])
    if card is None:
        return self._send_json({"error": "no such card"}, 404)
    kind = data.get("kind", "fill")
    if kind == "fill":
        ex = generate_fill_in_blank(card)
        return self._send_json({"kind": "fill", "prompt": ex.prompt, "answer": ex.answer})
    if kind == "mcq":
        ex = generate_multiple_choice(card, deck)
        return self._send_json({"kind": "mcq", "prompt": ex.prompt, "choices": ex.choices,
                                 "answer_index": ex.answer_index})
    if kind == "order":
        ex = generate_word_order(card)
        if ex is None:
            return self._send_json({"error": "card has no example sentence to scramble"}, 400)
        return self._send_json({"kind": "order", "scrambled": ex.scrambled, "answer": ex.answer})
    self._send_json({"error": f"unknown exercise kind: {kind}"}, 400)


def _delete_card(self: Handler, data: dict) -> None:
    deck = _load_deck()
    deck.remove(data["card_id"])
    deck.save(DECK_PATH)
    self._send_json({"ok": True})


def _vad(self: Handler, data: dict) -> None:
    """
    POST /api/vad  { "samples": [...float32...], "sr": 16000 }
    Returns detected speech segments as [{start, end, duration}].
    Accepts raw float32 PCM in the range [-1, 1].
    """
    import numpy as np
    samples = data.get("samples")
    sr      = int(data.get("sr", 16_000))
    if not samples:
        return self._send_json({"error": "samples required"}, 400)

    audio    = np.array(samples, dtype=np.float32)
    segments = vad.detect(audio)
    features = vad.frame_features(audio)
    self._send_json({
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "duration": round(s.duration, 3)}
            for s in segments
        ],
        "n_frames":   len(features["times"]),
        "total_secs": round(float(len(audio) / sr), 3),
        "sr":         sr,
    })


def _dedup_words(text: str) -> str:
    """Remove adjacent duplicate words from a decoded translation."""
    words = text.split()
    out = []
    for w in words:
        if not out or w != out[-1]:
            out.append(w)
    return " ".join(out)


def _conjugate(self: Handler, data: dict) -> None:
    """
    POST /api/conjugate  { "verb": "hablar" }
    Returns full conjugation table across 6 tenses and 6 persons.
    Also accepts any conjugated form — returns the table for its infinitive.
    """
    verb = data.get("verb", "").strip().lower()
    if not verb:
        return self._send_json({"error": "verb required"}, 400)

    # Try direct infinitive lookup first; fall back to form reverse-lookup
    table = conjugate(verb)
    if table is None:
        matches = lookup_form(verb)
        if matches:
            table = conjugate(matches[0]["infinitive"])
            if table is None:
                return self._send_json({"error": f"unknown verb: {verb}"}, 404)
        else:
            return self._send_json({"error": f"unknown verb: {verb}"}, 404)

    tenses_out = {}
    for tense_key, forms in table.tenses.items():
        tenses_out[tense_key] = {
            "name": TENSE_NAMES.get(tense_key, tense_key),
            "forms": [
                {"person_es": PERSONS[i], "person_en": EN_PERSONS[i], "form": f}
                for i, f in enumerate(forms)
            ],
        }

    self._send_json({
        "infinitive":    table.infinitive,
        "english":       table.english,
        "gerund_es":     table.gerund_es,
        "gerund_en":     table.gerund_en,
        "participle_es": table.participle_es,
        "participle_en": table.participle_en,
        "verb_class":    table.verb_class,
        "is_regular":    table.is_regular,
        "notes":         table.notes,
        "tenses":        tenses_out,
    })


def _lookup_form(self: Handler, data: dict) -> None:
    """POST /api/lookup_form { "form": "fue" } — reverse-lookup a conjugated form."""
    form = data.get("form", "").strip().lower()
    if not form:
        return self._send_json({"error": "form required"}, 400)
    matches = lookup_form(form)
    self._send_json({"form": form, "matches": matches})


def _translate(self: Handler, data: dict) -> None:
    import torch
    text = data.get("text", "").strip()
    direction = data.get("direction", "es→en")
    beam = int(data.get("beam", 4))

    if not text:
        return self._send_json({"error": "text is required"}, 400)

    result = _get_nmt(direction)
    if result is None:
        available = [d for d, p in NMT_CHECKPOINTS.items() if p.exists()]
        return self._send_json({
            "error": (
                f"No trained model for {direction!r}. "
                f"Run: .venv/bin/python3 scripts/phase7_train.py  to train one. "
                f"Available: {available or 'none yet'}"
            )
        }, 404)

    model, src_vocab, tgt_vocab = result
    device = next(model.parameters()).device

    # ── Linguistic preprocessing ─────────────────────────────────────────────
    # Normalize OOV constructs before BPE encoding so the model sees
    # known tokens instead of UNK. E.g. "$20" → "20 dólares".
    if direction == "es→en":
        prep = normalize_es_input(text)
        normalized_text = prep.normalized
    else:
        prep = None
        normalized_text = text

    src_ids = src_vocab.encode(normalized_text.lower())
    if not src_ids:
        return self._send_json({"translation": ""})

    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # ── Translation + attention in one shot ─────────────────────────────────
    attn_result = model.translate_with_attention(src_tensor, beam_size=beam)
    out_ids   = attn_result["out_ids"]
    attn_mat  = attn_result["attention"]   # (T_out, T_src) floats
    entropy   = attn_result["entropy"]     # (T_out,) floats, 0=confident 1=uncertain

    translation = tgt_vocab.decode(out_ids)

    # Source tokens (BPE pieces, for the heatmap axis labels)
    src_tokens = [src_vocab._id_to_token.get(i, "?") for i in src_ids]
    # Target tokens aligned with the attention rows
    tgt_tokens = [tgt_vocab._id_to_token.get(i, "?") for i in out_ids]

    # ── Post-processing ──────────────────────────────────────────────────────
    translation = postprocess_en_output(translation)

    coverage_warnings: list[str] = []
    if prep is not None:
        failures = prep.coverage_failures(translation)
        if failures:
            coverage_warnings = [f.target_hint for f in failures]
            translation = prep.repair(translation)

    avg_entropy = (sum(entropy) / len(entropy)) if entropy else None
    failed = detect_translation_failure(
        normalized_text, translation,
        avg_entropy=avg_entropy,
        src_tokens=src_tokens,
    )

    literal: str | None = None
    if failed or coverage_warnings:
        src_lang = direction.split("→")[0]
        lex = _get_fallback(direction)
        # Use normalized text for ES (currency/abbrev expanded), raw for others
        fallback_input = normalized_text if src_lang == "es" else text
        raw_literal = lex.translate_literal(fallback_input)
        if raw_literal.lower() != translation.lower():
            literal = raw_literal

    self._send_json({
        "translation": translation,
        "direction": direction,
        "normalized_input": normalized_text if normalized_text != text else None,
        "coverage_warnings": coverage_warnings or None,
        "low_confidence": failed or None,
        "literal_fallback": literal,
        # Attention heatmap data
        "attention": {
            "matrix":     attn_mat,    # (T_out, T_src)
            "src_tokens": src_tokens,  # BPE pieces of the source
            "tgt_tokens": tgt_tokens,  # BPE pieces of the output
            "entropy":    entropy,     # per-output-token confidence
        } if attn_mat else None,
    })


_ROUTES = {
    "/api/detect": _detect,
    "/api/grammar": _grammar,
    "/api/morphology": _morphology,
    "/api/extract": _extract,
    "/api/add_card": _add_card,
    "/api/review": _review,
    "/api/exercise": _exercise,
    "/api/delete_card": _delete_card,
    "/api/translate":   _translate,
    "/api/conjugate":   _conjugate,
    "/api/lookup_form": _lookup_form,
    "/api/vad":         _vad,
}


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"LinguaForge web demo running at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
