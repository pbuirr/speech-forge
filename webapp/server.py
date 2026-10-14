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
import argparse
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
from speech_engine.understanding import LinguisticPipeline, morphology_payload, grammar_payload
from speech_engine.conjugation import (
    conjugate,
    lookup_form,
    TENSE_NAMES,
    PERSONS,
    EN_PERSONS,
)
from speech_engine.nmt import load_checkpoint
from speech_engine.preprocessing import (
    normalize_es_input,
    postprocess_en_output,
    detect_translation_failure,
)
from speech_engine.lexical_fallback import LexicalFallback
from speech_engine.vad import VoiceActivityDetector
from speech_engine.local_asr import LocalCTCRecognizer, read_pcm_wav, MAX_WAV_BYTES
from speech_engine.input_evidence import input_evidence

STATIC_DIR = Path(__file__).resolve().parent / "static"
DECK_PATH = PROJECT_ROOT / "data" / "decks" / "web.json"
PORT = 8765
NMT_DEVICE = "cpu"
local_asr = LocalCTCRecognizer(
    PROJECT_ROOT / "checkpoints" / "best.pt", PROJECT_ROOT / "data" / "tokenizer.json", "en"
)

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

# Multilingual model covers ES/FR/DE → EN with language-token conditioning.
# Falls back to monolingual checkpoints if the multilingual one is absent.
MULTILINGUAL_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "nmt_multilingual_16k_ep100.pt"
MULTILINGUAL_LANGS = {"es", "fr", "de"}
LANG_TOKENS = {"es": "<ES>", "fr": "<FR>", "de": "<DE>"}

NMT_CHECKPOINTS = {
    "es→en": PROJECT_ROOT / "data" / "models" / "nmt_es_en.pt",
    "fr→en": PROJECT_ROOT / "data" / "models" / "nmt_fr_en.pt",
}

# Engines are stateless / cheap to construct — one shared instance per process.
detector = LanguageDetector()
checker = GrammarChecker()
morph = MorphologyEngine()
vad = VoiceActivityDetector()

# Lexical fallback cache — trained per language pair on first request.
# Training IBM Model 1 on 200k pairs takes ~10s; cached thereafter.
_lex_cache: dict[str, object] = {}


def _get_fallback(direction: str):
    """Return (or build) the IBM Model 1 fallback for a language pair."""
    if direction not in _lex_cache:
        src, tgt = direction.split("→")
        corpora = []
        metadata = _nmt_metadata.get(
            "__multi__" if _uses_multilingual_nmt(direction) else direction, {}
        )
        training_corpus = metadata.get("training_corpus")
        if training_corpus:
            # A model's held-out partitions must not enter its statistical fallback.
            path = Path(training_corpus)
            if path.is_file():
                corpora.append(path)
        else:
            real = CORPUS_PATHS.get(direction)
            if real and real.exists():
                corpora.append(real)
            curated = _CURATED.get(direction)
            if curated and curated.exists():
                corpora.append(curated)
        _lex_cache[direction] = LexicalFallback.from_corpora(corpora, src_lang=src, tgt_lang=tgt)
    return _lex_cache[direction]


# NMT models — loaded lazily on first /api/translate call.
# _nmt_cache["__multi__"] holds the shared multilingual model.
_nmt_cache: dict[str, tuple] = {}
_nmt_metadata: dict[str, dict] = {}
_nmt_fingerprints: dict[str, tuple] = {}


def _uses_multilingual_nmt(direction: str) -> bool:
    return (
        direction in {f"{lang}→en" for lang in MULTILINGUAL_LANGS}
        and MULTILINGUAL_CHECKPOINT.exists()
    )


def _available_nmt_directions() -> list[str]:
    directions = {d for d, path in NMT_CHECKPOINTS.items() if path.exists()}
    if MULTILINGUAL_CHECKPOINT.exists():
        directions.update(f"{lang}→en" for lang in MULTILINGUAL_LANGS)
    return sorted(directions)


def _get_nmt(direction: str):
    """Return (model, src_vocab, tgt_vocab) for the given direction.

    Prefers the multilingual 16k model for ES/FR/DE → EN when available.
    Falls back to per-language monolingual checkpoints, then returns None.
    """
    use_multi = _uses_multilingual_nmt(direction)

    cache_key = "__multi__" if use_multi else direction
    ckpt_path = MULTILINGUAL_CHECKPOINT if use_multi else NMT_CHECKPOINTS.get(direction)
    if ckpt_path is None or not ckpt_path.exists():
        _nmt_cache.pop(cache_key, None)
        _nmt_metadata.pop(cache_key, None)
        _nmt_fingerprints.pop(cache_key, None)
        return None
    stat = ckpt_path.stat()
    fingerprint = (str(ckpt_path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache_key not in _nmt_cache or _nmt_fingerprints.get(cache_key) != fingerprint:
        # Load before publishing cache entries; invalid replacements produce errors.
        model, src_vocab, tgt_vocab, metadata = load_checkpoint(ckpt_path)
        model.to(NMT_DEVICE)
        _nmt_metadata[cache_key] = metadata
        _nmt_cache[cache_key] = (model, src_vocab, tgt_vocab)
        _nmt_fingerprints[cache_key] = fingerprint
        for pair in [f"{lang}→en" for lang in MULTILINGUAL_LANGS] if use_multi else [direction]:
            _lex_cache.pop(pair, None)

    return _nmt_cache.get(cache_key)


def _load_deck() -> FlashcardDeck:
    if DECK_PATH.exists():
        return FlashcardDeck.load(DECK_PATH)
    DECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    return FlashcardDeck(name="web")


_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
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
        if path == "/api/speech":
            return self._send_json(local_asr.status())
        if path == "/":
            return self._send_static("index.html")
        if path == "/api/deck":
            try:
                deck = _load_deck()
                due_ids = {c.id for c in deck.due()}
                cards = [dict(c.to_dict(), is_due=c.id in due_ids) for c in deck.all()]
                return self._send_json({"name": deck.name, "cards": cards, "stats": deck.stats()})
            except (OSError, ValueError) as error:
                return self._send_json({"error": f"Deck unavailable: {error}"}, 500)
        if path == "/api/languages":
            from speech_engine.language_detector import LANG_DB

            _GRAMMAR = {"en", "es", "fr"}
            directions = _available_nmt_directions()
            _nmt_src = {d.split("→")[0] for d in directions}
            _nmt_tgt = {d.split("→")[1] for d in directions}
            langs = [
                {
                    "code": code,
                    "name": defn.name,
                    "script": defn.script,
                    "dialect_of": defn.dialect_of,
                    "caps": {
                        "detection": True,
                        "morphology": morph.capabilities(code)["morphology"] != "unavailable",
                        "grammar": code in _GRAMMAR,
                        "nmt_src": code in _nmt_src,
                        "nmt_tgt": code in _nmt_tgt,
                    },
                }
                for code, defn in LANG_DB.items()
            ]
            return self._send_json(
                {"languages": langs, "total": len(langs), "translation_directions": directions}
            )
        self._send_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/transcribe":
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_WAV_BYTES:
                    return self._send_json({"error": "Provide a PCM WAV file of at most 2 MB"}, 413)
                payload = self.rfile.read(length)
                try:
                    samples = read_pcm_wav(payload)
                except ValueError as error:
                    return self._send_json({"error": str(error)}, 400)
                if not local_asr.status()["available"]:
                    return self._send_json({"error": local_asr.status()["message"]}, 503)
                transcript = local_asr.transcribe(samples)
                identity = local_asr.status()
                return self._send_json(
                    {
                        "transcript": transcript,
                        "language": local_asr.language,
                        "checkpoint_name": identity["checkpoint_name"],
                        "checkpoint_sha256": identity["checkpoint_sha256"],
                        "engine": "local_ctc",
                        "confidence": None,
                        "input_evidence": input_evidence(
                            {
                                "method": "local_ctc",
                                "language": local_asr.language,
                                "model_sha256": identity["checkpoint_sha256"],
                            }
                        ),
                        "duration_s": len(samples) / 16000,
                    }
                )
            data = self._read_json()
            handler = _ROUTES.get(path)
            if handler is None:
                return self._send_json({"error": f"no such endpoint: {path}"}, 404)
            return handler(self, data)
        except Exception as e:  # noqa: BLE001 — surface the error to the page, don't crash the server
            return self._send_json({"error": str(e)}, 500)


# ── POST endpoint implementations ────────────────────────────────────────────


def _detect(self: Handler, data: dict) -> None:
    try:
        evidence = input_evidence(data.get("input_evidence"))
    except ValueError as error:
        return self._send_json({"error": str(error)}, 400)
    text = data.get("text", "")
    result = detector.detect(text)
    self._send_json(
        {
            "language": result.language,
            "input_evidence": evidence,
            "evidence_scope": "generated_text" if evidence else "supplied_text",
            "automatic_routing_allowed": evidence is None,
            "name": result.name,
            "dialect": result.dialect,
            "script": result.script,
            "confidence": result.confidence,
            "is_mixed": result.is_mixed,
            "top": [
                {"code": m.code, "name": m.name, "confidence": m.confidence} for m in result.top
            ],
        }
    )


def _grammar(self: Handler, data: dict) -> None:
    try:
        evidence = input_evidence(data.get("input_evidence"))
    except ValueError as error:
        return self._send_json({"error": str(error)}, 400)
    text = data.get("text", "")
    lang = data.get("lang", "en")
    r = checker.check(text, lang=lang)
    payload = grammar_payload(r, checker)
    payload["input_evidence"] = evidence
    self._send_json(payload)


def _morphology(self: Handler, data: dict) -> None:
    try:
        evidence = input_evidence(data.get("input_evidence"))
    except ValueError as error:
        return self._send_json({"error": str(error)}, 400)
    text = data.get("text", "")
    lang = data.get("lang", "en")
    r = morph.analyze(text, lang=lang)
    payload = morphology_payload(r, morph)
    payload["input_evidence"] = evidence
    self._send_json(payload)


def _analyze(self: Handler, data: dict) -> None:
    try:
        result = LinguisticPipeline(morph, checker, detector).analyze(
            data.get("text", ""), data.get("lang"), data.get("input_evidence")
        )
    except ValueError as error:
        return self._send_json({"error": str(error)}, 400)
    self._send_json(result.to_dict(morph, checker))


def _extract(self: Handler, data: dict) -> None:
    text = data.get("text", "")
    lang = data.get("lang", "en")
    deck = _load_deck()
    try:
        result = LinguisticPipeline(morph, checker, detector).analyze(
            text, lang, data.get("input_evidence")
        )
        new_cards = deck.extract_from_understanding(result, morph, checker)
    except ValueError as error:
        return self._send_json({"error": str(error)}, 400)
    deck.save(DECK_PATH)
    self._send_json({"added": len(new_cards), "cards": [c.to_dict() for c in new_cards]})


def _add_card(self: Handler, data: dict) -> None:
    analysis = data.get("analysis")
    if analysis is not None and not isinstance(analysis, dict):
        return self._send_json({"error": "analysis must be an object"}, 400)
    context = analysis.get("context") if analysis else None
    if context is not None and not isinstance(context, dict):
        return self._send_json({"error": "analysis context must be an object"}, 400)
    if analysis is not None and any(
        v is not None
        for v in (
            data.get("input_evidence"),
            analysis.get("input_evidence"),
            (context or {}).get("input_evidence"),
        )
    ):
        return self._send_json(
            {
                "error": "Automatic learning cards are unavailable for model-generated input; add manually checked material"
            },
            400,
        )
    deck = _load_deck()
    card = deck.add(
        front=data["front"],
        back=data["back"],
        lang=data.get("lang", "en"),
        tags=(
            ["auto", "needs-review"]
            + (["pos-uncertain"] if analysis.get("pos_review_recommended") is True else [])
        )
        if analysis is not None
        else ["manual"],
        analysis=analysis,
        example=data.get("example"),
    )
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
        ex = generate_fill_in_blank(card, morph_engine=morph)
        return self._send_json(
            {
                "kind": "fill",
                "prompt": ex.prompt,
                "answer": ex.answer,
                "lang": ex.lang,
                "source": ex.source,
                "span": list(ex.span) if ex.span is not None else None,
                "span_unit": "unicode_codepoints",
            }
        )
    if kind == "mcq":
        ex = generate_multiple_choice(card, deck)
        if ex is None:
            return self._send_json(
                {
                    "error": "No distinct same-language distractors; add more cards or use a fill exercise."
                },
                400,
            )
        return self._send_json(
            {
                "kind": "mcq",
                "prompt": ex.prompt,
                "choices": ex.choices,
                "answer_index": ex.answer_index,
            }
        )
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
    sr = data.get("sr", 16_000)
    if not samples:
        return self._send_json({"error": "samples required"}, 400)
    if type(sr) is not int or sr < 1000 or sr > 192000:
        return self._send_json(
            {"error": "sr must be an integer sample rate from 1000 to 192000 Hz"}, 400
        )

    audio = np.array(samples, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        return self._send_json(
            {"error": "samples must be finite mono PCM values between -1 and 1"}, 400
        )
    detector_for_audio = vad if sr == vad.sr else VoiceActivityDetector(sr=sr)
    segments = detector_for_audio.detect(audio)
    features = detector_for_audio.frame_features(audio)
    self._send_json(
        {
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "duration": round(s.duration, 3),
                }
                for s in segments
            ],
            "n_frames": len(features["times"]),
            "total_secs": round(float(len(audio) / sr), 3),
            "sr": sr,
        }
    )


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

    self._send_json(
        {
            "infinitive": table.infinitive,
            "english": table.english,
            "gerund_es": table.gerund_es,
            "gerund_en": table.gerund_en,
            "participle_es": table.participle_es,
            "participle_en": table.participle_en,
            "verb_class": table.verb_class,
            "is_regular": table.is_regular,
            "notes": table.notes,
            "tenses": tenses_out,
        }
    )


def _lookup_form(self: Handler, data: dict) -> None:
    """POST /api/lookup_form { "form": "fue" } — reverse-lookup a conjugated form."""
    form = data.get("form", "").strip().lower()
    if not form:
        return self._send_json({"error": "form required"}, 400)
    matches = lookup_form(form)
    self._send_json({"form": form, "matches": matches})


def _translate(self: Handler, data: dict) -> None:
    import torch

    original_text = data.get("text", "")
    try:
        evidence = input_evidence(data.get("input_evidence"))
    except ValueError as error:
        return self._send_json({"error": str(error)}, 400)
    if not isinstance(original_text, str):
        return self._send_json({"error": "text must be a string"}, 400)
    text = original_text.strip()
    direction = data.get("direction", "es→en")
    beam = int(data.get("beam", 4))
    if not 1 <= beam <= 8:
        return self._send_json({"error": "beam must be between 1 and 8"}, 400)

    if not text:
        return self._send_json({"error": "text is required"}, 400)

    parts = direction.split("→") if isinstance(direction, str) else []
    if (
        len(parts) == 2
        and parts[0].strip()
        and parts[0].strip().lower() == parts[1].strip().lower()
    ):
        from speech_engine.linguistic_resources import language_code

        try:
            language_code(parts[0].strip())
        except ValueError as error:
            return self._send_json({"error": str(error)}, 400)
        return self._send_json(
            {
                "translation": original_text,
                "raw_translation": original_text,
                "direction": direction,
                "method": "identity",
                "attention": None,
                "input_evidence": evidence,
                "output_evidence": evidence,
                "model_status": "Source and target selections match; showing your original text. No translation or correction was performed.",
            }
        )
    result = _get_nmt(direction)
    if result is None:
        multi_note = (
            " (multilingual model not found)" if MULTILINGUAL_CHECKPOINT.exists() is False else ""
        )
        available = _available_nmt_directions()
        return self._send_json(
            {
                "input_evidence": evidence,
                "error": (
                    f"No trained model for {direction!r}{multi_note}. "
                    f"Available: {available or 'none yet'}"
                ),
            },
            404,
        )

    model, src_vocab, tgt_vocab = result
    metadata = _nmt_metadata.get(
        "__multi__" if _uses_multilingual_nmt(direction) else direction, {}
    )
    device = next(model.parameters()).device
    src_lang = direction.split("→")[0]
    is_multi = _uses_multilingual_nmt(direction)

    # ── Linguistic preprocessing ─────────────────────────────────────────────
    if direction == "es→en":
        prep = normalize_es_input(text)
        normalized_text = prep.normalized
    else:
        prep = None
        normalized_text = text

    bpe_ids = src_vocab.encode(normalized_text.lower(), add_bos=True, add_eos=True)
    if not bpe_ids:
        return self._send_json({"translation": ""})

    # Multilingual: prepend language token before BOS so encoder sees it first
    if is_multi:
        lang_tok = LANG_TOKENS[src_lang]
        lang_id = src_vocab._token_to_id.get(lang_tok)
        if lang_id is not None:
            src_ids = [lang_id] + bpe_ids
        else:
            src_ids = bpe_ids
    else:
        src_ids = bpe_ids

    max_source_tokens = model._config.get("max_src_len", 256)
    if len(src_ids) > max_source_tokens:
        return self._send_json(
            {
                "error": f"Translation input exceeds this model's {max_source_tokens}-token limit. Use a shorter passage."
            },
            413,
        )
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # ── Translation + attention in one shot ─────────────────────────────────
    attn_result = model.translate_with_attention(src_tensor, beam_size=beam)
    out_ids = attn_result["out_ids"]
    attn_mat = attn_result["attention"]  # (T_out, T_src) floats
    entropy = attn_result["entropy"]  # (T_out,) floats, 0=confident 1=uncertain

    translation = tgt_vocab.decode(out_ids)
    raw_translation = translation

    # Source tokens for the heatmap — skip the leading language token if present
    src_ids_for_labels = src_ids[1:] if is_multi else src_ids
    src_tokens = [src_vocab._id_to_token.get(i, "?") for i in src_ids_for_labels]
    # Target tokens aligned with the attention rows
    tgt_tokens = [tgt_vocab._id_to_token.get(i, "?") for i in out_ids]

    # ── Post-processing ──────────────────────────────────────────────────────
    translation = postprocess_en_output(translation)

    coverage_warnings: list[str] = []
    if prep is not None:
        failures = prep.coverage_failures(translation)
        if failures:
            coverage_warnings = [f.target_hint for f in failures]

    avg_entropy = (sum(entropy) / len(entropy)) if entropy else None
    failed = detect_translation_failure(
        normalized_text,
        translation,
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

    self._send_json(
        {
            "translation": translation,
            "input_evidence": evidence,
            "output_evidence": input_evidence(
                {
                    "method": "local_translation",
                    "language": direction.split("→")[-1],
                    "original_input_evidence": evidence.get("original_input_evidence", evidence)
                    if evidence
                    else None,
                }
            ),
            "raw_translation": raw_translation,
            "direction": direction,
            "model_status": metadata.get(
                "quality_status", "Local model; translation quality has not been verified here."
            ),
            "normalized_input": normalized_text if normalized_text != text else None,
            "coverage_warnings": coverage_warnings or None,
            "low_confidence": failed or bool(coverage_warnings) or None,
            "literal_fallback": literal,
            # Attention heatmap data
            "attention": {
                "matrix": attn_mat,  # (T_out, T_src)
                "src_tokens": src_tokens,  # BPE pieces of the source
                "tgt_tokens": tgt_tokens,  # BPE pieces of the output
                "entropy": entropy,  # per-output-token confidence
            }
            if attn_mat
            else None,
        }
    )


_ROUTES = {
    "/api/analyze": _analyze,
    "/api/detect": _detect,
    "/api/grammar": _grammar,
    "/api/morphology": _morphology,
    "/api/extract": _extract,
    "/api/add_card": _add_card,
    "/api/review": _review,
    "/api/exercise": _exercise,
    "/api/delete_card": _delete_card,
    "/api/translate": _translate,
    "/api/conjugate": _conjugate,
    "/api/lookup_form": _lookup_form,
    "/api/vad": _vad,
}


def main() -> None:
    global local_asr, NMT_DEVICE
    parser = argparse.ArgumentParser(description="LinguaForge local application")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--nmt-device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--asr-checkpoint", type=Path, default=local_asr.checkpoint)
    parser.add_argument("--asr-tokenizer", type=Path, default=local_asr.tokenizer_path)
    parser.add_argument(
        "--asr-language", default="en", help="Language supported by your trained CTC checkpoint"
    )
    parser.add_argument(
        "--linguistic-resources", type=Path, default=PROJECT_ROOT / "data" / "linguistics"
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    NMT_DEVICE = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.nmt_device == "auto"
        else args.nmt_device
    )
    if NMT_DEVICE == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --nmt-device cpu")
    local_asr = LocalCTCRecognizer(args.asr_checkpoint, args.asr_tokenizer, args.asr_language)
    for resource in morph.load_resources(args.linguistic_resources):
        print(f"Loaded {resource['language']}: {resource['forms']} corpus-attested forms")
    for language in morph.tagger.load_calibrations(args.linguistic_resources / "calibration"):
        print(f"Loaded development POS calibration: {language}")
    from speech_engine.dependency import load_dependency_models

    for language in load_dependency_models(morph, args.linguistic_resources / "syntax"):
        print(f"Loaded experimental local dependency parser: {language}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"LinguaForge web demo running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
