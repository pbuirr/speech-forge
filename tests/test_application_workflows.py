import json
import os
import pytest
from speech_engine.flashcard import FlashcardDeck
from test_morphology_integration import http_json


def test_multilingual_translation_routes_and_capabilities(tmp_path, monkeypatch):
    from webapp import server

    multi = tmp_path / "multilingual.pt"
    multi.touch()
    mono = tmp_path / "custom.pt"
    mono.touch()
    monkeypatch.setattr(server, "MULTILINGUAL_CHECKPOINT", multi)
    monkeypatch.setattr(server, "NMT_CHECKPOINTS", {"it→en": mono})
    monkeypatch.setattr(server, "_nmt_cache", {})
    monkeypatch.setattr(server, "_nmt_metadata", {})
    from unittest.mock import Mock

    model = Mock()
    loaded = []

    def load(path):
        loaded.append(path)
        return (model, "source", "target", {})

    monkeypatch.setattr(server, "load_checkpoint", load)
    assert server._get_nmt("es→tr") is None
    assert not loaded
    assert server._get_nmt("es→en") == (model, "source", "target")
    assert server._get_nmt("fr→en") == (model, "source", "target")
    assert loaded == [multi]
    result = http_json("/api/languages")
    assert result["translation_directions"] == ["de→en", "es→en", "fr→en", "it→en"]
    languages = {item["code"]: item for item in result["languages"]}
    assert languages["de"]["caps"]["nmt_src"]
    assert languages["en"]["caps"]["nmt_tgt"]
    assert not languages["tr"]["caps"]["nmt_tgt"]
    missing = http_json(
        "/api/translate", {"text": "hola", "direction": "es→tr"}, expected_status=404
    )
    assert "it→en" in missing["error"]


def test_fallback_uses_only_declared_training_partition(tmp_path, monkeypatch):
    from webapp import server

    path = tmp_path / "train.tsv"
    path.write_text("hola\thello\n", encoding="utf-8")
    monkeypatch.setattr(server, "_nmt_metadata", {"es→en": {"training_corpus": str(path)}})
    monkeypatch.setattr(server, "_lex_cache", {})
    monkeypatch.setattr(server, "MULTILINGUAL_CHECKPOINT", tmp_path / "absent.pt")
    captured = []

    def build(paths, **kwargs):
        captured.extend(paths)
        return "fallback"

    monkeypatch.setattr(server.LexicalFallback, "from_corpora", build)
    assert server._get_fallback("es→en") == "fallback"
    assert captured == [path]


def test_installed_translation_replacement_reloads_model_and_training_fallback(
    tmp_path, monkeypatch
):
    from webapp import server
    from unittest.mock import Mock

    path = tmp_path / "model.pt"
    path.write_bytes(b"first")
    monkeypatch.setattr(server, "MULTILINGUAL_CHECKPOINT", tmp_path / "missing.pt")
    monkeypatch.setattr(server, "NMT_CHECKPOINTS", {"es→en": path})
    monkeypatch.setattr(server, "_nmt_cache", {})
    monkeypatch.setattr(server, "_nmt_metadata", {})
    monkeypatch.setattr(server, "_nmt_fingerprints", {})
    monkeypatch.setattr(server, "_lex_cache", {"es→en": "old fallback"})
    loads = []

    def load(checkpoint):
        model = Mock()
        loads.append(model)
        return model, "src", "tgt", {"training_corpus": str(checkpoint) + str(len(loads))}

    monkeypatch.setattr(server, "load_checkpoint", load)
    assert server._get_nmt("es→en")[0] is loads[0]
    assert "es→en" not in server._lex_cache
    server._get_nmt("es→en")
    assert len(loads) == 1
    server._lex_cache["es→en"] = "stale"
    path.write_bytes(b"new checkpoint")
    assert server._get_nmt("es→en")[0] is loads[1]
    assert "es→en" not in server._lex_cache
    assert server._nmt_metadata["es→en"]["training_corpus"].endswith("2")


def test_translation_rejects_unsupported_size_and_beam(monkeypatch):
    import torch
    from types import SimpleNamespace
    from webapp import server

    model = SimpleNamespace(
        parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]), _config={"max_src_len": 256}
    )
    vocab = SimpleNamespace(encode=lambda *args, **kwargs: list(range(300)))
    monkeypatch.setattr(server, "_get_nmt", lambda direction: (model, vocab, vocab))
    response = http_json(
        "/api/translate", {"text": "bonjour", "direction": "fr→en"}, expected_status=413
    )
    assert "256-token limit" in response["error"]
    assert (
        "beam must"
        in http_json("/api/translate", {"text": "bonjour", "beam": 9}, expected_status=400)["error"]
    )


def test_unicode_deck_atomic_save_and_failure_preserves_original(tmp_path, monkeypatch):
    deck = FlashcardDeck("多言語")
    deck.add("Çocuklar", "children — أطفال", lang="tr")
    path = tmp_path / "deck.json"
    deck.save(path)
    original = path.read_bytes()
    assert json.loads(original.decode("utf-8"))["name"] == "多言語"
    assert FlashcardDeck.load(path).all()[0].front == "Çocuklar"
    deck.add("学校", "school", lang="ja")

    def fail_replace(*args):
        raise OSError("simulated disk replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        deck.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_deck_returns_json_error(tmp_path, monkeypatch):
    from webapp import server

    path = tmp_path / "deck.json"
    path.write_text("invalid", encoding="utf-8")
    monkeypatch.setattr(server, "DECK_PATH", path)
    assert "Deck unavailable" in http_json("/api/deck", expected_status=500)["error"]


def test_legacy_windows_deck_can_migrate_to_utf8(tmp_path, monkeypatch):
    import locale

    deck = FlashcardDeck("français")
    deck.add("école", "school", lang="fr")
    path = tmp_path / "legacy.json"
    deck.save(path)
    path.write_bytes(path.read_text(encoding="utf-8").encode("cp1252"))
    monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")
    loaded = FlashcardDeck.load(path)
    assert loaded.all()[0].front == "école"
    loaded.save(path)
    assert "école" in path.read_text(encoding="utf-8")


def test_vad_honors_sample_rate_and_rejects_invalid_pcm():
    import numpy as np
    from speech_engine.vad import VoiceActivityDetector

    samples = np.zeros(8000, dtype=np.float32)
    result = http_json("/api/vad", {"samples": samples.tolist(), "sr": 8000})
    assert result["total_secs"] == 1
    assert result["n_frames"] == len(
        VoiceActivityDetector(sr=8000).frame_features(samples)["times"]
    )
    for data in [
        {"samples": [0.0], "sr": 0},
        {"samples": [[0.0]], "sr": 16000},
        {"samples": [float("nan")], "sr": 16000},
        {"samples": [2.0], "sr": 16000},
    ]:
        assert "error" in http_json("/api/vad", data, expected_status=400)


@pytest.mark.parametrize("code", ["en", "tr", "ja", "en-US"])
def test_matching_language_selections_preserve_original_without_model(code, monkeypatch):
    from webapp import server

    def forbidden(direction):
        raise AssertionError("Identity must not load or run NMT")

    monkeypatch.setattr(server, "_get_nmt", forbidden)
    text = "  The teacher did not call Pedro.\n2026 → 2027.  "
    result = http_json("/api/translate", {"text": text, "direction": code + "→" + code})
    assert result["translation"] == text == result["raw_translation"]
    assert result["method"] == "identity" and result["attention"] is None
    assert "No translation or correction" in result["model_status"]
