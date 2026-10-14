import pytest
from speech_engine.input_evidence import input_evidence
from speech_engine.flashcard import FlashcardDeck
from speech_engine.understanding import LinguisticPipeline


def test_generated_input_cannot_be_upgraded_by_a_confidence_flag():
    value = input_evidence(
        {"method": "local_ctc", "language": "en-US", "status": "verified", "confidence": 1}
    )
    assert value["status"] == "predicted" and value["confidence"] is None
    assert value["review_required"] and value["language"] == "en-us"
    assert input_evidence(None) is None
    with pytest.raises(ValueError):
        input_evidence({"method": "verified"})
    assert (
        input_evidence({"method": "local_ctc", "model_sha256": "A" * 64})["model_sha256"]
        == "a" * 64
    )
    with pytest.raises(ValueError, match="fingerprint"):
        input_evidence({"method": "local_ctc", "model_sha256": "verified"})


def test_shared_analysis_and_learning_retain_generated_input(monkeypatch):
    from webapp import server
    from test_morphology_integration import http_json

    evidence = {"method": "local_ctc", "language": "en", "model_sha256": "a" * 64}
    payload = {"text": "John did not buy two apples.", "lang": "en", "input_evidence": evidence}
    result = http_json("/api/analyze", payload)
    assert result["input_evidence"]["status"] == "predicted"
    assert result["input_evidence"]["model_sha256"] == evidence["model_sha256"]
    assert result["morphology"]["input_evidence"] == result["input_evidence"]
    assert result["grammar"]["input_evidence"] == result["input_evidence"]
    assert result["text"] == payload["text"]
    for endpoint in ("detect", "morphology", "grammar"):
        assert http_json("/api/" + endpoint, payload)["input_evidence"] == result["input_evidence"]
    deck = FlashcardDeck()
    monkeypatch.setattr(server, "_load_deck", lambda: deck)
    monkeypatch.setattr(
        deck, "save", lambda _: pytest.fail("Unverified input must not persist cards")
    )
    response = http_json("/api/extract", payload, expected_status=400)
    assert "model-generated" in response["error"]
    assert deck.stats()["cards"] == 0
    response = http_json(
        "/api/add_card",
        {
            "front": "John",
            "back": "proper noun",
            "analysis": {"context": {"input_evidence": evidence}},
        },
        expected_status=400,
    )
    assert "model-generated" in response["error"]
    response = http_json("/api/translate", {**payload, "direction": "en→en"})
    assert response["method"] == "identity"
    assert response["input_evidence"] == response["output_evidence"]
    assert response["input_evidence"]["confidence"] is None
    response = http_json("/api/translate", {**payload, "direction": "en→tr"}, expected_status=404)
    assert response["input_evidence"]["review_required"]


def test_core_learning_guard_applies_to_translation_as_well_as_speech():
    from webapp import server

    pipeline = LinguisticPipeline(server.morph, server.checker, server.detector)
    result = pipeline.analyze(
        "Cats eat fish.", "en", {"method": "local_translation", "language": "en"}
    )
    deck = FlashcardDeck()
    with pytest.raises(ValueError, match="model-generated"):
        deck.extract_from_understanding(result, server.morph, server.checker)
    assert deck.stats()["cards"] == 0


def test_generated_text_cannot_silently_change_automatic_language_routing():
    from webapp import server

    pipeline = LinguisticPipeline(server.morph, server.checker, server.detector)
    text = "it the resit the secan bo falspon the gre the firs ther slis a bowat an ap norly wie el a grand"
    assert server.detector.detect(text).language != "en"  # reproduced corrupt transcript
    result = pipeline.analyze(text, "auto", {"method": "local_ctc", "language": "en"})
    assert result.lang == "en" and result.language_selection == "source_configured"
    assert (
        result.to_dict(server.morph, server.checker)["detection"]["automatic_routing_allowed"]
        is False
    )
    assert pipeline.analyze(text, "tr", {"method": "local_ctc", "language": "en"}).lang == "tr"
    with pytest.raises(ValueError, match="Select a source language"):
        pipeline.analyze(text, "auto", {"method": "browser_speech"})
    translation = pipeline.analyze(
        "Bonjour.", "auto", {"method": "local_translation", "language": "fr"}
    )
    assert translation.lang == "fr" and translation.language_selection == "source_configured"


def test_http_automatic_routing_uses_declared_speech_language():
    from test_morphology_integration import http_json

    payload = {
        "text": "it the resit the secan bo falspon the gre the firs ther slis a bowat an ap norly wie el a grand",
        "lang": "auto",
        "input_evidence": {"method": "local_ctc", "language": "en"},
    }
    result = http_json("/api/analyze", payload)
    assert result["lang"] == "en" and result["language_selection"] == "source_configured"
    assert result["detection"]["evidence_scope"] == "generated_text"
    assert not result["detection"]["automatic_routing_allowed"]
    detection = http_json("/api/detect", payload)
    assert detection["evidence_scope"] == "generated_text"
    assert not detection["automatic_routing_allowed"]
    payload["input_evidence"] = {"method": "browser_speech"}
    response = http_json("/api/analyze", payload, expected_status=400)
    assert "Select a source language" in response["error"]


def test_translation_reuse_retains_original_speech_provenance(monkeypatch):
    import torch
    from webapp import server
    from test_morphology_integration import http_json

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self._config = {"max_src_len": 256}

        def translate_with_attention(self, source, beam_size):
            return {"out_ids": [4], "attention": [[0.2] * source.shape[1]], "entropy": [0.2]}

    class Vocab:
        _id_to_token = {2: "<bos>", 3: "<eos>", 4: "hello"}

        def encode(self, text, **kwargs):
            return [2, 4, 3]

        def decode(self, ids):
            return "hello"

    monkeypatch.setattr(server, "_get_nmt", lambda _: (Model(), Vocab(), Vocab()))
    monkeypatch.setattr(server, "_uses_multilingual_nmt", lambda _: False)
    monkeypatch.setattr(
        server,
        "_get_fallback",
        lambda _: type("Fallback", (), {"translate_literal": lambda self, text: "hello"})(),
    )
    original = {
        "method": "local_ctc",
        "language": "es",
        "model_sha256": "a" * 64,
        "status": "verified",
    }
    translated = http_json(
        "/api/translate", {"text": "hola", "direction": "es→en", "input_evidence": original}
    )
    output = translated["output_evidence"]
    assert output["method"] == "local_translation" and output["language"] == "en"
    assert output["original_input_evidence"] == input_evidence(original)
    assert output["original_input_evidence"]["status"] == "predicted"
    analyzed = http_json(
        "/api/analyze",
        {"text": translated["translation"], "lang": "auto", "input_evidence": output},
    )
    assert analyzed["lang"] == "en"
    assert analyzed["input_evidence"] == output
    assert analyzed["input_evidence"]["review_required"]
