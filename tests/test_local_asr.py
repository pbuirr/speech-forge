import io
import wave

import numpy as np
import pytest

from speech_engine.local_asr import LocalCTCRecognizer, read_pcm_wav


def wav_bytes(samples=1600, rate=16000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(np.full(samples * channels, 16384, dtype="<i2").tobytes())
    return buffer.getvalue()


def test_wav_validation_and_training_normalization():
    result = read_pcm_wav(wav_bytes(channels=2))
    assert result.shape == (1600,)
    assert np.allclose(result, 0.95)
    for payload in [b"broken", wav_bytes(rate=8000), wav_bytes(samples=10), wav_bytes()[:-4]]:
        with pytest.raises(ValueError):
            read_pcm_wav(payload)


def test_original_ctc_checkpoint_inference_roundtrip(tmp_path, monkeypatch):
    # Synthetic weights verify serialization/inference, NOT speech accuracy.
    import torch
    from speech_engine.model_ctc import CTCModel
    from speech_engine.tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    model = CTCModel(tokenizer.vocab_size, cnn_channels=(2, 2, 2), hidden_size=4, num_lstm_layers=1)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(-20)
        model.classifier.bias[tokenizer.encode("a")[0]] = 20
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"cnn_channels": (2, 2, 2), "hidden_size": 4, "num_lstm_layers": 1},
        },
        path,
    )
    recognizer = LocalCTCRecognizer(path, tokenizer_path, "en")
    assert recognizer.status()["available"]
    assert not recognizer.status()["accuracy_verified"]
    assert recognizer.status()["checkpoint_name"] == "model.pt"
    assert recognizer.status()["checkpoint_sha256"] is None
    assert recognizer.transcribe(read_pcm_wav(wav_bytes())) == "a"
    import hashlib

    loaded_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert recognizer.status()["checkpoint_sha256"] == loaded_digest
    assert recognizer.transcribe(read_pcm_wav(wav_bytes())) == "a"
    # Metadata-bound training-only fusion; legacy checkpoints remain greedy.
    import hashlib

    manifest = tmp_path / "train.tsv"
    manifest.write_text("unused.wav\ta\n", encoding="utf-8")
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["config"]["asr_decoder"] = {
        "beam_width": 10,
        "lm_weight": 0.2,
        "token_reward": 0.2,
        "training_manifest_path": str(manifest),
        "training_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    torch.save(checkpoint, path)
    calls = []

    def beam(logits, tok, **options):
        assert options["lm"] is not None and options["lm_weight"] == 0.2
        calls.append(options)
        return ["a"]

    monkeypatch.setattr("speech_engine.decode.beam_decode", beam)
    assert (
        LocalCTCRecognizer(path, tokenizer_path, "en").transcribe(read_pcm_wav(wav_bytes())) == "a"
    )
    assert len(calls) == 1
    assert calls[0]["lm"].context_size == 2
    checkpoint["config"]["asr_decoder"]["lm_order"] = 5
    torch.save(checkpoint, path)
    assert (
        LocalCTCRecognizer(path, tokenizer_path, "en").transcribe(read_pcm_wav(wav_bytes())) == "a"
    )
    assert calls[-1]["lm"].context_size == 4
    manifest.write_text("unused.wav\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint"):
        LocalCTCRecognizer(path, tokenizer_path, "en").transcribe(read_pcm_wav(wav_bytes()))


def test_transcription_endpoint_unavailable_and_invalid_audio(tmp_path, monkeypatch):
    from webapp import server
    from test_morphology_integration import http_json

    monkeypatch.setattr(
        server,
        "local_asr",
        LocalCTCRecognizer(tmp_path / "missing.pt", tmp_path / "missing.json", "en"),
    )
    assert not http_json("/api/speech")["available"]
    for payload, status in [(wav_bytes(), 503), (b"invalid", 400)]:
        raw = (
            f"POST /api/transcribe HTTP/1.0\r\nContent-Type: audio/wav\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n"
        ).encode() + payload

        class Connection:
            def __init__(self):
                self.output = bytearray()

            def makefile(self, *args):
                return io.BytesIO(raw)

            def sendall(self, data):
                self.output.extend(data)

        connection = Connection()
        server.Handler(connection, ("127.0.0.1", 0), None)
        assert bytes(connection.output).startswith(f"HTTP/1.0 {status}".encode())


def test_ctc_rejects_same_size_tokenizer_with_different_character_assignments(tmp_path):
    import hashlib
    import json
    import torch
    from speech_engine.tokenizer import CharTokenizer

    tokenizer_path = tmp_path / "tokenizer.json"
    CharTokenizer().save(tokenizer_path)
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": {"tokenizer_sha256": digest}, "model": {}}, checkpoint)
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    data["char_to_id"]["a"], data["char_to_id"]["b"] = (
        data["char_to_id"]["b"],
        data["char_to_id"]["a"],
    )
    tokenizer_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        LocalCTCRecognizer(checkpoint, tokenizer_path, "en").transcribe(
            np.zeros(1600, dtype=np.float32)
        )
