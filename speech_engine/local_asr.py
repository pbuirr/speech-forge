"""Transcribe a local WAV file with an installed CTC checkpoint."""

from __future__ import annotations

import io
import hashlib
import threading
import wave
from pathlib import Path

import numpy as np

MAX_WAV_BYTES = 2_000_000


def read_pcm_wav(payload: bytes) -> np.ndarray:
    if len(payload) > MAX_WAV_BYTES:
        raise ValueError("WAV exceeds the 2 MB limit")
    try:
        with wave.open(io.BytesIO(payload), "rb") as audio:
            if (
                audio.getframerate() != 16000
                or audio.getsampwidth() != 2
                or audio.getnchannels() not in (1, 2)
                or audio.getcomptype() != "NONE"
            ):
                raise ValueError("Use a 16 kHz, 16-bit PCM WAV with one or two channels")
            frames, channels = audio.getnframes(), audio.getnchannels()
            if not 400 <= frames <= 480000:
                raise ValueError("Audio must be between 0.025 and 30 seconds")
            raw = audio.readframes(frames)
            if len(raw) != frames * channels * 2:
                raise ValueError("WAV audio is truncated")
    except (wave.Error, EOFError) as error:
        raise ValueError("Invalid PCM WAV file") from error
    samples = (
        np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1)
        / 32768
    )
    peak = float(np.abs(samples).max())
    # Match the original training preprocessing.
    return samples / peak * 0.95 if peak >= 1e-6 else samples


class LocalCTCRecognizer:
    def __init__(self, checkpoint: Path, tokenizer: Path, language: str):
        self.checkpoint, self.tokenizer_path, self.language = checkpoint, tokenizer, language
        self._runtime = None
        self._checkpoint_sha256 = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        available = self.checkpoint.is_file() and self.tokenizer_path.is_file()
        return {
            "available": available,
            "language": self.language,
            "engine": "local_ctc",
            "checkpoint_name": self.checkpoint.name,
            "checkpoint_sha256": self._checkpoint_sha256,
            "accuracy_verified": False,
            "message": "Configured local CTC checkpoint; recognition accuracy is not verified."
            if available
            else "Local CTC unavailable: install your trained checkpoint and matching tokenizer.",
            "format": "16 kHz, 16-bit PCM WAV; mono or stereo; up to 30 seconds",
        }

    def transcribe(self, samples: np.ndarray) -> str:
        from speech_engine.model_ctc import CTCModel
        from speech_engine.tokenizer import CharTokenizer
        from speech_engine.features import log_mel_spectrogram
        from speech_engine.decode import greedy_decode
        from speech_engine.decode import beam_decode
        import torch

        with self._lock:
            if self._runtime is None:
                tokenizer = CharTokenizer.load(self.tokenizer_path)
                payload = self.checkpoint.read_bytes()
                checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                config = checkpoint["config"]
                if config.get("tokenizer_sha256"):
                    digest = hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest()
                    if digest != config["tokenizer_sha256"]:
                        raise ValueError(
                            "Tokenizer fingerprint does not match the trained CTC checkpoint"
                        )
                if (
                    config.get("asr_language")
                    and config["asr_language"].split("-")[0] != self.language.split("-")[0]
                ):
                    raise ValueError(
                        "Configured speech language does not match checkpoint training"
                    )
                model = CTCModel(
                    vocab_size=tokenizer.vocab_size,
                    cnn_channels=tuple(config["cnn_channels"]),
                    hidden_size=config["hidden_size"],
                    num_lstm_layers=config["num_lstm_layers"],
                    time_reduction=config.get("time_reduction", 2),
                )
                model.load_state_dict(checkpoint["model"], strict=True)
                model.eval()
                decoder = config.get("asr_decoder", {})
                lm = None
                if decoder:
                    from speech_engine.dataset import load_manifest
                    from speech_engine.lm import NgramLM
                    from speech_engine.tokenizer import normalize_text

                    manifest = Path(decoder["training_manifest_path"])
                    if (
                        hashlib.sha256(manifest.read_bytes()).hexdigest()
                        != decoder["training_manifest_sha256"]
                    ):
                        raise ValueError("Speech decoding training manifest fingerprint changed")
                    if (
                        not 1 <= decoder["beam_width"] <= 32
                        or not 0 <= decoder["lm_weight"] <= 1
                        or not 0 <= decoder["token_reward"] <= 1
                    ):
                        raise ValueError("Invalid speech decoding configuration")
                    lm = NgramLM(min_count=1, order=decoder.get("lm_order", 3))
                    lm.train(
                        [
                            list(normalize_text(text, tokenizer.normalization))
                            for _, text in load_manifest(manifest)
                        ]
                    )
                    lm.finalize()
                self._runtime = model, tokenizer, decoder, lm
                self._checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
            model, tokenizer, decoder, lm = self._runtime
            with torch.inference_mode():
                mel = torch.from_numpy(log_mel_spectrogram(samples)).unsqueeze(0)
                logits = model(mel)
                if decoder:
                    return beam_decode(
                        logits,
                        tokenizer,
                        beam_width=decoder["beam_width"],
                        lm=lm,
                        lm_weight=decoder["lm_weight"],
                        token_reward=decoder["token_reward"],
                    )[0].strip()
                return greedy_decode(logits, tokenizer)[0].strip()
