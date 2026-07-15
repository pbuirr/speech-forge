"""
speech_engine — LinguaForge Phase 1-4

Public API for audio preprocessing and feature extraction.
All other modules (model_ctc, model_transformer, train, decode, evaluate)
will be added in later phases.
"""

from speech_engine.audio import (
    AudioMetadata,
    load_audio,
    to_mono,
    resample,
    normalize_amplitude,
    save_audio,
    get_metadata,
    preprocess,
)
from speech_engine.features import (
    log_mel_spectrogram,
    visualize_spectrogram,
    pad_or_truncate,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    SAMPLE_RATE,
)

__all__ = [
    # audio
    "AudioMetadata",
    "load_audio",
    "to_mono",
    "resample",
    "normalize_amplitude",
    "save_audio",
    "get_metadata",
    "preprocess",
    # features
    "log_mel_spectrogram",
    "visualize_spectrogram",
    "pad_or_truncate",
    # constants
    "N_MELS",
    "N_FFT",
    "HOP_LENGTH",
    "SAMPLE_RATE",
]
