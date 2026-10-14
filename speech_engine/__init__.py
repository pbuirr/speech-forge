"""Speech, translation, linguistic analysis and learning components for LinguaForge.

Only the audio and feature helpers are re-exported here; import the other
modules directly (e.g. ``speech_engine.model_ctc``).
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
