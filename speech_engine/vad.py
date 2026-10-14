"""Voice activity detection from short-time energy, zero-crossing rate and spectral flatness.

A four-state machine (silence → onset → speech → hangover) turns the frame
features into segments, with the noise floor estimated from the quietest
frames. Uses only NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


# Public types


@dataclass
class SpeechSegment:
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"SpeechSegment({self.start:.3f}s – {self.end:.3f}s, {self.duration:.3f}s)"


class VADState(Enum):
    SILENCE = auto()
    ONSET = auto()  # possible speech start — waiting for confirmation
    SPEECH = auto()
    HANGOVER = auto()  # possible speech end — waiting to see if it resumes


# Feature extraction


def _frame_audio(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Split 1-D audio into overlapping frames, shape (n_frames, frame_len)."""
    n = len(audio)
    n_frames = 1 + max(0, (n - frame_len) // hop_len)
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        chunk = audio[start : min(end, n)]
        frames[i, : len(chunk)] = chunk
    return frames


def _rms(frames: np.ndarray) -> np.ndarray:
    """Root-mean-square energy per frame, shape (n_frames,)."""
    return np.sqrt(np.mean(frames**2, axis=1) + 1e-12)


def _zcr(frames: np.ndarray, sr: int) -> np.ndarray:
    """Zero-crossing rate in crossings-per-second, shape (n_frames,)."""
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / 2
    frame_secs = frames.shape[1] / sr
    return crossings / (frame_secs + 1e-12)


def _spectral_flatness(frames: np.ndarray) -> np.ndarray:
    """
    Wiener entropy (spectral flatness) per frame.
    Close to 1 → noise-like; close to 0 → tonal/speech-like.
    """
    eps = 1e-12
    n = frames.shape[1]
    # Apply Hann window to reduce spectral leakage
    window = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(n) / (n - 1))
    wf = frames * window

    mag = np.abs(np.fft.rfft(wf, axis=1)) + eps
    geo = np.exp(np.mean(np.log(mag), axis=1))
    arith = np.mean(mag, axis=1)
    return geo / (arith + eps)


# Noise-floor estimator


class _NoiseFloor:
    """Running minimum of energy over a sliding window of frames."""

    def __init__(self, window_frames: int = 40, init_energy: float = 1e-4) -> None:
        self._buf = [init_energy] * window_frames
        self._w = window_frames

    def update(self, energy: float) -> float:
        self._buf.append(energy)
        if len(self._buf) > self._w:
            self._buf.pop(0)
        return min(self._buf)

    @property
    def floor(self) -> float:
        return min(self._buf)


# State machine


@dataclass
class _StateMachine:
    min_onset_frames: int = 3  # frames needed to confirm speech start
    hangover_frames: int = 8  # frames to wait before ending a segment
    min_speech_frames: int = 5  # discard segments shorter than this

    _state: VADState = field(default=VADState.SILENCE, init=False)
    _onset_count: int = field(default=0, init=False)
    _hangover_count: int = field(default=0, init=False)
    _speech_count: int = field(default=0, init=False)
    _segment_start: int = field(default=0, init=False)
    _completed: list[tuple[int, int]] = field(default_factory=list, init=False)

    def step(self, frame_idx: int, is_active: bool) -> None:
        s = self._state

        if s == VADState.SILENCE:
            if is_active:
                self._state = VADState.ONSET
                self._onset_count = 1
                self._segment_start = frame_idx

        elif s == VADState.ONSET:
            if is_active:
                self._onset_count += 1
                if self._onset_count >= self.min_onset_frames:
                    self._state = VADState.SPEECH
                    self._speech_count = self._onset_count
            else:
                # False alarm — not enough consecutive active frames
                self._state = VADState.SILENCE

        elif s == VADState.SPEECH:
            if is_active:
                self._speech_count += 1
            else:
                self._state = VADState.HANGOVER
                self._hangover_count = 1

        elif s == VADState.HANGOVER:
            if is_active:
                self._state = VADState.SPEECH
                self._speech_count += 1
                self._hangover_count = 0
            else:
                self._hangover_count += 1
                if self._hangover_count >= self.hangover_frames:
                    seg_len = frame_idx - self._segment_start
                    if seg_len >= self.min_speech_frames:
                        self._completed.append((self._segment_start, frame_idx))
                    self._state = VADState.SILENCE

    def flush(self, total_frames: int) -> None:
        """Flush any open segment at end of stream."""
        if self._state in (VADState.SPEECH, VADState.HANGOVER):
            seg_len = total_frames - self._segment_start
            if seg_len >= self.min_speech_frames:
                self._completed.append((self._segment_start, total_frames))
        self._state = VADState.SILENCE

    def pop_segments(self) -> list[tuple[int, int]]:
        segs, self._completed = self._completed, []
        return segs

    @property
    def current_state(self) -> VADState:
        return self._state


# Main detector


class VoiceActivityDetector:
    """
    Detects speech segments in a mono PCM waveform.

    Parameters
    ----------
    sr : int
        Sample rate (Hz).  Must match the audio passed to detect().
    frame_ms : int
        Analysis frame length in milliseconds (default 25 ms).
    hop_ms : int
        Frame hop in milliseconds (default 10 ms).
    energy_snr_db : float
        Minimum SNR above estimated noise floor to label a frame active.
    zcr_max : float
        Maximum zero-crossing rate (crossings/sec) for speech frames.
        Frames above this are likely silence or broadband noise.
    flatness_max : float
        Maximum spectral flatness for speech frames (0=tonal, 1=noise).
    min_onset_frames : int
        Consecutive active frames required to confirm a speech onset.
    hangover_frames : int
        Consecutive inactive frames before a segment is closed.
    min_speech_frames : int
        Segments shorter than this are discarded (suppresses clicks).
    """

    def __init__(
        self,
        sr: int = 16_000,
        frame_ms: int = 25,
        hop_ms: int = 10,
        energy_snr_db: float = 8.0,
        zcr_max: float = 3_000.0,
        flatness_max: float = 0.85,
        min_onset_frames: int = 3,
        hangover_frames: int = 8,
        min_speech_frames: int = 5,
    ) -> None:
        self.sr = sr
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop_len = int(sr * hop_ms / 1000)
        self.energy_snr = 10 ** (energy_snr_db / 20)  # linear ratio
        self.zcr_max = zcr_max
        self.flatness_max = flatness_max
        self._sm_kwargs = dict(
            min_onset_frames=min_onset_frames,
            hangover_frames=hangover_frames,
            min_speech_frames=min_speech_frames,
        )

    # Batch detect

    def detect(self, audio: np.ndarray) -> list[SpeechSegment]:
        """
        Detect speech segments in a mono float32 waveform.

        Parameters
        ----------
        audio : np.ndarray, shape (n_samples,), dtype float32
            Normalised to [-1, 1].

        Returns
        -------
        list[SpeechSegment]
            Each segment has .start and .end in seconds.
        """
        audio = np.asarray(audio, dtype=np.float32)
        frames = _frame_audio(audio, self.frame_len, self.hop_len)
        energy = _rms(frames)
        zcr = _zcr(frames, self.sr)
        flat = _spectral_flatness(frames)

        # Adaptive noise floor from the quietest 20% of frames
        sorted_e = np.sort(energy)
        floor_est = float(np.mean(sorted_e[: max(1, len(sorted_e) // 5)]))
        threshold = floor_est * self.energy_snr

        sm = _StateMachine(**self._sm_kwargs)
        for i, (e, z, f) in enumerate(zip(energy, zcr, flat)):
            active = e > threshold and z < self.zcr_max and f < self.flatness_max
            sm.step(i, active)

        sm.flush(len(frames))
        raw = sm.pop_segments()
        return [self._frames_to_seg(s, e) for s, e in raw]

    def frame_features(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-frame features for visualisation / debugging."""
        audio = np.asarray(audio, dtype=np.float32)
        frames = _frame_audio(audio, self.frame_len, self.hop_len)
        times = np.arange(len(frames)) * self.hop_len / self.sr
        return {
            "times": times,
            "energy": _rms(frames),
            "zcr": _zcr(frames, self.sr),
            "flatness": _spectral_flatness(frames),
        }

    # Streaming API

    def stream(self) -> "VADStream":
        """
        Return a streaming context that processes audio chunk by chunk.

        Usage:
            stream = vad.stream()
            for chunk in mic_chunks:
                state, segments = stream.push(chunk)
        """
        return VADStream(self)

    # Internal

    def _frames_to_seg(self, start_frame: int, end_frame: int) -> SpeechSegment:
        start_sec = start_frame * self.hop_len / self.sr
        end_sec = end_frame * self.hop_len / self.sr
        return SpeechSegment(start=start_sec, end=end_sec)


# Streaming wrapper


class VADStream:
    """
    Stateful streaming VAD.  Feed microphone chunks one at a time; get back
    the current VAD state and any segments that have just been completed.

    The noise floor is estimated from the first `warmup` chunks before
    thresholding begins.
    """

    def __init__(self, vad: VoiceActivityDetector, warmup: int = 5) -> None:
        self._vad = vad
        self._warmup = warmup
        self._chunk_count = 0
        self._buffer = np.array([], dtype=np.float32)
        self._frame_idx = 0
        self._noise_floor = _NoiseFloor()
        self._sm = _StateMachine(**vad._sm_kwargs)
        self._samples_in = 0  # total samples processed, for time-stamping

    def push(self, chunk: np.ndarray) -> tuple[VADState, list[SpeechSegment]]:
        """
        Push one audio chunk.

        Returns
        -------
        (state, completed_segments)
            state             : current VADState
            completed_segments: any segments that completed in this chunk
        """
        vad = self._vad
        self._buffer = np.concatenate([self._buffer, chunk.astype(np.float32)])
        self._chunk_count += 1
        completed: list[SpeechSegment] = []

        # Process all complete frames in the buffer
        while len(self._buffer) >= vad.frame_len:
            frame = self._buffer[: vad.frame_len]
            self._buffer = self._buffer[vad.hop_len :]

            e = float(_rms(frame[np.newaxis])[0])
            z = float(_zcr(frame[np.newaxis], vad.sr)[0])
            f = float(_spectral_flatness(frame[np.newaxis])[0])
            nf = self._noise_floor.update(e)

            # Skip thresholding during warmup — just update noise floor
            if self._chunk_count <= self._warmup:
                self._frame_idx += 1
                continue

            threshold = nf * vad.energy_snr
            active = e > threshold and z < vad.zcr_max and f < vad.flatness_max
            self._sm.step(self._frame_idx, active)
            self._frame_idx += 1

            for s, e_idx in self._sm.pop_segments():
                completed.append(vad._frames_to_seg(s, e_idx))

        return self._sm.current_state, completed

    def flush(self) -> list[SpeechSegment]:
        """Signal end of stream; returns any open segment."""
        self._sm.flush(self._frame_idx)
        segs = self._sm.pop_segments()
        return [self._vad._frames_to_seg(s, e) for s, e in segs]

    @property
    def state(self) -> VADState:
        return self._sm.current_state


# Convenience functions


def detect_speech(audio: np.ndarray, sr: int = 16_000, **kwargs) -> list[SpeechSegment]:
    """One-shot helper: detect speech segments in `audio` at sample rate `sr`."""
    return VoiceActivityDetector(sr=sr, **kwargs).detect(audio)


def trim_silence(
    audio: np.ndarray,
    sr: int = 16_000,
    pad_ms: int = 50,
    **kwargs,
) -> np.ndarray:
    """
    Return `audio` trimmed to the span of detected speech, with `pad_ms`
    of silence preserved on each side.
    """
    segs = detect_speech(audio, sr, **kwargs)
    if not segs:
        return audio
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(segs[0].start * sr) - pad)
    end = min(len(audio), int(segs[-1].end * sr) + pad)
    return audio[start:end]
