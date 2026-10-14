"""Speech datasets, batching and corpus loaders.

``ASRDataset`` turns (audio path, transcript) pairs into (log-Mel, token IDs),
with an on-disk feature cache and optional speed perturbation. ``collate_fn``
pads a batch and records the real lengths that CTC loss needs.
``BucketBatchSampler`` groups utterances of similar length.

Loaders: ``load_librispeech`` (LibriSpeech layout), ``load_manifest``
(audio_path<TAB>transcript) and ``load_common_voice``.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from speech_engine.audio import normalize_amplitude, preprocess, resample
from speech_engine.features import (
    MAX_FRAMES,
    SAMPLE_RATE,
    log_mel_spectrogram,
)
from speech_engine.tokenizer import PAD_ID, CharTokenizer, normalize_text


# Core dataset


class ASRDataset(Dataset):
    """(audio path, transcript) pairs as (log-Mel, token IDs) examples.

    Args:
        examples:      list of (audio_path, transcript) pairs
        tokenizer:     CharTokenizer used to encode transcripts
        max_frames:    clips longer than this are rejected
        sample_rate:   audio is resampled to this rate
        cache_dir:     if set, spectrograms are cached here as .npy files
        speed_factors: speed perturbation factors; the dataset length is
                       len(examples) × len(speed_factors)

    ``__getitem__`` returns (mel (n_mels, frames) float32, tokens (L,) int64),
    with no BOS/EOS in the tokens.
    """

    def __init__(
        self,
        examples: list[tuple[str, str]],
        tokenizer: CharTokenizer,
        max_frames: int = MAX_FRAMES,
        sample_rate: int = SAMPLE_RATE,
        cache_dir: Optional[Path] = None,
        speed_factors: tuple[float, ...] = (1.0,),
    ) -> None:
        if not examples:
            raise ValueError("examples list is empty — nothing to load")
        if (
            not speed_factors
            or any(not 0.5 <= f <= 2.0 for f in speed_factors)
            or len(set(speed_factors)) != len(speed_factors)
        ):
            raise ValueError("Speed factors must be distinct values within [0.5, 2.0]")
        # Index space is (speed slot, recording): index = slot * len(examples) + recording.
        # Deterministic per index, so samplers, workers and caches agree on what was drawn.
        self.speed_factors = tuple(float(f) for f in speed_factors)
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.examples) * len(self.speed_factors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        slot, recording = divmod(idx, len(self.examples))
        audio_path, transcript = self.examples[recording]

        # Spectrogram
        speed = self.speed_factors[slot]
        # Keep the one-argument call for speed 1.0 (existing overrides/tests rely on it).
        spec = (
            self._load_or_compute_spec(audio_path)
            if speed == 1.0
            else self._load_or_compute_spec(audio_path, speed)
        )
        # Preserve actual lengths; collate_fn pads only to this batch's longest clip.
        # Padding here made every recording appear to be max_frames long to CTC.
        if spec.shape[1] > self.max_frames:
            raise ValueError(
                "Audio exceeds max_frames; filter or segment audio and transcript together"
            )
        mel = torch.from_numpy(spec)  # (n_mels, actual_frames), float32

        # Transcript
        # encode() applies normalize_text() internally.
        # CTC targets have no BOS/EOS.
        token_ids = self.tokenizer.encode(transcript)
        if self.tokenizer.decode(token_ids) != normalize_text(
            transcript, self.tokenizer.normalization
        ):
            raise ValueError(
                "Transcript contains characters absent from the trained tokenizer; do not silently alter supervision or evaluation references"
            )
        from speech_engine.corpus import lexical_normalization_loss

        if lexical_normalization_loss(transcript, self.tokenizer):
            raise ValueError(
                "Transcript normalization drops letters or numbers; supply evidenced spoken-form supervision or an appropriate tokenizer"
            )
        if not token_ids:
            # A padding token is not a valid supervised transcript.
            raise ValueError("Transcript is empty after configured normalization")
        tokens = torch.tensor(token_ids, dtype=torch.long)  # (L,)

        return mel, tokens

    def _load_or_compute_spec(self, audio_path: str, speed: float = 1.0) -> np.ndarray:
        """Load a cached spectrogram or compute and cache it on first access.

        Speed 1.0 keeps the original cache key and float32 format. Perturbed
        copies use a separate key and are stored as float16 to halve disk use.
        """
        if self.cache_dir is not None:
            # Use the audio file's stem as the cache key.
            # For datasets where filenames collide across splits (rare), add
            # a hash of the full path to the key.
            source = Path(audio_path).resolve()
            stat = source.stat()
            identity = f"{source}|{stat.st_size}|{stat.st_mtime_ns}|{self.sample_rate}|logmel-v1"
            if speed != 1.0:
                identity += f"|speed{speed:g}|float16"
            cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                return np.load(str(cache_path)).astype(np.float32, copy=False)
            spec = (
                self._compute_spec(audio_path)
                if speed == 1.0
                else self._compute_spec(audio_path, speed)
            )
            np.save(str(cache_path), spec if speed == 1.0 else spec.astype(np.float16))
            return spec
        return (
            self._compute_spec(audio_path)
            if speed == 1.0
            else self._compute_spec(audio_path, speed)
        )

    def _compute_spec(self, audio_path: str, speed: float = 1.0) -> np.ndarray:
        waveform, _ = preprocess(audio_path, target_sr=self.sample_rate)
        if speed != 1.0:
            # Speed perturbation (tempo and pitch together): treat the signal as if
            # recorded at speed * rate, then resample back to the model rate.
            waveform = normalize_amplitude(
                resample(waveform, round(self.sample_rate * speed), self.sample_rate)
            )
        return log_mel_spectrogram(waveform, sample_rate=self.sample_rate)

    def __repr__(self) -> str:
        return (
            f"ASRDataset(n={len(self)}, max_frames={self.max_frames}, "
            f"vocab_size={self.tokenizer.vocab_size})"
        )


# Collate function


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    """Pad a list of (mel, tokens) examples into one batch.

    Returns a dict:
        "mel"           (B, n_mels, T_max) float32
        "mel_lengths"   (B,) int64, real frames
        "tokens"        (B, L_max) int64, padded with pad_token_id
        "token_lengths" (B,) int64, real tokens

    Encoder output lengths for CTC are computed from ``mel_lengths`` by the model.
    """
    mels, token_seqs = zip(*batch)

    # Pad mel spectrograms
    mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
    T_max = int(mel_lengths.max().item())
    n_mels = mels[0].shape[0]

    mel_batch = torch.zeros(len(mels), n_mels, T_max, dtype=torch.float32)
    for i, mel in enumerate(mels):
        mel_batch[i, :, : mel.shape[1]] = mel

    # Pad token sequences
    token_lengths = torch.tensor([len(t) for t in token_seqs], dtype=torch.long)
    L_max = int(token_lengths.max().item())

    token_batch = torch.full(
        (len(token_seqs), L_max),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    for i, tok in enumerate(token_seqs):
        token_batch[i, : len(tok)] = tok

    return {
        "mel": mel_batch,  # (B, 80, T_max)
        "mel_lengths": mel_lengths,  # (B,)
        "tokens": token_batch,  # (B, L_max)
        "token_lengths": token_lengths,  # (B,)
    }


# Length-bucketed sampling


class BucketBatchSampler(torch.utils.data.Sampler):
    """Draw a fixed sampling budget with replacement, then batch similar lengths.

    Each epoch draws ``samples_per_epoch`` indices (uniform or weighted), sorts
    each window of ``batch_size * window_batches`` draws by length, cuts it into
    batches and shuffles batch order. Exposure per recording matches the plain
    replacement sampler; only which draws share a batch changes, which reduces
    padding. The generator state is saved for exact resume.
    """

    def __init__(
        self, lengths, batch_size, samples_per_epoch, weights=None, seed=839, window_batches=50
    ):
        if batch_size < 1 or samples_per_epoch < 1 or window_batches < 1:
            raise ValueError("Batch size, sample budget and window must be positive")
        self.lengths = torch.as_tensor(lengths, dtype=torch.float64)
        if self.lengths.ndim != 1 or len(self.lengths) == 0:
            raise ValueError("Lengths must be a non-empty vector")
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        if self.weights is not None and (
            len(self.weights) != len(self.lengths) or not torch.all(self.weights > 0)
        ):
            raise ValueError("Positive sampling weights must match lengths")
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.window = batch_size * window_batches
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        full, rest = divmod(self.samples_per_epoch, self.window)
        return full * math.ceil(self.window / self.batch_size) + math.ceil(rest / self.batch_size)

    def __iter__(self):
        if self.weights is None:
            draws = torch.randint(
                len(self.lengths), (self.samples_per_epoch,), generator=self.generator
            )
        else:
            draws = torch.multinomial(
                self.weights, self.samples_per_epoch, replacement=True, generator=self.generator
            )
        batches = []
        for start in range(0, len(draws), self.window):
            window = draws[start : start + self.window]
            window = window[torch.argsort(self.lengths[window], stable=True)]
            batches.extend(
                window[i : i + self.batch_size].tolist()
                for i in range(0, len(window), self.batch_size)
            )
        for position in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[position]


# DataLoader factory


def build_dataloader(
    examples: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_frames: int = MAX_FRAMES,
    cache_dir: Optional[Path] = None,
    samples_per_epoch: int | None = None,
    sampling_weights: list[float] | None = None,
    seed: int = 839,
    lengths: list[float] | None = None,
    speed_factors: tuple[float, ...] = (1.0,),
) -> DataLoader:
    """Build the DataLoader used for training and evaluation.

    Args:
        examples:        list of (audio_path, transcript) pairs
        tokenizer:       fitted CharTokenizer
        batch_size:      examples per batch
        shuffle:         shuffle when no sample budget is given
        num_workers:     DataLoader worker processes
        max_frames:      spectrogram frame cap
        cache_dir:       spectrogram cache directory
        samples_per_epoch: draws per epoch with replacement (the "sampled cycle")
        seed:            sampler seed (839 reproduces earlier runs)
        lengths:         per-recording lengths; with a sample budget, enables
                         length-bucketed batches
        speed_factors:   training-only speed perturbation; each draw picks one
                         factor uniformly (requires a sample budget)
    """
    if tuple(speed_factors) != (1.0,) and samples_per_epoch is None:
        raise ValueError("Speed perturbation requires an explicit sample budget")
    dataset = ASRDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_frames=max_frames,
        cache_dir=cache_dir,
        speed_factors=tuple(speed_factors),
    )
    copies = len(dataset.speed_factors)
    if sampling_weights is not None:
        sampling_weights = list(sampling_weights) * copies
    collate = functools.partial(
        collate_fn, pad_token_id=tokenizer.pad_id
    )  # picklable for worker processes
    if lengths is not None:
        if samples_per_epoch is None:
            raise ValueError("Length bucketing requires an explicit sample budget")
        if len(lengths) != len(examples):
            raise ValueError("Lengths must match examples")
        expanded = [length / factor for factor in dataset.speed_factors for length in lengths]
        batch_sampler = BucketBatchSampler(
            expanded, batch_size, samples_per_epoch, sampling_weights, seed=seed
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=False,
        )
    sampler = None
    if samples_per_epoch is not None:
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        generator = torch.Generator().manual_seed(seed)
        if sampling_weights is not None:
            if len(sampling_weights) != len(dataset) or any(
                not np.isfinite(w) or w <= 0 for w in sampling_weights
            ):
                raise ValueError("Positive finite sampling weights must match the dataset")
            sampler = torch.utils.data.WeightedRandomSampler(
                sampling_weights, samples_per_epoch, replacement=True, generator=generator
            )
        else:
            sampler = torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=samples_per_epoch, generator=generator
            )
    elif sampling_weights is not None:
        raise ValueError("Weighted sampling requires an explicit sample budget")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,  # set True when training on CUDA
        drop_last=False,
    )


# Format loaders
# Each returns list[tuple[str, str]] = [(audio_path, transcript), ...]
# The uniform type makes it trivial to combine datasets:
#   examples = load_librispeech(root, "train-clean-100") + load_manifest("custom.tsv")


def load_librispeech(root: str | Path, split: str = "test-clean") -> list[tuple[str, str]]:
    """Load (audio_path, transcript) pairs from a LibriSpeech directory.

    LibriSpeech directory layout::

        {root}/{split}/{speaker_id}/{chapter_id}/
            {speaker_id}-{chapter_id}-{utt_id}.flac   — audio
            {speaker_id}-{chapter_id}.trans.txt        — transcripts

    Transcript file format (one utterance per line)::

        1089-134686-0000 MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES
        1089-134686-0001 AND WE ARE GLAD TO WELCOME HIS GOSPEL

    Args:
        root:  path to the folder that contains the split folder
               e.g. "data/LibriSpeech" if split is at "data/LibriSpeech/test-clean/"
        split: folder name of the split ("train-clean-100", "test-clean", etc.)

    Returns:
        list of (absolute_path_to_flac, lowercase_transcript)

    Download::

        python scripts/speech/acquire_librispeech.py --archive test-clean
        # or via torchaudio:
        import torchaudio
        torchaudio.datasets.LIBRISPEECH("data/", url="test-clean", download=True)
    """
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"LibriSpeech split not found: {split_dir}\n"
            f"  Download: python scripts/speech/acquire_librispeech.py --archive {split}\n"
            f"  Or:       import torchaudio; "
            f"torchaudio.datasets.LIBRISPEECH('data/', url='{split}', download=True)"
        )

    examples: list[tuple[str, str]] = []
    for trans_file in sorted(split_dir.rglob("*.trans.txt")):
        chapter_dir = trans_file.parent
        for line in trans_file.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            # LibriSpeech stores audio as .flac; older copies may use .wav
            for ext in (".flac", ".wav"):
                audio_path = chapter_dir / f"{utt_id}{ext}"
                if audio_path.exists():
                    examples.append((str(audio_path), transcript.lower()))
                    break

    if not examples:
        raise RuntimeError(
            f"No examples found in {split_dir}. "
            "Check that the split was downloaded and extracted correctly."
        )
    return examples


def load_manifest(path: str | Path) -> list[tuple[str, str]]:
    """Load (audio_path, transcript) pairs from a TSV manifest.

    One ``path<TAB>transcript`` per line; lines starting with # are skipped.
    Relative paths are resolved against the manifest's directory.

    Args:
        path: manifest file

    Returns:
        list of (resolved_audio_path, transcript) pairs
    """
    path = Path(path).resolve()
    manifest_dir = path.parent
    examples: list[tuple[str, str]] = []

    with open(path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected tab-separated "
                    f"'audio_path<TAB>transcript', got:\n  {line!r}"
                )
            audio_str, transcript = parts
            audio_path = Path(audio_str)
            if not audio_path.is_absolute():
                audio_path = manifest_dir / audio_path
            examples.append((str(audio_path), transcript))

    return examples


def load_common_voice(
    root: str | Path,
    split: str = "test",
) -> list[tuple[str, str]]:
    """Load (clip path, sentence) pairs from an extracted Common Voice language folder.

    Expects ``{root}/{split}.tsv`` and ``{root}/clips/``. Clips missing on disk
    are skipped.

    Args:
        root:  language folder, e.g. "data/common_voice/es"
        split: "train", "dev" or "test"

    Returns:
        list of (path_to_mp3, transcript) pairs
    """
    root = Path(root)
    tsv_path = root / f"{split}.tsv"
    clips_dir = root / "clips"

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Common Voice TSV not found: {tsv_path}\n"
            f"  Download at https://commonvoice.mozilla.org/en/datasets"
        )

    examples: list[tuple[str, str]] = []
    missing = 0
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            filename = row.get("path", "").strip()
            transcript = row.get("sentence", "").strip()
            if not filename or not transcript:
                continue
            audio_path = clips_dir / filename
            if audio_path.exists():
                examples.append((str(audio_path), transcript))
            else:
                missing += 1

    if missing:
        print(f"[dataset] Warning: {missing} clips listed in TSV but not found in {clips_dir}")
    return examples
