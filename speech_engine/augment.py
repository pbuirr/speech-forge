"""SpecAugment-style frequency and time masking for training batches.

Masks only real (unpadded) frames and leaves transcripts unchanged.
"""

import random


def mask_features(mel, lengths, rng: random.Random, frequency_width=8, time_width=20):
    """One frequency mask and one time mask per utterance (time capped at 10% of real frames).

    Masked values are filled with the utterance mean. Uses its own RNG, so model
    and dropout randomness are unaffected, and never touches padding or the cache.
    """
    if mel.ndim != 3 or len(lengths) != len(mel):
        raise ValueError("Expected batch, frequency, time and matching lengths")
    if frequency_width < 0 or time_width < 0:
        raise ValueError("Mask widths must be nonnegative")
    output = mel.clone()
    for i, length in enumerate(lengths.tolist()):
        if not 0 < length <= mel.shape[2]:
            raise ValueError("Invalid real feature length")
        fill = mel[i, :, :length].mean()
        width = rng.randint(0, min(frequency_width, mel.shape[1] - 1))
        start = rng.randint(0, mel.shape[1] - width)
        output[i, start : start + width, :length] = fill
        width = rng.randint(0, min(time_width, length // 10))
        start = rng.randint(0, length - width)
        output[i, :, start : start + width] = fill
    return output


def spec_augment(
    mel,
    lengths,
    rng: random.Random,
    frequency_masks=2,
    frequency_width=27,
    time_masks=2,
    time_ratio=0.05,
):
    """SpecAugment-style policy: several frequency and time masks per utterance.

    Defaults follow the scale of Park et al. (2019) LibriSpeech policies (two masks
    of each kind, frequency width up to 27 of 80 bins) with time masks capped at
    ``time_ratio`` of each utterance's real length. No time warping. Fills with the
    utterance mean, like ``mask_features``; padding and cached data are unchanged.
    Benefit must be measured at matched budget; it is not assumed.
    """
    if mel.ndim != 3 or len(lengths) != len(mel):
        raise ValueError("Expected batch, frequency, time and matching lengths")
    if min(frequency_masks, frequency_width, time_masks) < 0 or not 0 <= time_ratio <= 0.5:
        raise ValueError("Mask counts/widths must be nonnegative and time_ratio within [0, 0.5]")
    output = mel.clone()
    bins = mel.shape[1]
    for i, length in enumerate(lengths.tolist()):
        if not 0 < length <= mel.shape[2]:
            raise ValueError("Invalid real feature length")
        fill = mel[i, :, :length].mean()
        for _ in range(frequency_masks):
            width = rng.randint(0, min(frequency_width, bins - 1))
            start = rng.randint(0, bins - width)
            output[i, start : start + width, :length] = fill
        limit = int(length * time_ratio)
        for _ in range(time_masks):
            width = rng.randint(0, limit)
            start = rng.randint(0, length - width)
            output[i, :, start : start + width] = fill
    return output
