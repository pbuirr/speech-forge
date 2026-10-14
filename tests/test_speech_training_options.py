"""Opt-in speech training options: bucketing, speed perturbation, SpecAugment, seeds.

Defaults must reproduce earlier behaviour; new options must preserve exposure,
lengths and determinism. These are behaviour checks, not accuracy claims.
"""

import pickle
import random
from collections import Counter

import numpy as np
import soundfile as sf
import torch

from speech_engine.augment import spec_augment
from speech_engine.dataset import ASRDataset, BucketBatchSampler, build_dataloader
from speech_engine.tokenizer import CharTokenizer


def _clips(tmp_path, seconds=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5)):
    rows = []
    rng = np.random.default_rng(0)
    for i, duration in enumerate(seconds):
        path = tmp_path / f"clip{i}.wav"
        t = np.arange(int(16000 * duration)) / 16000
        sf.write(
            path,
            (
                0.3 * np.sin(2 * np.pi * (200 + 40 * i) * t) + 0.01 * rng.standard_normal(len(t))
            ).astype(np.float32),
            16000,
        )
        rows.append((str(path), "a cab"))
    return rows


def test_bucket_sampler_preserves_budget_and_groups_lengths():
    lengths = [float(i % 17) + 1 for i in range(400)]
    sampler = BucketBatchSampler(
        lengths, batch_size=8, samples_per_epoch=1000, seed=5, window_batches=10
    )
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 1000
    assert len(batches) == len(sampler)
    spread = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    assert spread < 4  # random batches of 8 from 1..17 would average far more
    again = list(
        BucketBatchSampler(lengths, batch_size=8, samples_per_epoch=1000, seed=5, window_batches=10)
    )
    assert again == batches


def test_bucket_sampler_resume_reproduces_next_epoch():
    lengths = list(range(1, 50))
    sampler = BucketBatchSampler(lengths, 4, 60, seed=11)
    list(sampler)
    state = sampler.generator.get_state()
    expected = list(sampler)
    resumed = BucketBatchSampler(lengths, 4, 60, seed=11)
    resumed.generator.set_state(state)
    assert list(resumed) == expected


def test_weighted_bucket_sampler_follows_weights():
    weights = [1.0] * 10 + [9.0] * 10
    sampler = BucketBatchSampler(list(range(20)), 5, 20000, weights=weights, seed=3)
    counts = Counter(i for batch in sampler for i in batch)
    heavy = sum(counts[i] for i in range(10, 20))
    assert 0.85 < heavy / 20000 < 0.95


def test_speed_perturbation_changes_length_and_caches_separately(tmp_path):
    rows = _clips(tmp_path, (2.0,))
    tok = CharTokenizer().fit([text for _, text in rows])
    dataset = ASRDataset(
        rows, tok, max_frames=400, cache_dir=tmp_path / "cache", speed_factors=(1.0, 0.9, 1.1)
    )
    assert len(dataset) == 3
    frames = [dataset[i][0].shape[1] for i in range(3)]
    assert abs(frames[1] / frames[0] - 1 / 0.9) < 0.02
    assert abs(frames[2] / frames[0] - 1 / 1.1) < 0.02
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 3
    cached = [dataset[i][0] for i in range(3)]
    assert all(c.dtype == torch.float32 for c in cached)
    assert torch.equal(cached[0], dataset[0][0])


def test_default_dataset_keeps_original_cache_key_and_values(tmp_path):
    rows = _clips(tmp_path, (1.5,))
    tok = CharTokenizer().fit([text for _, text in rows])
    plain = ASRDataset(rows, tok, max_frames=400, cache_dir=tmp_path / "a")
    explicit = ASRDataset(rows, tok, max_frames=400, cache_dir=tmp_path / "b", speed_factors=(1.0,))
    assert len(plain) == 1
    assert torch.equal(plain[0][0], explicit[0][0])
    assert [p.name for p in (tmp_path / "a").glob("*.npy")] == [
        p.name for p in (tmp_path / "b").glob("*.npy")
    ]


def test_bucketed_loader_with_speed_and_workers_is_picklable(tmp_path):
    rows = _clips(tmp_path)
    tok = CharTokenizer().fit([text for _, text in rows])
    loader = build_dataloader(
        rows,
        tok,
        batch_size=2,
        samples_per_epoch=12,
        seed=7,
        lengths=[sf.info(p).duration for p, _ in rows],
        speed_factors=(1.0, 0.9, 1.1),
        max_frames=500,
        cache_dir=tmp_path / "cache",
    )
    pickle.dumps(loader.collate_fn)
    batches = list(loader)
    assert sum(b["mel"].shape[0] for b in batches) == 12
    assert all(int(b["mel_lengths"].max()) == b["mel"].shape[2] for b in batches)


def test_default_loader_unchanged_sampling(tmp_path):
    rows = _clips(tmp_path)
    tok = CharTokenizer().fit([text for _, text in rows])
    loader = build_dataloader(rows, tok, batch_size=2, samples_per_epoch=10, max_frames=500)
    generator = torch.Generator().manual_seed(839)
    expected = list(
        torch.utils.data.RandomSampler(
            range(len(rows)), replacement=True, num_samples=10, generator=generator
        )
    )
    assert list(loader.sampler) == expected


def test_spec_augment_masks_only_real_frames_and_is_deterministic():
    mel = torch.randn(2, 80, 300)
    lengths = torch.tensor([300, 200])
    a = spec_augment(mel, lengths, random.Random(1))
    b = spec_augment(mel, lengths, random.Random(1))
    assert torch.equal(a, b)
    assert torch.equal(a[1, :, 200:], mel[1, :, 200:])  # padding untouched
    assert not torch.equal(a, mel)
    assert torch.equal(
        spec_augment(mel, lengths, random.Random(1), frequency_masks=0, time_masks=0), mel
    )


def test_resume_is_exact_with_bucketing_speed_and_spec_augment(tmp_path, monkeypatch):
    import shutil
    from speech_engine import train as training
    from speech_engine.model_ctc import CTCModel
    from speech_engine.train import TrainConfig

    monkeypatch.setattr(
        ASRDataset,
        "_load_or_compute_spec",
        lambda self, path, speed=1.0: np.full(
            (80, int(50 / speed)), int(path[-1]) / 10, dtype=np.float32
        ),
    )
    examples = [(f"row{i}", "hi") for i in range(4)]
    tokenizer = CharTokenizer()

    def loader():
        return build_dataloader(
            examples,
            tokenizer,
            batch_size=2,
            samples_per_epoch=6,
            seed=41,
            lengths=[1.0, 2.0, 3.0, 4.0],
            speed_factors=(1.0, 0.9, 1.1),
        )

    def model():
        return CTCModel(
            tokenizer.vocab_size,
            cnn_channels=(2, 4, 6),
            hidden_size=8,
            num_lstm_layers=2,
            dropout=0.2,
        )

    def config(directory):
        cfg = TrainConfig(
            cnn_channels=(2, 4, 6),
            hidden_size=8,
            num_lstm_layers=2,
            max_epochs=2,
            batch_size=2,
            warmup_steps=0,
            log_every=100,
            save_every=1,
            checkpoint_dir=str(directory),
        )
        cfg.spec_augment = True
        return cfg

    restart = tmp_path / "after-first-epoch.pt"
    original_save = training.save_checkpoint

    def save(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if path.name == "latest.pt" and args[2] == 0:
            shutil.copyfile(path, restart)

    monkeypatch.setattr(training, "save_checkpoint", save)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        uninterrupted = model()
        training.train(
            uninterrupted, loader(), None, config(tmp_path / "full"), torch.device("cpu")
        )
        expected = {key: value.clone() for key, value in uninterrupted.state_dict().items()}
        torch.manual_seed(999)
        resumed = model()
        training.train(
            resumed,
            loader(),
            None,
            config(tmp_path / "resumed"),
            torch.device("cpu"),
            resume_from=restart,
        )
        assert all(torch.equal(value, expected[key]) for key, value in resumed.state_dict().items())


def test_amp_requires_cuda():
    import pytest
    from speech_engine import train as training
    from speech_engine.model_ctc import CTCModel
    from speech_engine.train import TrainConfig

    cfg = TrainConfig(max_epochs=1)
    cfg.amp = True
    with pytest.raises(ValueError, match="CUDA"):
        training.train(
            CTCModel(10, cnn_channels=(2, 4, 6), hidden_size=8, num_lstm_layers=1),
            [],
            None,
            cfg,
            torch.device("cpu"),
        )
