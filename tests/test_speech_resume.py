import shutil

import numpy as np
import torch

from speech_engine.dataset import ASRDataset, build_dataloader
from speech_engine.model_ctc import CTCModel
from speech_engine.tokenizer import CharTokenizer
from speech_engine.train import TrainConfig


def test_resumed_training_preserves_sampling_and_dropout(tmp_path, monkeypatch):
    from speech_engine import train as training

    monkeypatch.setattr(
        ASRDataset,
        "_load_or_compute_spec",
        lambda self, path: np.full((80, 50), int(path[-1]) / 10, dtype=np.float32),
    )
    examples = [(f"row{i}", "hi") for i in range(4)]
    tokenizer = CharTokenizer()

    def loader():
        return build_dataloader(examples, tokenizer, batch_size=2, samples_per_epoch=6)

    def model():
        return CTCModel(
            tokenizer.vocab_size,
            cnn_channels=(2, 4, 6),
            hidden_size=8,
            num_lstm_layers=2,
            dropout=0.2,
        )

    def config(directory):
        return TrainConfig(
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
        torch.manual_seed(999)  # A fresh process does not inherit the original RNG.
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
