import json
import pytest
from pathlib import Path
from speech_engine.corpus import load_metadata, validate_partitions, lexical_normalization_loss
from speech_engine.tokenizer import CharTokenizer


def test_metadata_blocks_identity_and_prompt_leakage(tmp_path):
    path = tmp_path / "metadata.jsonl"
    records = [
        {
            "file": str(tmp_path / "a.wav"),
            "text": "hello",
            "speaker": "source:one",
            "corpus": "source",
            "language": "tr",
            "license": "CC BY 4.0",
            "source": "https://example.test",
            "prompt_id": "source:p1",
        },
        {
            "file": str(tmp_path / "b.wav"),
            "text": "world",
            "speaker": "source:two",
            "corpus": "source",
            "language": "en",
            "license": "CC BY 4.0",
            "source": "https://example.test",
            "prompt_id": "source:p2",
        },
    ]

    def metadata():
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return load_metadata(path)

    partitions = {"train": [(records[0]["file"], "hello")], "test": [(records[1]["file"], "world")]}
    validate_partitions(partitions, metadata())
    records[1]["prompt_id"] = "source:p1"
    with pytest.raises(ValueError, match="overlap"):
        validate_partitions(partitions, metadata())
    records[1]["prompt_id"] = "source:p2"
    records[1]["speaker"] = "source:one"
    with pytest.raises(ValueError, match="overlap"):
        validate_partitions(partitions, metadata())
    records[1]["speaker"] = "source:two"
    records[1]["text"] = "changed"
    with pytest.raises(ValueError, match="transcript"):
        validate_partitions(partitions, metadata())


def test_canonical_and_aliased_audio_cannot_bypass_overlap_checks(tmp_path):
    canonical = str((tmp_path / "a.wav").resolve())
    alias = str(tmp_path / "sub" / ".." / "a.wav")
    record = {"file": canonical, "text": "hi", "speaker": "source:one", "prompt_id": "source:p1"}
    from speech_engine.corpus import record_for

    assert record_for(Path(alias), {canonical: record}) is record
    with pytest.raises(ValueError, match="overlap"):
        validate_partitions(
            {"train": [(canonical, "hi")], "test": [(alias, "hi")]}, {canonical: record}
        )


def test_lossy_supervision_is_distinct_from_orthographic_punctuation():
    tok = CharTokenizer()
    assert not lexical_normalization_loss("Hello, don't leave!", tok)
    assert lexical_normalization_loss("I paid 20", tok)
    assert lexical_normalization_loss("Beyoncé sings", tok)
    unicode_tok = CharTokenizer(normalization="unicode").fit(["東京 20", "türkçe"])
    assert not lexical_normalization_loss("東京 20!", unicode_tok)
    assert not lexical_normalization_loss("türkçe", unicode_tok)
    assert lexical_normalization_loss("العربية", unicode_tok)


def test_all_dataset_loaders_reject_suppressed_numbers(monkeypatch):
    import numpy as np
    from speech_engine.dataset import ASRDataset

    dataset = ASRDataset([("unused.wav", "I paid 20")], CharTokenizer())
    monkeypatch.setattr(
        dataset, "_load_or_compute_spec", lambda _: np.zeros((80, 100), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="drops letters or numbers"):
        dataset[0]


def test_fixed_training_budget_sampler_is_independent_and_reproducible():
    import torch
    from speech_engine.dataset import build_dataloader

    examples = [("unused" + str(i), "hi") for i in range(3)]
    loader = build_dataloader(examples, CharTokenizer(), batch_size=2, samples_per_epoch=8)
    initial = torch.random.get_rng_state().clone()
    indices = list(loader.sampler)
    assert torch.equal(torch.random.get_rng_state(), initial)
    assert len(indices) == 8 and len(loader) == 4
    other = build_dataloader(examples, CharTokenizer(), batch_size=2, samples_per_epoch=8)
    assert indices == list(other.sampler)


def test_balancing_respects_small_languages_dialects_and_speakers(tmp_path):
    from speech_engine.corpus import balanced_weights

    records = {}
    examples = []
    for corpus, language, dialect, speaker, count in [
        ("large", "en", "north", "a", 8),
        ("large", "tr", "east", "b", 1),
        ("small", "ja", None, "c", 2),
    ]:
        for i in range(count):
            path = str((tmp_path / (speaker + str(i) + ".wav")).resolve())
            records[path] = {
                "corpus": corpus,
                "language": language,
                "dialect": dialect,
                "speaker": speaker,
            }
            examples.append((path, "unused"))
    weights = balanced_weights(examples, records)
    assert sum(weights) == pytest.approx(1)
    assert sum(weights[:8]) == pytest.approx(0.25)
    assert weights[8] == pytest.approx(0.25)
    assert sum(weights[9:]) == pytest.approx(0.5)
