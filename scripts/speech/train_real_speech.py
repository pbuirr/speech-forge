"""Train the CTC model on real speech manifests, with resumable checkpoints and a JSON report."""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import soundfile as sf
import torch
from speech_engine.dataset import load_manifest, build_dataloader
from speech_engine.tokenizer import CharTokenizer
from speech_engine.model_ctc import CTCModel
from speech_engine.train import TrainConfig, train, load_checkpoint
from speech_engine.decode import greedy_decode
from speech_engine.evaluate import evaluate_dataset
from scripts.speech.prepare_real_speech import validate_disjoint
from speech_engine.corpus import (
    load_metadata,
    validate_partitions,
    record_for,
    metadata_digest,
    balanced_weights,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/speech_real")
    p.add_argument("--output", type=Path, default=ROOT / "data/training/asr_real")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument(
        "--architecture",
        choices=("compact", "wide", "full"),
        default="compact",
        help="Original CNN/BiLSTM sizes: compact128x2, wide256x2 with compact CNN, full256x3 with 32/64/128 CNN",
    )
    p.add_argument(
        "--time-reduction",
        type=int,
        choices=(2, 4),
        default=2,
        help="Checkpoint-bound encoder stride; legacy models use two",
    )
    p.add_argument("--peak-lr", type=float, default=1e-3)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Explicit local training hardware; auto prefers available CUDA",
    )
    p.add_argument(
        "--feature-cache", type=Path, help="Reuse deterministic audio features across experiments"
    )
    p.add_argument(
        "--no-train-feature-cache",
        action="store_true",
        help="Compute training features on the fly (saves disk; use --num-workers); dev/test stay cached",
    )
    p.add_argument(
        "--skip-test",
        action="store_true",
        help="Select candidates on development before opening test results",
    )
    p.add_argument(
        "--masked-batch-norm",
        action="store_true",
        help="Exclude padded time cells from CNN training moments",
    )
    p.add_argument(
        "--feature-masking",
        action="store_true",
        help="Training-only bounded time/frequency masking",
    )
    p.add_argument("--train-limit", type=int, default=600)
    p.add_argument("--eval-limit", type=int, default=80)
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        help="Matched optimizer budget, uniform replacement sampling",
    )
    p.add_argument(
        "--metadata", type=Path, help="Explicit multilingual corpus/speaker/prompt identities"
    )
    p.add_argument(
        "--train-corpus", help="Optional training corpus restriction; evaluation is unchanged"
    )
    p.add_argument(
        "--balance-corpora",
        action="store_true",
        help="Equal corpus/language/dialect/speaker sampling mass",
    )
    starts = p.add_mutually_exclusive_group()
    starts.add_argument("--resume", type=Path)
    starts.add_argument(
        "--warm-start",
        type=Path,
        help="Restore acoustic weights with a new optimizer/schedule for expanded data",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=839,
        help="Model initialisation, dropout, sampler and augmentation seed; data selection stays fixed",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size; draws per cycle stay --samples-per-epoch",
    )
    p.add_argument(
        "--bucket-batches",
        action="store_true",
        help="Batch similar-length draws together (same exposure, less padding)",
    )
    p.add_argument(
        "--amp", action="store_true", help="CUDA mixed precision; CTC loss stays float32"
    )
    p.add_argument(
        "--num-workers", type=int, default=0, help="Data-loading worker processes for training"
    )
    p.add_argument(
        "--spec-augment",
        action="store_true",
        help="Two frequency masks (<=27 bins) and two time masks (<=5%% of length)",
    )
    p.add_argument(
        "--speed-perturb",
        action="store_true",
        help="Training-only speed perturbation 0.9/1.0/1.1 (cached as float16)",
    )
    p.add_argument("--train-max-duration", type=float, default=10)
    p.add_argument("--eval-max-duration", type=float, default=10)
    args = p.parse_args()
    dimensions = {
        "compact": ((16, 32, 64), 128, 2),
        "wide": ((16, 32, 64), 256, 2),
        "full": ((32, 64, 128), 256, 3),
    }[args.architecture]
    if (
        args.train_limit < 1
        or args.eval_limit < 1
        or (args.samples_per_epoch is not None and args.samples_per_epoch < 1)
    ):
        raise ValueError("Sample limits must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative")
    if args.feature_masking and args.spec_augment:
        raise ValueError("Choose either --feature-masking or --spec-augment")
    if (args.bucket_batches or args.speed_perturb) and not args.samples_per_epoch:
        raise ValueError("Bucketing and speed perturbation require --samples-per-epoch")
    if args.epochs < 1 or not 0 < args.peak_lr <= 0.01:
        raise ValueError("Require positive epochs and a learning rate within (0, .01]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    speed_factors = (1.0, 0.9, 1.1) if args.speed_perturb else (1.0,)
    partitions = {
        name: load_manifest(args.data / f"{name}.tsv") for name in ["train", "dev", "test"]
    }
    metadata = load_metadata(args.metadata) if args.metadata else None
    if args.balance_corpora and (not metadata or not args.samples_per_epoch):
        raise ValueError("Balanced sampling requires metadata and an explicit update budget")
    if metadata:
        validate_partitions(partitions, metadata)
    else:
        validate_disjoint(partitions)
    if args.train_corpus:
        if not metadata:
            raise ValueError("Training corpus selection requires metadata")
        partitions["train"] = [
            r
            for r in partitions["train"]
            if record_for(r[0], metadata)["corpus"] == args.train_corpus
        ]
    if not 1 <= args.train_max_duration <= 29 or not 1 <= args.eval_max_duration <= 29:
        raise ValueError("Duration bounds must be within 1..29 seconds")
    accepted = {}
    durations = {}
    for name, rows in partitions.items():
        # Filter rather than truncating audio while retaining its complete transcript.
        rows = [
            (path, text)
            for path, text in rows
            if 1
            <= sf.info(path).duration
            <= (args.train_max_duration if name == "train" else args.eval_max_duration)
        ]
        random.Random(839).shuffle(rows)
        accepted[name] = rows[: args.train_limit if name == "train" else args.eval_limit]
        if not accepted[name]:
            raise ValueError("No usable speech in " + name)
        durations[name] = sum(sf.info(path).duration for path, text in accepted[name])
    args.output.mkdir(parents=True, exist_ok=True)
    if metadata and any(
        record_for(p, metadata)["language"].split("-")[0] != "en" for p, _ in accepted["train"]
    ):
        raise ValueError(
            "This bounded runner declares an English checkpoint; other languages require an appropriate tokenizer and checkpoint language"
        )
    tok = CharTokenizer().fit([text for path, text in accepted["train"]])
    if args.warm_start or args.resume:
        previous = torch.load(
            args.warm_start or args.resume, map_location="cpu", weights_only=True
        )["config"]
        restored = (
            tuple(previous["cnn_channels"]),
            previous["hidden_size"],
            previous["num_lstm_layers"],
        )
        if restored != dimensions:
            raise ValueError(
                "Restored architecture differs; use matching architecture or train from scratch"
            )
        if previous.get("time_reduction", 2) != args.time_reduction:
            raise ValueError("Restored encoder stride differs; train the new stride from scratch")
        tokenizer_payload = Path(previous["tokenizer_path"]).read_bytes()
        if (
            previous.get("tokenizer_sha256")
            and hashlib.sha256(tokenizer_payload).hexdigest() != previous["tokenizer_sha256"]
        ):
            raise ValueError("Restored tokenizer fingerprint differs from checkpoint")
        original_tok = CharTokenizer.load(previous["tokenizer_path"])
        if (
            tok._char_to_id != original_tok._char_to_id
            or tok.normalization != original_tok.normalization
        ):
            raise ValueError("Restored tokenizer IDs or normalization differ")
    manifest_text = {
        name: "".join(f"{path}\t{text}\n" for path, text in rows) for name, rows in accepted.items()
    }
    accepted_hashes = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in manifest_text.items()
    }
    if args.resume:
        prior_hashes = previous.get("accepted_manifest_sha256")
        if previous.get("samples_per_epoch") != args.samples_per_epoch:
            raise ValueError("Resume sampling budget changed; use explicit warm-start")
        if previous.get("balance_corpora", False) != args.balance_corpora:
            raise ValueError("Resume sampling policy changed; use explicit warm-start")
        if previous.get("feature_masking", False) != args.feature_masking:
            raise ValueError("Resume augmentation changed; use explicit warm-start instead")
        if previous.get("masked_batch_norm", False) != args.masked_batch_norm:
            raise ValueError("Resume normalization changed; use explicit warm-start instead")
        for key, value, default in (
            ("seed", args.seed, 839),
            ("batch_size", args.batch_size, 8),
            ("bucket_batches", args.bucket_batches, False),
            ("spec_augment", args.spec_augment, False),
            ("speed_factors", list(speed_factors), [1.0]),
            ("amp", args.amp, False),
        ):
            if previous.get(key, default) != value:
                raise ValueError(f"Resume {key} changed; use explicit warm-start instead")
        if prior_hashes and prior_hashes != accepted_hashes:
            raise ValueError("Resume data selection changed; use explicit warm-start instead")
        if not prior_hashes:
            prior_dir = Path(previous["tokenizer_path"]).parent
            for name, text in manifest_text.items():
                old = prior_dir / f"{name}.tsv"
                if (
                    not old.is_file()
                    or hashlib.sha256(old.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                    != accepted_hashes[name]
                ):
                    raise ValueError(
                        "Legacy resume data cannot be matched; use explicit warm-start"
                    )
    if args.resume or args.warm_start:
        (args.output / "tokenizer.json").write_bytes(tokenizer_payload)
    else:
        tok.save(args.output / "tokenizer.json")
    for name, text in manifest_text.items():
        (args.output / f"{name}.tsv").write_bytes(text.encode("utf-8"))
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "accepted_manifest_sha256": accepted_hashes,
                "audio_hours": {name: seconds / 3600 for name, seconds in durations.items()},
                "train_max_duration": args.train_max_duration,
                "eval_max_duration": args.eval_max_duration,
            }
        ),
        encoding="utf-8",
    )
    lengths = (
        [sf.info(path).duration for path, _ in accepted["train"]] if args.bucket_batches else None
    )
    loaders = {
        name: build_dataloader(
            rows,
            tok,
            batch_size=args.batch_size if name == "train" else 1,
            shuffle=name == "train",
            samples_per_epoch=args.samples_per_epoch if name == "train" else None,
            sampling_weights=balanced_weights(rows, metadata)
            if name == "train" and args.balance_corpora
            else None,
            # Slowed copies (0.9) are longer; allow for them in training only.
            max_frames=int(
                (
                    args.train_max_duration / min(speed_factors)
                    if name == "train"
                    else args.eval_max_duration
                )
                * 100
            )
            + 100,
            cache_dir=None
            if name == "train" and args.no_train_feature_cache
            else (args.feature_cache or args.output / "features") / name,
            seed=args.seed,
            num_workers=args.num_workers if name == "train" else 0,
            lengths=lengths if name == "train" else None,
            speed_factors=speed_factors if name == "train" else (1.0,),
        )
        for name, rows in accepted.items()
    }
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    config = TrainConfig(
        cnn_channels=dimensions[0],
        hidden_size=dimensions[1],
        num_lstm_layers=dimensions[2],
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        peak_lr=args.peak_lr,
        warmup_steps=30,
        save_every=1,
        log_every=50,
        checkpoint_dir=str(args.output),
        tokenizer_path=str(args.output / "tokenizer.json"),
        device=str(device),
    )
    config.tokenizer_sha256 = hashlib.sha256(
        (args.output / "tokenizer.json").read_bytes()
    ).hexdigest()
    config.accepted_manifest_sha256 = accepted_hashes
    config.asr_language = "en"
    config.time_reduction = args.time_reduction
    config.transcript_normalization = tok.normalization
    config.masked_batch_norm = args.masked_batch_norm
    config.feature_masking = args.feature_masking
    config.spec_augment = args.spec_augment
    config.amp = args.amp
    config.seed = args.seed
    config.augmentation_seed = args.seed
    config.bucket_batches = args.bucket_batches
    config.speed_factors = list(speed_factors)
    config.samples_per_epoch = args.samples_per_epoch
    config.balance_corpora = args.balance_corpora
    if args.metadata:
        metadata_path = args.output / "metadata.jsonl"
        metadata_path.write_bytes(args.metadata.read_bytes())
        config.dataset_metadata_path = str(metadata_path.resolve())
        config.dataset_metadata_sha256 = metadata_digest(metadata_path)
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=True)["config"]
        if previous.get("tokenizer_sha256", config.tokenizer_sha256) != config.tokenizer_sha256:
            raise ValueError("Resume tokenizer does not match checkpoint training")
    model = CTCModel(
        tok.vocab_size,
        cnn_channels=config.cnn_channels,
        hidden_size=config.hidden_size,
        num_lstm_layers=config.num_lstm_layers,
        time_reduction=args.time_reduction,
    ).to(device)
    model.encoder.masked_batch_norm = args.masked_batch_norm
    parameter_count = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if args.warm_start:
        load_checkpoint(args.warm_start, model, device=device)
    print(
        json.dumps(
            {
                "counts": {name: len(rows) for name, rows in accepted.items()},
                "audio_hours": {name: seconds / 3600 for name, seconds in durations.items()},
                "warm_start": str(args.warm_start) if args.warm_start else None,
            }
        ),
        flush=True,
    )
    history = train(
        model, loaders["train"], loaders["dev"], config, device, resume_from=args.resume
    )
    load_checkpoint(args.output / "best.pt", model, device=device)
    report = {
        "data": "Explicit attributed corpus metadata"
        if metadata
        else "Mini LibriSpeech SLR31; CC BY 4.0; original CNN/BiLSTM CTC",
        "counts": {name: len(rows) for name, rows in accepted.items()},
        "seed": args.seed,
        "batch_size": args.batch_size,
        "bucket_batches": args.bucket_batches,
        "amp": args.amp,
        "spec_augment": args.spec_augment,
        "speed_factors": list(speed_factors),
        "audio_hours": {name: seconds / 3600 for name, seconds in durations.items()},
        "train_max_duration": args.train_max_duration,
        "eval_max_duration": args.eval_max_duration,
        "warm_start_sha256": hashlib.sha256(args.warm_start.read_bytes()).hexdigest()
        if args.warm_start
        else None,
        "checkpoint_sha256": hashlib.sha256((args.output / "best.pt").read_bytes()).hexdigest(),
        "accepted_manifest_sha256": accepted_hashes,
        "manifest_sha256": {
            name: hashlib.sha256((args.data / f"{name}.tsv").read_bytes()).hexdigest()
            for name in accepted
        },
        "peak_lr": args.peak_lr,
        "architecture": args.architecture,
        "parameters": parameter_count,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "time_reduction": args.time_reduction,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
        if device.type == "cuda"
        else None,
        "feature_masking": args.feature_masking,
        "samples_per_epoch": args.samples_per_epoch,
        "train_corpus": args.train_corpus,
        "balance_corpora": args.balance_corpora,
        "metadata_sha256": metadata_digest(args.metadata) if args.metadata else None,
        "history": history,
        "development": evaluate_dataset(model, loaders["dev"], tok, greedy_decode, device),
        "held_out_speakers": None
        if args.skip_test
        else evaluate_dataset(model, loaders["test"], tok, greedy_decode, device),
        "limits": "English read-speech experiment; no multilingual/noisy speech claim; batch-one evaluation avoids padded-context decoding",
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
