"""CTC training loop.

Linear warmup then cosine decay of the learning rate, gradient clipping,
optional SpecAugment and mixed precision (CUDA), and checkpoints
(``latest.pt`` every epoch, ``best.pt`` on improved dev loss) that resume
exactly, including sampler and RNG state.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from speech_engine.tokenizer import BLANK_ID


# Configuration


@dataclass
class TrainConfig:
    """Hyperparameters for one training run."""

    # Data
    data_dir: str = "data/toy_dataset"
    manifest: Optional[str] = None  # TSV manifest (overrides data_dir)
    librispeech_root: Optional[str] = None  # LibriSpeech root dir
    librispeech_split: str = "train-clean-100"

    # Model
    cnn_channels: tuple = (32, 64, 128)  # conv output channels per block
    hidden_size: int = 256  # LSTM hidden units per direction
    num_lstm_layers: int = 3

    # Training
    batch_size: int = 4
    max_epochs: int = 50
    peak_lr: float = 3e-4  # Adam peak learning rate
    min_lr: float = 1e-6  # floor LR after cosine decay
    warmup_steps: int = 500  # steps before cosine decay begins
    grad_clip: float = 5.0  # global gradient norm clip
    feature_masking: bool = False
    spec_augment: bool = False  # stronger multi-mask policy (augment.spec_augment)
    amp: bool = False  # CUDA mixed precision; CTC loss stays float32
    augmentation_seed: int = 839

    # Output
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "data/tokenizer.json"
    save_every: int = 5  # save checkpoint every N epochs
    log_every: int = 10  # print loss every N steps

    # Device
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"


def resolve_device(spec: str) -> torch.device:
    """Resolve "auto" to CUDA, then Apple MPS, then CPU.

    nn.CTCLoss isn't supported on MPS, so MPS falls back to CPU with a message.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS is fast for many ops but CTCLoss has an MPS fallback issue.
        # Keep CPU for stability until PyTorch adds full MPS CTC support.
        return torch.device("cpu")
    return torch.device(spec)


# Learning rate schedule


def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Learning rate at ``step``: linear warmup to ``peak_lr``, then cosine decay to ``min_lr``.

    Args:
        step:         global training step, from 0
        warmup_steps: warmup length
        total_steps:  total training steps
        peak_lr:      learning rate at the end of warmup
        min_lr:       learning rate at the end of training
    """
    if step < warmup_steps:
        # Linear warmup: 0 → peak_lr
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    # Cosine decay: peak_lr → min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# Checkpointing


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    config: TrainConfig,
    rng_state: Optional[dict] = None,
    scaler=None,
) -> None:
    """Save model + optimizer state to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "best_loss": best_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.__dict__,
            "rng_state": rng_state,
            **({"scaler": scaler.state_dict()} if scaler is not None else {}),
        },
        str(path),
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a checkpoint. Returns the metadata dict (epoch, step, best_loss)."""
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    if hasattr(model, "time_reduction") and model.time_reduction != ckpt.get("config", {}).get(
        "time_reduction", 2
    ):
        raise ValueError("Checkpoint encoder stride differs from the constructed model")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# Training and evaluation steps


def validate_ctc_alignment(tokens, token_lengths, input_lengths):
    positions = torch.arange(max(0, tokens.shape[1] - 1), device=tokens.device)[None, :]
    repeats = ((tokens[:, 1:] == tokens[:, :-1]) & (positions < token_lengths[:, None] - 1)).sum(
        dim=1
    )
    if torch.any(input_lengths < token_lengths + repeats):
        raise ValueError(
            "CTC output frames cannot align the complete transcript, including repeated characters"
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss_fn: nn.CTCLoss,
    config: TrainConfig,
    device: torch.device,
    step: int,
    total_steps: int,
    scaler=None,
) -> tuple[float, int]:
    """Run one epoch and return (mean_loss, global_step).

    Per batch: set the scheduled learning rate, optionally augment, forward, CTC
    loss, backward (scaled under AMP), clip gradients and step the optimizer.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Epoch/step-derived separate RNG preserves the control's shuffle/dropout RNG.
    augmentation_rng = random.Random(config.augmentation_seed + step)

    for batch in loader:
        # 1. Update learning rate
        lr = cosine_lr(step, config.warmup_steps, total_steps, config.peak_lr, config.min_lr)
        set_lr(optimizer, lr)

        # 2. Move batch to device
        mel = batch["mel"].to(device)  # (B, 80, T)
        tokens = batch["tokens"].to(device)  # (B, L_max)
        token_lengths = batch["token_lengths"].to(device)  # (B,)
        mel_lengths = batch["mel_lengths"].to(device)  # (B,)

        if getattr(config, "spec_augment", False):
            from speech_engine.augment import spec_augment

            mel = spec_augment(mel, mel_lengths, augmentation_rng)
        elif config.feature_masking:
            from speech_engine.augment import mask_features

            mel = mask_features(mel, mel_lengths, augmentation_rng)

        # 3. Forward pass
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler is not None
        ):
            log_probs = model(mel, mel_lengths)  # (T_out, B, vocab_size)
        log_probs = log_probs.float()  # CTC is computed in float32
        input_lengths = model.encoder_output_length(mel_lengths)  # (B,)
        validate_ctc_alignment(tokens, token_lengths, input_lengths)

        # Ensure input_lengths >= token_lengths (CTC constraint)
        # zero_infinity=True handles edge cases silently
        loss = ctc_loss_fn(log_probs, tokens, input_lengths, token_lengths)
        if not torch.isfinite(loss):
            raise ValueError(
                "Non-finite CTC loss; check audio, transcripts and training configuration"
            )

        # 4. Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip true gradients, not scaled ones
        else:
            loss.backward()

        # 5. Gradient clipping
        # Clips the global L2 norm of all gradients to config.grad_clip.
        # Returns the pre-clip norm — useful for debugging unstable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # 6. Weight update
        if scaler is not None:
            scaler.step(optimizer)  # skips the step if gradients overflowed
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        step += 1

        if step % config.log_every == 0:
            print(
                f"    step {step:>6}  loss {loss.item():.4f}"
                f"  lr {lr:.2e}  grad_norm {grad_norm:.3f}"
            )

    mean_loss = total_loss / max(1, n_batches)
    return mean_loss, step


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    device: torch.device,
) -> float:
    """Run one evaluation epoch. Returns mean CTC loss (no gradients)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)

        log_probs = model(mel, mel_lengths)
        input_lengths = model.encoder_output_length(mel_lengths)
        validate_ctc_alignment(tokens, token_lengths, input_lengths)
        loss = ctc_loss_fn(log_probs, tokens, input_lengths, token_lengths)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# Main training function


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainConfig,
    device: torch.device,
    resume_from: Optional[Path] = None,
) -> dict:
    """Full training loop.

    Args:
        model:        CTCModel instance (already moved to device)
        train_loader: DataLoader for training data
        val_loader:   DataLoader for validation data (None → skip validation)
        config:       TrainConfig with all hyperparameters
        device:       torch.device
        resume_from:  path to a checkpoint to resume from

    Returns:
        dict with "train_losses", "val_losses", "best_val_loss", "total_steps"
    """
    # Setup
    ckpt_dir = Path(config.checkpoint_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.peak_lr)
    ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True, reduction="mean")
    if getattr(config, "amp", False) and device.type != "cuda":
        raise ValueError("Mixed precision (amp) requires CUDA")
    scaler = torch.amp.GradScaler("cuda") if getattr(config, "amp", False) else None

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    steps_per_epoch = len(train_loader)
    total_steps = config.max_epochs * steps_per_epoch

    if resume_from and Path(resume_from).exists():
        ckpt = load_checkpoint(Path(resume_from), model, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_loss", float("inf"))
        saved_rng = ckpt.get("rng_state")
        if saved_rng:
            torch.set_rng_state(saved_rng["torch"].cpu())
            random.setstate(saved_rng["python"])
            if device.type == "cuda" and saved_rng.get("cuda") is not None:
                torch.cuda.set_rng_state(saved_rng["cuda"].cpu(), device)
            for name, generator in [
                ("sampler", getattr(train_loader.sampler, "generator", None)),
                ("batch_sampler", getattr(train_loader.batch_sampler, "generator", None)),
                ("loader", getattr(train_loader, "generator", None)),
            ]:
                if generator is not None and saved_rng.get(name) is not None:
                    generator.set_state(saved_rng[name].cpu())
            if scaler is not None and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
        else:
            print(
                "  Legacy checkpoint has no sampling/RNG state; restoring weights and optimizer only"
            )
        print(f"  Resumed from {resume_from} (epoch {start_epoch}, step {global_step})")

    model.to(device)

    # History
    train_losses: list[float] = []
    val_losses: list[float] = []
    epoch_elapsed_s: list[float] = []

    print(f"\n  Training for {config.max_epochs} epochs")
    print(
        f"  {steps_per_epoch} steps/epoch × {config.max_epochs} epochs = {total_steps} total steps"
    )
    print(
        f"  LR: 0 → {config.peak_lr:.0e} (warmup {config.warmup_steps} steps) → {config.min_lr:.0e}"
    )
    print(f"  Device: {device}  ·  Batch size: {config.batch_size}")
    print()

    for epoch in range(start_epoch, config.max_epochs):
        t0 = time.perf_counter()
        print(f"Epoch {epoch + 1}/{config.max_epochs}")

        # Train
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            ctc_loss,
            config,
            device,
            global_step,
            total_steps,
            scaler=scaler,
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = None
        if val_loader is not None:
            val_loss = eval_epoch(model, val_loader, ctc_loss, device)
            val_losses.append(val_loss)

        elapsed = time.perf_counter() - t0
        epoch_elapsed_s.append(elapsed)
        val_str = f"  val_loss: {val_loss:.4f}" if val_loss is not None else ""
        print(f"  train_loss: {train_loss:.4f}{val_str}  ({elapsed:.1f}s)")

        # Checkpoint
        monitor_loss = val_loss if val_loss is not None else train_loss
        rng_state = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        for name, generator in [
            ("sampler", getattr(train_loader.sampler, "generator", None)),
            ("batch_sampler", getattr(train_loader.batch_sampler, "generator", None)),
            ("loader", getattr(train_loader, "generator", None)),
        ]:
            if generator is not None:
                rng_state[name] = generator.get_state()

        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                config,
                rng_state=rng_state,
                scaler=scaler,
            )
            print(f"  ✓ New best loss: {best_val_loss:.4f} → saved to {ckpt_dir}/best.pt")

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                config,
                rng_state=rng_state,
                scaler=scaler,
            )
            print(f"  Checkpoint saved → {ckpt_dir}/latest.pt")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
        "epoch_elapsed_s": epoch_elapsed_s,
    }
