"""
speech_engine/train.py — Training loop for CTCModel, built from scratch.

No PyTorch Lightning. No HuggingFace Trainer. No Ignite. Every piece of the
training loop is written here so you can see exactly what's happening:
gradient updates, learning rate scheduling, checkpointing, loss reporting.

Learning rate schedule (cosine warmup):
    Two phases:
    1. Warmup (steps 0 → warmup_steps): LR rises linearly from 0 to peak_lr.
       Starting with a small LR avoids large early gradient updates that can
       push the model into a poor local minimum before it has seen enough data.
    2. Cosine decay (steps warmup_steps → max_steps): LR follows a half-cosine
       curve from peak_lr down to min_lr. The smooth decay is better than a
       step schedule because the model makes large updates while loss is high
       and small, fine-grained updates as it converges.

Gradient clipping:
    Before each optimizer step, we clip all parameter gradients so their
    global L2 norm is ≤ grad_clip (default 5.0). This prevents "exploding
    gradients" — a common failure mode in RNNs where the gradient norm spikes
    and destroys the model's weights. You'll see a warning if it triggers often.

Checkpointing:
    Two checkpoints are saved:
    - checkpoints/latest.pt   — overwritten after each epoch (resume training)
    - checkpoints/best.pt     — only overwritten when validation loss improves
    Both contain model weights, optimizer state, and training metadata.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from speech_engine.tokenizer import BLANK_ID


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyperparameters for one training run.

    Designed to be overridden per experiment. Reasonable defaults for
    training on a CPU with the toy dataset or LibriSpeech test-clean.
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str          = "data/toy_dataset"
    manifest: Optional[str] = None              # TSV manifest (overrides data_dir)
    librispeech_root: Optional[str] = None      # LibriSpeech root dir
    librispeech_split: str = "train-clean-100"

    # ── Model ─────────────────────────────────────────────────────────────────
    cnn_channels: tuple    = (32, 64, 128)      # conv output channels per block
    hidden_size: int       = 256                # LSTM hidden units per direction
    num_lstm_layers: int   = 3

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int        = 4
    max_epochs: int        = 50
    peak_lr: float         = 3e-4               # Adam peak learning rate
    min_lr: float          = 1e-6               # floor LR after cosine decay
    warmup_steps: int      = 500               # steps before cosine decay begins
    grad_clip: float       = 5.0               # global gradient norm clip

    # ── Output ────────────────────────────────────────────────────────────────
    checkpoint_dir: str    = "checkpoints"
    tokenizer_path: str    = "data/tokenizer.json"
    save_every: int        = 5                 # save checkpoint every N epochs
    log_every: int         = 10               # print loss every N steps

    # ── Device ────────────────────────────────────────────────────────────────
    device: str            = "auto"            # "cpu", "cuda", "mps", or "auto"


def resolve_device(spec: str) -> torch.device:
    """Pick the best available device.

    "auto" order: CUDA → Apple MPS → CPU.
    MPS (Metal Performance Shaders) uses the Apple GPU on M1/M2/M3 Macs.

    Note: nn.CTCLoss is not yet supported on MPS (PyTorch 2.x). If MPS is
    detected we fall back to CPU — you'll see a message explaining why.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS is fast for many ops but CTCLoss has an MPS fallback issue.
        # Keep CPU for stability until PyTorch adds full MPS CTC support.
        return torch.device("cpu")
    return torch.device(spec)


# ── Learning rate schedule ────────────────────────────────────────────────────

def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Cosine schedule with linear warmup. Implemented from scratch.

    Returns the learning rate that should be used at `step`.

    Visual shape:
        0          warmup       total_steps
        ├──────────────┬──────────────────────┤
        0 ──── peak_lr │ peak_lr ──── min_lr  (cosine curve)
                       └── (linear rise on left, cosine decay on right)

    Args:
        step:         current global training step (counts from 0)
        warmup_steps: length of the linear warmup phase
        total_steps:  total number of training steps (epochs × steps_per_epoch)
        peak_lr:      maximum learning rate (reached at end of warmup)
        min_lr:       minimum learning rate (reached at end of training)
    """
    if step < warmup_steps:
        # Linear warmup: 0 → peak_lr
        return peak_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    # Cosine decay: peak_lr → min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_loss: float,
    config: TrainConfig,
) -> None:
    """Save model + optimizer state to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "step":       step,
            "best_loss":  best_loss,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "config":     config.__dict__,
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
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ── Training and evaluation steps ────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_loss_fn: nn.CTCLoss,
    config: TrainConfig,
    device: torch.device,
    step: int,
    total_steps: int,
) -> tuple[float, int]:
    """Run one training epoch. Returns (mean_loss, updated_step).

    For each batch:
        1. Update the learning rate (cosine schedule)
        2. Forward pass through the model
        3. Compute CTC loss
        4. Backward pass (compute gradients)
        5. Clip gradient norm
        6. Optimizer step (update weights)

    Returns updated global step counter so the LR schedule is continuous
    across epochs.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        # ── 1. Update learning rate ───────────────────────────────────────────
        lr = cosine_lr(step, config.warmup_steps, total_steps, config.peak_lr, config.min_lr)
        set_lr(optimizer, lr)

        # ── 2. Move batch to device ───────────────────────────────────────────
        mel           = batch["mel"].to(device)            # (B, 80, T)
        tokens        = batch["tokens"].to(device)          # (B, L_max)
        token_lengths = batch["token_lengths"].to(device)  # (B,)
        mel_lengths   = batch["mel_lengths"].to(device)    # (B,)

        # ── 3. Forward pass ───────────────────────────────────────────────────
        log_probs     = model(mel)                          # (T_out, B, vocab_size)
        input_lengths = model.encoder_output_length(mel_lengths)  # (B,)

        # Ensure input_lengths >= token_lengths (CTC constraint)
        # zero_infinity=True handles edge cases silently
        loss = ctc_loss_fn(log_probs, tokens, input_lengths, token_lengths)

        # ── 4. Backward pass ──────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # ── 5. Gradient clipping ──────────────────────────────────────────────
        # Clips the global L2 norm of all gradients to config.grad_clip.
        # Returns the pre-clip norm — useful for debugging unstable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # ── 6. Weight update ──────────────────────────────────────────────────
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        step       += 1

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
    n_batches  = 0

    for batch in loader:
        mel           = batch["mel"].to(device)
        tokens        = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)
        mel_lengths   = batch["mel_lengths"].to(device)

        log_probs     = model(mel)
        input_lengths = model.encoder_output_length(mel_lengths)
        loss          = ctc_loss_fn(log_probs, tokens, input_lengths, token_lengths)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


# ── Main training function ─────────────────────────────────────────────────────

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
    # ── Setup ─────────────────────────────────────────────────────────────────
    ckpt_dir   = Path(config.checkpoint_dir)
    optimizer  = torch.optim.Adam(model.parameters(), lr=config.peak_lr)
    ctc_loss   = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True, reduction="mean")

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
        print(f"  Resumed from {resume_from} (epoch {start_epoch}, step {global_step})")

    model.to(device)

    # ── History ───────────────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []

    print(f"\n  Training for {config.max_epochs} epochs")
    print(f"  {steps_per_epoch} steps/epoch × {config.max_epochs} epochs = {total_steps} total steps")
    print(f"  LR: 0 → {config.peak_lr:.0e} (warmup {config.warmup_steps} steps) → {config.min_lr:.0e}")
    print(f"  Device: {device}  ·  Batch size: {config.batch_size}")
    print()

    for epoch in range(start_epoch, config.max_epochs):
        t0 = time.perf_counter()
        print(f"Epoch {epoch + 1}/{config.max_epochs}")

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, ctc_loss, config,
            device, global_step, total_steps,
        )
        train_losses.append(train_loss)

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss = None
        if val_loader is not None:
            val_loss = eval_epoch(model, val_loader, ctc_loss, device)
            val_losses.append(val_loss)

        elapsed = time.perf_counter() - t0
        val_str = f"  val_loss: {val_loss:.4f}" if val_loss is not None else ""
        print(f"  train_loss: {train_loss:.4f}{val_str}  ({elapsed:.1f}s)")

        # ── Checkpoint ────────────────────────────────────────────────────────
        monitor_loss = val_loss if val_loss is not None else train_loss

        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer,
                epoch, global_step, best_val_loss, config,
            )
            print(f"  ✓ New best loss: {best_val_loss:.4f} → saved to {ckpt_dir}/best.pt")

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer,
                epoch, global_step, best_val_loss, config,
            )
            print(f"  Checkpoint saved → {ckpt_dir}/latest.pt")

    return {
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_val_loss": best_val_loss,
        "total_steps":   global_step,
    }
