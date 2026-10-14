"""Training loop for the Transformer speech recognizer.

Same learning-rate schedule as ``train.py``. ``prepare_seq2seq_batch`` adds
BOS/EOS to the CTC-style token batches and shifts them into decoder input and
target.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from speech_engine.tokenizer import BOS_ID, EOS_ID, PAD_ID
from speech_engine.train import cosine_lr, save_checkpoint, set_lr


# Batch preparation


def prepare_seq2seq_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a CTC-style batch into seq2seq decoder_input / target tensors.

    Input (from collate_fn):
        batch["tokens"]        (B, L_max)  — token IDs, no BOS/EOS, right-padded
        batch["token_lengths"] (B,)        — actual length per example
        batch["mel"]           (B, 80, T)
        batch["mel_lengths"]   (B,)

    Output:
        mel:             (B, 80, T)
        mel_lengths:     (B,)
        decoder_input:   (B, L+1)  — [BOS, c1, c2, ..., cn,    PAD...]
        target:          (B, L+1)  — [c1, c2, ..., cn, EOS,    PAD...]
        target_lengths:  (B,)      — L+1 (BOS/EOS each add 1 to length)

    CrossEntropyLoss ignores PAD_ID positions, so the padding is harmless.
    """
    mel = batch["mel"].to(device)
    mel_lengths = batch["mel_lengths"].to(device)
    tokens = batch["tokens"]  # stay on CPU for list ops
    token_lengths = batch["token_lengths"]

    B, L = tokens.shape
    new_L = L + 1  # +1 for the BOS token on the input side (or EOS on target side)

    dec_inputs = torch.full((B, new_L), PAD_ID, dtype=torch.long)
    targets = torch.full((B, new_L), PAD_ID, dtype=torch.long)
    t_lengths = torch.zeros(B, dtype=torch.long)

    for b in range(B):
        n = int(token_lengths[b])
        toks = tokens[b, :n].tolist()

        dec_inputs[b, 0] = BOS_ID
        dec_inputs[b, 1 : n + 1] = torch.tensor(toks, dtype=torch.long)

        targets[b, :n] = torch.tensor(toks, dtype=torch.long)
        targets[b, n] = EOS_ID

        t_lengths[b] = n + 1

    return mel, mel_lengths, dec_inputs.to(device), targets.to(device), t_lengths


# Training / eval steps


def train_seq2seq_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss_fn: nn.CrossEntropyLoss,
    config,  # TrainConfig (from train.py) or dict-like with same fields
    device: torch.device,
    step: int,
    total_steps: int,
) -> tuple[float, int]:
    """One seq2seq training epoch.

    Returns:
        (mean_loss, updated_step)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        lr = cosine_lr(step, config.warmup_steps, total_steps, config.peak_lr, config.min_lr)
        set_lr(optimizer, lr)

        mel, mel_lengths, dec_input, target, _ = prepare_seq2seq_batch(batch, device)

        # Forward: (B, L+1, vocab_size)
        logits = model(mel, dec_input, mel_lengths)

        B, L, V = logits.shape
        loss = ce_loss_fn(logits.reshape(B * L, V), target.reshape(B * L))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        step += 1

        if step % config.log_every == 0:
            print(
                f"    step {step:>6}  loss {loss.item():.4f}"
                f"  lr {lr:.2e}  grad_norm {grad_norm:.3f}"
            )

    return total_loss / max(1, n_batches), step


@torch.no_grad()
def eval_seq2seq_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        mel, mel_lengths, dec_input, target, _ = prepare_seq2seq_batch(batch, device)
        logits = model(mel, dec_input, mel_lengths)
        B, L, V = logits.shape
        loss = ce_loss_fn(logits.reshape(B * L, V), target.reshape(B * L))
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# Main training function


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config,
    device: torch.device,
    resume_from: Optional[Path] = None,
) -> dict:
    """Training loop for ``TransformerASR``; same structure as ``train.train`` with cross-entropy loss.

    Returns:
        dict with "train_losses", "val_losses", "best_val_loss", "total_steps"
    """
    from speech_engine.train import load_checkpoint

    ckpt_dir = Path(config.checkpoint_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.peak_lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)

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

    print(f"\n  Training for {config.max_epochs} epochs  (CrossEntropyLoss)")
    print(f"  {steps_per_epoch} steps/epoch × {config.max_epochs} = {total_steps} total")
    print(f"  LR: 0 → {config.peak_lr:.0e} → {config.min_lr:.0e}  |  device: {device}\n")

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(start_epoch, config.max_epochs):
        t0 = time.perf_counter()
        print(f"Epoch {epoch + 1}/{config.max_epochs}")

        train_loss, global_step = train_seq2seq_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            config,
            device,
            global_step,
            total_steps,
        )
        train_losses.append(train_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = eval_seq2seq_epoch(model, val_loader, ce_loss, device)
            val_losses.append(val_loss)

        elapsed = time.perf_counter() - t0
        val_str = f"  val_loss: {val_loss:.4f}" if val_loss is not None else ""
        print(f"  train_loss: {train_loss:.4f}{val_str}  ({elapsed:.1f}s)")

        monitor = val_loss if val_loss is not None else train_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            save_checkpoint(
                ckpt_dir / "best_transformer.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                config,
            )
            print(f"  ✓ New best: {best_val_loss:.4f} → {ckpt_dir}/best_transformer.pt")

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                ckpt_dir / "latest_transformer.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                config,
            )
            print(f"  Checkpoint → {ckpt_dir}/latest_transformer.pt")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
    }
