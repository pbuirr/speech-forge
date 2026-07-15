"""
speech_engine/positional_encoding.py — Positional encodings, from scratch.

Transformers process a sequence as a set (all positions in parallel), so
positional information must be injected explicitly. Without it, the model
would be permutation-invariant — "the cat sat" and "sat cat the" would
produce identical representations.

Two types are implemented:

SinusoidalEncoding (for the encoder):
    Fixed, non-learned positional encoding from "Attention Is All You Need"
    (Vaswani et al. 2017). Uses sine/cosine functions at different frequencies:

        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Each position gets a unique signature across the d_model dimensions.
    The choice of base 10000 ensures that nearby positions have similar
    encodings and distant positions are distinguishable.

    We use this for the encoder because audio frame positions can be
    arbitrary length (open-ended), and sinusoidal encodings generalise to
    lengths longer than seen during training.

LearnedPositionalEmbedding (for the decoder):
    Learned embeddings for each absolute position (like BERT, GPT, Whisper).
    Initialised from N(0, 0.02) — the same distribution as token embeddings.

    We use this for the decoder because:
    - Decoder positions are bounded (max transcript length is known)
    - Learned embeddings allow the model to adapt positional information
      jointly with the rest of the decoder
    - Whisper uses learned positional embeddings and it works well
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from speech_engine.tokenizer import PAD_ID


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Build the sinusoidal encoding table.

    Returns:
        (1, max_len, d_model) — the leading 1 allows broadcasting over batch.

    Example::

        pe = sinusoidal_encoding(3000, 256)  # pre-compute for all positions
        x = x + pe[:, :T]                   # add to sequence of length T
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

    # Compute frequency denominators in log-space for numerical stability.
    # div_term[i] = 1 / 10000^(2i/d_model) = exp(-2i * log(10000) / d_model)
    half_d = d_model // 2
    div_term = torch.exp(
        torch.arange(0, half_d, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sin
    pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims: cos

    return pe.unsqueeze(0)  # (1, max_len, d_model)


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Adds a pre-computed sinusoidal position table to the input.
    The table is registered as a buffer (not a parameter) so it
    moves to the correct device with .to(device) but is not updated
    by the optimizer.

    Args:
        d_model:  model dimension
        max_len:  maximum sequence length to pre-compute (3000 = 30s at 10ms/frame)
        dropout:  applied after adding positional encoding
    """

    def __init__(self, d_model: int, max_len: int = 3000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Register as buffer: moves with .to(device), not updated by optimizer
        self.register_buffer("pe", sinusoidal_encoding(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (Whisper-style).

    Maps each position index 0..max_len-1 to a d_model-dimensional vector.
    Initialised from N(0, 0.02) — the same as token embeddings.

    Args:
        d_model:  model dimension
        max_len:  maximum number of decoder positions
        dropout:  applied after adding positional embedding
    """

    def __init__(self, d_model: int, max_len: int = 448, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) — already token-embedded
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.dropout(x + self.embedding(positions))
