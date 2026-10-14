"""Positional encodings for the Transformer models.

``SinusoidalEncoding`` (encoder): fixed sin/cos encodings (Vaswani et al., 2017),
which extend to sequence lengths not seen in training.
``LearnedPositionalEmbedding`` (decoder): one learned vector per position,
initialised from N(0, 0.02).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


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

    pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sin
    pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims: cos

    return pe.unsqueeze(0)  # (1, max_len, d_model)


class SinusoidalEncoding(nn.Module):
    """Adds a fixed sinusoidal position table to the input.

    The table is a buffer, so it follows the module's device but isn't trained.

    Args:
        d_model:  model dimension
        max_len:  positions to precompute (3000 = 30 s at 10 ms per frame)
        dropout:  applied after adding the encoding
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
    """One learned d_model vector per position 0..max_len-1, initialised from N(0, 0.02).

    Args:
        d_model:  model dimension
        max_len:  maximum decoder positions
        dropout:  applied after adding the embedding
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
