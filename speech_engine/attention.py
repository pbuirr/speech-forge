"""Multi-head scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V

Used for encoder self-attention, causal decoder self-attention and
decoder-to-encoder cross-attention in the Transformer models.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    Q: torch.Tensor,  # (B, H, T_q, d_k)
    K: torch.Tensor,  # (B, H, T_k, d_k)
    V: torch.Tensor,  # (B, H, T_k, d_v)
    mask: torch.Tensor | None = None,  # (B, 1, T_q, T_k) bool — True = attend
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(Q Kᵀ / √d_k) V with an optional boolean mask (True = attend).

    Returns:
        output:       (B, H, T_q, d_v)
        attn_weights: (B, H, T_q, T_k)
    """
    d_k = Q.size(-1)
    scale = math.sqrt(d_k)

    # (B, H, T_q, T_k) — raw attention logits
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    if mask is not None:
        # mask=False means "blocked" — fill those positions with -inf so
        # softmax drives their weight to 0.
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    # If a query attends to all-masked keys, softmax produces NaN → replace.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, V)  # (B, H, T_q, d_v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q, K, V projections.

    Args:
        d_model:  model dimension (must be divisible by n_heads)
        n_heads:  number of parallel attention heads
        dropout:  dropout on attention weights

    Usage::

        attn = MultiHeadAttention(d_model=256, n_heads=4)

        # Self-attention:
        out, weights = attn(x, x, x)              # x: (B, T, 256)

        # Cross-attention with padding mask:
        out, weights = attn(query, memory, memory, mask=mem_mask)

        # Causal self-attention:
        out, weights = attn(x, x, x, mask=causal_mask)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # No bias in Q/K/V projections (a common choice)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, T_q, d_model)
        key: torch.Tensor,  # (B, T_k, d_model)
        value: torch.Tensor,  # (B, T_k, d_model)
        mask: torch.Tensor | None = None,  # (B, 1, T_q, T_k) bool — True = attend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:       (B, T_q, d_model)
            attn_weights: (B, n_heads, T_q, T_k)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H, D = self.n_heads, self.d_k

        # Linear projections then reshape to (B, H, T, d_k)
        Q = self.w_q(query).view(B, T_q, H, D).transpose(1, 2)  # (B, H, T_q, d_k)
        K = self.w_k(key).view(B, T_k, H, D).transpose(1, 2)  # (B, H, T_k, d_k)
        V = self.w_v(value).view(B, T_k, H, D).transpose(1, 2)  # (B, H, T_k, d_k)

        out, attn_w = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.drop)

        # Merge heads: (B, H, T_q, d_k) → (B, T_q, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.w_o(out), attn_w
