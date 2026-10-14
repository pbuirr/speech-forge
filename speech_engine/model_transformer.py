"""
speech_engine/model_transformer.py — Transformer encoder-decoder ASR model.

This is Phase 4 of LinguaForge. It replaces Phase 3's BiLSTM stack with a
Transformer encoder-decoder, following the architecture of Whisper but built
entirely from our own components.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  ConvEncoder  (reused from Phase 3)                         │
    │  mel (B,80,T) → (B, T//2, d_model)                         │
    └────────────────────────┬────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────┐
    │  TransformerEncoder                                         │
    │  SinusoidalEncoding + N × (self-attn + FFN)                │
    │  → memory (B, T//2, d_model)                               │
    └────────────────────────┬────────────────────────────────────┘
                             │  cross-attention
    ┌────────────────────────▼────────────────────────────────────┐
    │  TransformerDecoder                                         │
    │  TokenEmb + LearnedPosEmb + N × (self-attn + xattn + FFN) │
    │  tgt_ids (B,L) → logits (B, L, vocab_size)                 │
    └─────────────────────────────────────────────────────────────┘

Training (teacher forcing):
    decoder input:  [BOS, c1, c2, ..., cn]    (length L)
    decoder target: [c1, c2, ..., cn, EOS]    (length L)
    loss: CrossEntropyLoss(logits, target, ignore_index=PAD_ID)

Inference (auto-regressive greedy):
    start with [BOS]
    repeat: feed current sequence → take argmax of last logit → append
    stop when EOS generated or max_len reached

Pre-norm vs post-norm:
    We use pre-norm (LayerNorm before each sublayer, like Whisper/GPT-2 style):
        x = x + sublayer(norm(x))
    vs post-norm (original Transformer, "Attention Is All You Need"):
        x = norm(x + sublayer(x))
    Pre-norm is more stable at higher learning rates and doesn't require
    careful warmup tuning.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from speech_engine.attention import MultiHeadAttention
from speech_engine.model_ctc import ConvEncoder
from speech_engine.positional_encoding import LearnedPositionalEmbedding, SinusoidalEncoding
from speech_engine.tokenizer import BOS_ID, EOS_ID, PAD_ID


# ── Mask helpers ──────────────────────────────────────────────────────────────


def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask — position i may attend to positions 0..i.

    Shape: (1, 1, T, T) bool — True where attention is allowed.
    The extra dims broadcast over batch and heads.
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Key-padding mask — True where position is NOT padding.

    Shape: (B, 1, 1, max_len) bool — True = real token, False = pad.
    Broadcasting over heads (dim 1) and query positions (dim 2).
    """
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)
    return (arange < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, max_len)


# ── Building blocks ───────────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """Position-wise FFN: Linear(d_model→d_ff) → GELU → Linear(d_ff→d_model).

    Applied independently to each position after the attention sublayer.
    d_ff is typically 4× d_model (e.g. d_model=256, d_ff=1024).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """One encoder layer: pre-norm self-attention + pre-norm FFN.

    Pre-norm layout:
        out = x + self_attn(LayerNorm(x))
        out = out + ffn(LayerNorm(out))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=src_mask)
        x = x + self.drop(attn_out)
        # FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    """One decoder layer: causal self-attention + cross-attention + FFN.

    Three sublayers with pre-norm:
        out = x + self_attn(norm1(x), causal_mask)        # attend to own history
        out = out + cross_attn(norm2(out), memory, mask)  # attend to encoder
        out = out + ffn(norm3(out))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        return_cross_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 1. Causal self-attention over decoder sequence
        n1 = self.norm1(x)
        attn_out, _ = self.self_attn(n1, n1, n1, mask=tgt_mask)
        x = x + self.drop(attn_out)

        # 2. Cross-attention: queries from decoder, keys/values from encoder
        n2 = self.norm2(x)
        xattn_out, xattn_weights = self.cross_attn(n2, memory, memory, mask=memory_mask)
        x = x + self.drop(xattn_out)

        # 3. FFN
        x = x + self.drop(self.ffn(self.norm3(x)))

        if return_cross_attn:
            return x, xattn_weights  # xattn_weights: (B, H, T_out, T_src)
        return x


# ── Encoder and Decoder stacks ────────────────────────────────────────────────


class TransformerEncoder(nn.Module):
    """Stack of N TransformerEncoderLayers with sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        max_len: int = 3000,
    ) -> None:
        super().__init__()
        self.pos_enc = SinusoidalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Stack of N TransformerDecoderLayers with token + learned positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        max_len: int = 448,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = LearnedPositionalEmbedding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        return_cross_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            return_cross_attn: if True, also return the cross-attention weights
                from the LAST decoder layer, averaged over heads.
                Shape: (B, T_out, T_src) — row i is the attention distribution
                over source positions when generating output token i.
        """
        x = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_emb(x)
        last_xattn: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if is_last and return_cross_attn:
                x, last_xattn = layer(
                    x,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    return_cross_attn=True,
                )
            else:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        x = self.norm(x)  # (B, L, d_model)
        if return_cross_attn and last_xattn is not None:
            # Average over heads → (B, T_out, T_src)
            avg_attn = last_xattn.mean(dim=1)
            return x, avg_attn
        return x


# ── Full model ────────────────────────────────────────────────────────────────


class TransformerASR(nn.Module):
    """LinguaForge Phase 4: Transformer encoder-decoder ASR model.

    Combines:
      - ConvEncoder from Phase 3 (CNN frontend for spectrograms)
      - Transformer encoder (self-attention over audio frames)
      - Transformer decoder (auto-regressive token generation)

    Args:
        vocab_size:      number of output tokens (from CharTokenizer.vocab_size)
        n_mels:          mel bins (must match features.py N_MELS = 80)
        d_model:         Transformer hidden dimension
        n_heads:         number of attention heads (must divide d_model)
        d_ff:            FFN inner dimension (typically 4 × d_model)
        n_enc_layers:    Transformer encoder depth
        n_dec_layers:    Transformer decoder depth
        cnn_channels:    ConvEncoder channel counts per block
        dropout:         dropout throughout
        max_source_len:  max encoder output frames (1500 = 30s at 20ms/frame)
        max_target_len:  max decoder steps

    Training loss:
        nn.CrossEntropyLoss with ignore_index=PAD_ID applied to shifted targets.
        See train_transformer.py.

    Inference:
        Use model.generate(mel) for greedy decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.1,
        max_source_len: int = 1500,
        max_target_len: int = 448,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # CNN frontend: mel → (B, T//2, d_model)
        self.conv_encoder = ConvEncoder(
            n_mels=n_mels,
            channels=cnn_channels,
            hidden_size=d_model,
            dropout=dropout,
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model,
            n_heads,
            d_ff,
            n_enc_layers,
            dropout,
            max_source_len,
        )

        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size,
            d_model,
            n_heads,
            d_ff,
            n_dec_layers,
            dropout,
            max_target_len,
        )

        # Output projection — shared with decoder token embeddings (weight tying).
        # Weight tying: output_proj.weight = decoder.token_emb.weight
        # This enforces that the same vector space is used for encoding input tokens
        # and for scoring output tokens, and reduces the total parameter count.
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.decoder.token_emb.weight  # tie weights

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Token embedding already initialised to N(0, 0.02) in TransformerDecoder.__init__

    # ── Encode / decode as separate steps ─────────────────────────────────────

    def encode(
        self,
        mel: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """CNN + Transformer encode.

        Returns:
            memory:      (B, T//2, d_model) — encoder output
            memory_mask: (B, 1, 1, T//2) bool — True = valid frame (not padding)
        """
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.conv_encoder(x, mel_lengths)  # shared padding-aware acoustic frontend

        memory_mask = None
        if mel_lengths is not None:
            enc_len = self._enc_lengths(mel_lengths)
            memory_mask = make_padding_mask(enc_len, x.size(1))

        memory = self.encoder(x, src_mask=memory_mask)
        return memory, memory_mask

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transformer decode (full sequence — teacher forcing or incremental).

        Returns logits: (B, L, vocab_size)
        """
        L = tgt_ids.size(1)
        causal_mask = make_causal_mask(L, tgt_ids.device)  # (1, 1, L, L)
        dec_out = self.decoder(tgt_ids, memory, tgt_mask=causal_mask, memory_mask=memory_mask)
        return self.output_proj(dec_out)  # (B, L, vocab_size)

    # ── Full forward (teacher forcing) ────────────────────────────────────────

    def forward(
        self,
        mel: torch.Tensor,
        tgt_ids: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass for training.

        Args:
            mel:         (B, n_mels, T)  log-Mel spectrograms
            tgt_ids:     (B, L)  decoder input ids — [BOS, c1, c2, ..., cn]
            mel_lengths: (B,)   valid mel frame counts (for padding mask)

        Returns:
            logits: (B, L, vocab_size)
            Compare against target = [c1, c2, ..., cn, EOS] for cross-entropy.
        """
        memory, memory_mask = self.encode(mel, mel_lengths)
        return self.decode(tgt_ids, memory, memory_mask)

    # ── Auto-regressive inference ──────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
        max_len: int = 200,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Greedy auto-regressive decoding.

        Generates tokens one at a time until EOS or max_len.

        Args:
            mel:         (B, n_mels, T)
            mel_lengths: (B,) — valid frames, or None (assume no padding)
            max_len:     maximum tokens to generate per example
            temperature: divide logits by this before argmax (>1 = softer, <1 = sharper)

        Returns:
            list of B token-id lists (including BOS, excluding trailing padding)
        """
        self.eval()
        B = mel.size(0)
        memory, memory_mask = self.encode(mel, mel_lengths)

        generated = [[BOS_ID] for _ in range(B)]
        done = [False] * B

        for _ in range(max_len):
            if all(done):
                break

            tgt_ids = torch.tensor(generated, dtype=torch.long, device=mel.device)
            logits = self.decode(tgt_ids, memory, memory_mask)  # (B, L, V)
            next_id = (logits[:, -1, :] / max(temperature, 1e-8)).argmax(dim=-1)  # (B,)

            for b in range(B):
                if not done[b]:
                    tok = int(next_id[b])
                    generated[b].append(tok)
                    if tok == EOS_ID:
                        done[b] = True

        return generated

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _enc_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [ConvEncoder.output_length(int(l)) for l in mel_lengths],
            dtype=torch.long,
            device=mel_lengths.device,
        )

    def count_parameters(self) -> int:
        # Weight tying means output_proj.weight is shared — don't double-count.
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def summary(self) -> str:
        total = self.count_parameters() / 1e6
        conv_p = sum(p.numel() for p in self.conv_encoder.parameters()) / 1e6
        enc_p = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        dec_p = sum(p.numel() for p in self.decoder.parameters()) / 1e6
        n_enc = len(self.encoder.layers)
        n_dec = len(self.decoder.layers)
        n_heads = self.encoder.layers[0].self_attn.n_heads
        d_ff = self.encoder.layers[0].ffn.fc1.out_features
        return (
            f"TransformerASR\n"
            f"{'─' * 50}\n"
            f"  Architecture:   ConvEncoder → TransEnc → TransDec\n"
            f"  d_model:        {self.d_model}  "
            f"| n_heads: {n_heads}  | d_ff: {d_ff}\n"
            f"  enc layers:     {n_enc}  "
            f"| dec layers: {n_dec}\n"
            f"  vocab_size:     {self.vocab_size}  (weight-tied output proj)\n"
            f"{'─' * 50}\n"
            f"  ConvEncoder     {conv_p:.3f}M\n"
            f"  TransEncoder    {enc_p:.3f}M\n"
            f"  TransDecoder    {dec_p:.3f}M  (includes token emb)\n"
            f"  OutputProj      shared with token emb — 0 extra params\n"
            f"  TOTAL           {total:.3f}M\n"
            f"{'─' * 50}"
        )
