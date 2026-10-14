"""CNN + BiLSTM acoustic model trained with CTC.

    mel (B, 80, T)
      → ConvEncoder: 3 conv blocks, mel 80→10, time ÷2 (or ÷4)
      → BiLSTMStack: bidirectional LSTMs with residual connections and LayerNorm
      → Linear + log_softmax
      → (T', B, vocab), time-first for nn.CTCLoss

The default configuration has about 4.8M parameters and the compact one about
0.81M. Older checkpoints use twofold time reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Building blocks


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → GELU → Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv(x)
        if lengths is None or not self.training:
            x = self.bn(x)
        else:
            # Real time/frequency cells only: batch padding is not acoustic data.
            keep = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
            keep = keep[:, None, None, :]
            count = keep.sum() * x.shape[2]
            mean = (x * keep).sum((0, 2, 3)) / count
            variance = ((x - mean[None, :, None, None]).square() * keep).sum((0, 2, 3)) / count
            if self.bn.track_running_stats:
                with torch.no_grad():
                    self.bn.num_batches_tracked.add_(1)
                    momentum = (
                        self.bn.momentum
                        if self.bn.momentum is not None
                        else 1 / self.bn.num_batches_tracked.item()
                    )
                    self.bn.running_mean.lerp_(mean.detach(), momentum)
                    self.bn.running_var.lerp_(
                        (variance * count / (count - 1).clamp_min(1)).detach(), momentum
                    )
            x = (x - mean[None, :, None, None]) * torch.rsqrt(
                variance[None, :, None, None] + self.bn.eps
            )
            x = x * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
        return self.drop(self.act(x))


class ConvEncoder(nn.Module):
    """CNN frontend: (B, 1, n_mels, T) → (B, T', hidden_size).

    Each block halves the Mel axis (80 → 40 → 20 → 10). Block 3 always halves
    time; with ``time_reduction=4`` block 2 halves it too. The flattened
    channels × Mel features are projected to ``hidden_size``. Use
    ``output_length`` to get CTC input lengths.
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: tuple[int, int, int] = (32, 64, 128),
        hidden_size: int = 256,
        dropout: float = 0.1,
        time_reduction: int = 2,
    ) -> None:
        super().__init__()
        if time_reduction not in (2, 4):
            raise ValueError("Time reduction must be 2 or 4")
        self.time_reduction = time_reduction
        c0, c1, c2 = channels

        # Block 1: mel stride 2, time stride 1 → (B, c0, n_mels//2, T)
        self.block1 = ConvBlock(1, c0, (3, 3), stride=(2, 1), padding=(1, 1), dropout=dropout)
        # Block 2: mel stride 2, time stride 1 → (B, c1, n_mels//4, T)
        self.block2 = ConvBlock(
            c0,
            c1,
            (3, 3),
            stride=(2, 2 if time_reduction == 4 else 1),
            padding=(1, 1),
            dropout=dropout,
        )
        # Block 3: mel stride 2, time stride 2 → (B, c2, n_mels//8, T//2)
        self.block3 = ConvBlock(c1, c2, (3, 3), stride=(2, 2), padding=(1, 1), dropout=dropout)

        # After block 3, the mel dimension is n_mels // 8 = 10 for n_mels=80
        mel_out = n_mels // 8  # 80 → 10
        self.proj = nn.Linear(c2 * mel_out, hidden_size)
        self.masked_batch_norm = False

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        def mask_padding(value, valid):
            if valid is None:
                return value
            keep = torch.arange(value.shape[-1], device=value.device)[None, :] < valid[:, None]
            return value * keep[:, None, None, :]

        x = self.block1(x, lengths if self.masked_batch_norm else None)
        x = mask_padding(x, lengths)
        second_lengths = (
            (lengths + 1) // 2 if lengths is not None and self.time_reduction == 4 else lengths
        )
        x = self.block2(x, second_lengths if self.masked_batch_norm else None)
        x = mask_padding(x, second_lengths)
        final_lengths = (second_lengths + 1) // 2 if second_lengths is not None else None
        x = self.block3(x, final_lengths if self.masked_batch_norm else None)
        x = mask_padding(x, final_lengths)

        B, C, M, T_out = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_out, C * M)  # (B, T//2, C*M)
        return self.proj(x)  # (B, T//2, hidden_size)

    @staticmethod
    def output_length(input_length: int, time_reduction: int = 2) -> int:
        """Exact output frame count for a given input frame count.

        PyTorch Conv2d output formula: floor((L + 2*p - k) / s) + 1
        Block 3 time: kernel=3, padding=1, stride=2:
            T_out = floor((T + 2 - 3) / 2) + 1 = floor((T - 1) / 2) + 1

        Examples:
            input_length=300  → 150 output frames  (3s audio)
            input_length=3000 → 1500 output frames (30s audio)
        """
        return (input_length + time_reduction - 1) // time_reduction


class BiLSTMStack(nn.Module):
    """Stack of bidirectional LSTM layers with residual connections + LayerNorm.

    Layer structure:
        Layer 0: input_size → hidden_size*2  (first layer may change dimension)
        Layer k: hidden_size*2 → hidden_size*2  (residual, same dimension)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        out_size = hidden_size * 2  # bidirectional doubles the output

        self.lstms = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.projs = nn.ModuleList()  # input projections for residual

        for i in range(num_layers):
            in_size = input_size if i == 0 else out_size
            self.lstms.append(
                nn.LSTM(in_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
            )
            self.norms.append(nn.LayerNorm(out_size))
            self.drops.append(nn.Dropout(dropout))
            # Project the layer's input to out_size for the residual sum
            self.projs.append(
                nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()
            )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, input_size)
        for lstm, norm, drop, proj in zip(self.lstms, self.norms, self.drops, self.projs):
            residual = proj(x)  # (B, T, out_size) — same dim as LSTM output
            if lengths is None:
                out, _ = lstm(x)
            else:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                encoded, _ = lstm(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(
                    encoded, batch_first=True, total_length=x.shape[1]
                )
            x = norm(drop(out) + residual)  # residual + normalise
        return x  # (B, T, hidden_size*2)


# Full CTC model


class CTCModel(nn.Module):
    """CTC speech recognizer: ConvEncoder → BiLSTMStack → Linear → log_softmax.

    Args:
        vocab_size:      number of output classes (CharTokenizer.vocab_size)
        n_mels:          Mel bins (must match features.N_MELS)
        cnn_channels:    output channels of the 3 conv blocks
        hidden_size:     LSTM hidden size per direction
        num_lstm_layers: number of BiLSTM layers
        dropout:         dropout probability

    Usage::

        model = CTCModel(vocab_size=33)
        log_probs = model(batch["mel"])
        input_lens = model.encoder_output_length(batch["mel_lengths"])
        loss = nn.CTCLoss()(log_probs, batch["tokens"], input_lens, batch["token_lengths"])
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        hidden_size: int = 256,
        num_lstm_layers: int = 3,
        dropout: float = 0.1,
        time_reduction: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.time_reduction = time_reduction

        self.encoder = ConvEncoder(
            n_mels=n_mels,
            channels=cnn_channels,
            hidden_size=hidden_size,
            time_reduction=time_reduction,
            dropout=dropout,
        )
        self.rnn = BiLSTMStack(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv, Xavier for linear and LSTM input weights, orthogonal
        for LSTM recurrent weights, zero biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)  # preserves gradient norms through time
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            mel: (B, n_mels, T) log-Mel spectrograms

        Returns:
            (T', B, vocab_size) log-probabilities, time-first for nn.CTCLoss.
        """
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.encoder(x, mel_lengths)
        lengths = self.encoder_output_length(mel_lengths) if mel_lengths is not None else None
        x = self.rnn(x, lengths)
        x = self.classifier(x)  # (B, T//2, vocab_size)
        x = torch.log_softmax(x, dim=-1)
        return x.permute(1, 0, 2)  # (T//2, B, vocab_size) — time-first

    def encoder_output_length(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        """Encoder output frames for each input length (the ``input_lengths`` for nn.CTCLoss).

        Args:
            mel_lengths: (B,) int64, real frames per example

        Returns:
            (B,) int64, encoder frames per example
        """
        return (mel_lengths.to(dtype=torch.long) + self.time_reduction - 1) // self.time_reduction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        param_m = self.count_parameters() / 1e6
        enc_p = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        rnn_p = sum(p.numel() for p in self.rnn.parameters()) / 1e6
        cls_p = sum(p.numel() for p in self.classifier.parameters()) / 1e6
        return (
            f"CTCModel\n"
            f"{'─' * 44}\n"
            f"  Architecture:  ConvEncoder → BiLSTMStack → Linear\n"
            f"  n_mels:        {self.n_mels}\n"
            f"  hidden_size:   {self.hidden_size}\n"
            f"  vocab_size:    {self.vocab_size}\n"
            f"  time_reduce:   {self.time_reduction}×  "
            f"(T mel frames → ceil(T/{self.time_reduction}) encoder frames)\n"
            f"{'─' * 44}\n"
            f"  ConvEncoder    {enc_p:.3f}M parameters\n"
            f"  BiLSTMStack    {rnn_p:.3f}M parameters\n"
            f"  Classifier     {cls_p:.3f}M parameters\n"
            f"  TOTAL          {param_m:.3f}M parameters\n"
            f"{'─' * 44}"
        )
