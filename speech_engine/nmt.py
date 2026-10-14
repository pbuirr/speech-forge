"""Transformer encoder–decoder for text translation.

Shares the attention, feed-forward and decoder blocks with the speech
Transformer; the encoder takes token embeddings with sinusoidal positions
instead of spectrograms. Trained with teacher forcing and label-smoothed
cross-entropy; decodes with greedy or beam search (length penalty, EOS boost
and length limits).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from speech_engine.bpe import BPEVocab, BOS_ID, EOS_ID, PAD_ID
from speech_engine.model_transformer import (
    TransformerDecoder,
    TransformerEncoder,
    make_causal_mask,
    make_padding_mask,
)


class Seq2SeqNMT(nn.Module):
    """Transformer encoder–decoder for translation.

    The encoder takes token embeddings with sinusoidal positions; the decoder is
    the shared ``TransformerDecoder`` with learned positions and cross-attention.

    Args:
        src_vocab_size:  source BPE vocabulary size
        tgt_vocab_size:  target BPE vocabulary size
        d_model:         hidden size (default 256)
        n_heads:         attention heads, must divide d_model (default 4)
        n_layers:        encoder and decoder depth (default 3)
        d_ff:            feed-forward inner size (default 512)
        max_src_len:     maximum source length
        max_tgt_len:     maximum target length
        dropout:         dropout probability (default 0.1)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
        dropout: float = 0.1,
        tie_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Source side: token embedding only — TransformerEncoder adds positional encoding
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.src_embed.weight, std=0.02)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_src_len,
        )

        # Target side: TransformerDecoder owns its own embedding + positional encoding
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_tgt_len,
        )

        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # Weight tying: share target embedding and output projection weights.
        # Reduces parameters and tends to improve translation quality (Press & Wolf 2017).
        self.output_proj.weight = self.decoder.token_emb.weight

        # Full three-way tying for shared-vocab multilingual models (Johnson et al. 2017):
        # src_embed = decoder token_emb = output_proj. Only valid when src_vocab == tgt_vocab.
        if tie_embeddings:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("tie_embeddings requires src_vocab_size == tgt_vocab_size")
            self.src_embed.weight = self.decoder.token_emb.weight

        # Store config for checkpoint serialization
        self._config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_src_len": max_src_len,
            "max_tgt_len": max_tgt_len,
            "dropout": dropout,
            "tie_embeddings": tie_embeddings,
        }

    # Forward (teacher-forced training)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one teacher-forced forward pass.

        Args:
            src_ids:     (B, S) source token IDs
            tgt_ids:     (B, T) target token IDs (decoder input, starts with BOS)
            src_lengths: (B,) real lengths; None infers padding from PAD_ID

        Returns:
            logits: (B, T, tgt_vocab_size)
        """
        B, S = src_ids.shape
        T = tgt_ids.size(1)
        device = src_ids.device

        src_mask = (
            make_padding_mask(src_lengths.to(device), S)
            if src_lengths is not None
            else src_ids.ne(PAD_ID).unsqueeze(1).unsqueeze(2)
        )
        tgt_mask = make_causal_mask(T, device)

        src_emb = self.src_embed(src_ids) * math.sqrt(self.d_model)  # (B, S, d)
        memory = self.encoder(src_emb, src_mask)  # (B, S, d)
        dec_out = self.decoder(tgt_ids, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.output_proj(dec_out)  # (B, T, V_tgt)

    # Inference

    @torch.no_grad()
    def translate_greedy(
        self,
        src_ids: torch.Tensor,
        max_len: int = 100,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
    ) -> list[int]:
        """Greedy decoding: argmax at every step.

        Args:
            src_ids: (1, S) source token IDs (single sentence, batch size 1)
            max_len: maximum number of generated tokens

        Returns:
            list of target token IDs (without BOS, up to and including EOS)
        """
        self.eval()
        device = src_ids.device

        src_emb = self.src_embed(src_ids) * math.sqrt(self.d_model)
        memory = self.encoder(src_emb)

        tgt = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # (1, 1)
        for _ in range(max_len):
            tgt_mask = make_causal_mask(tgt.size(1), device)
            dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.output_proj(dec_out[:, -1, :])  # (1, V_tgt)
            next_id = int(logits.argmax(-1).item())
            tgt = torch.cat(
                [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            if next_id == eos_id:
                break

        return tgt[0, 1:].tolist()  # strip BOS

    @torch.no_grad()
    def translate_beam(
        self,
        src_ids: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 100,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        length_penalty: float = 0.6,
        no_repeat_ngram: int = 3,
        min_len_ratio: float = 0.5,
        coverage_penalty: float = 0.0,
        word_penalty: float = 0.0,
        eos_boost: bool = True,
        repeat_guard: bool = True,
        repeat_exempt_ids: set[int] | None = None,
    ) -> list[int]:
        """Beam search decoding.

        Maintains `beam_size` candidate sequences, expanding the one with the
        highest cumulative log-probability at each step.  Finished beams
        (those that produced EOS) are held in a separate pool and scored with
        a length penalty to discourage short outputs.
        Optional repeat_exempt_ids permits legitimate repeated subword fragments;
        EOS and repetition ablations preserve legacy defaults unless explicitly set.

        Args:
            src_ids:          (1, S) source token IDs
            beam_size:        number of beams to maintain
            max_len:          hard upper limit on generated length
            length_penalty:   power length normalization α — higher values favour
                              longer translations.  0.0 = no penalty.
            coverage_penalty: β for attention coverage penalty (Tu et al. 2016 /
                              Wu et al. 2016 §7).  0.0 = disabled.
                              Penalises beams that attend to source positions
                              fewer than once cumulatively; typical values 0.1–0.4.
            word_penalty:     additive bonus per output token applied to final score
                              (positive = prefer longer, negative = prefer shorter).
                              Useful for calibrating output length independently of
                              length_penalty. Typical range ±0.5.

        Returns:
            token IDs of the best completed beam (BOS excluded)
        """
        self.eval()
        device = src_ids.device
        if repeat_exempt_ids is None:
            repeat_exempt_ids = getattr(self, "_repeat_exempt_ids", set())

        # Cap generation at 2× source length; floor at min_len_ratio * src_len.
        # min_len_ratio prevents the model from emitting EOS early and ignoring
        # large chunks of the source (coverage failure).
        src_len = src_ids.size(1)
        max_len = min(max_len, max(10, int(src_len * 2.0)))
        min_out_len = max(1, int(src_len * min_len_ratio))
        eos_boost_after = max(3, src_len)  # start boosting EOS once output ≈ source length

        src_emb = self.src_embed(src_ids) * math.sqrt(self.d_model)
        memory = self.encoder(src_emb)  # (1, S, d)

        # Expand memory for beam_size parallel decoders
        memory_exp = memory.expand(beam_size, -1, -1)  # (B, S, d)

        use_cov = coverage_penalty > 0.0

        # Each beam: (log-prob, token-id-list, done)
        beams: list[tuple[float, list[int], bool]] = [(0.0, [bos_id], False)]
        # Cumulative cross-attention coverage per source position, one list per beam.
        # cov[i][s] = Σ_t attention_weight(step t → source s) for beam i.
        # Kept parallel to beams[]; empty when coverage_penalty == 0.
        beam_covs: list[list[float]] = [[0.0] * src_len] if use_cov else []
        finished: list[tuple[float, list[int]]] = []

        for step in range(max_len):
            active = [(lp, ids, done) for lp, ids, done in beams if not done]
            if not active:
                break
            # Track which slot in beams[] each active entry came from, so we
            # can look up its parent coverage vector when expanding candidates.
            active_beam_idx = [i for i, (_, _, done) in enumerate(beams) if not done]

            # Stack active beams into one batch
            n = len(active)
            tgt = torch.tensor(
                [ids for _, ids, _ in active], dtype=torch.long, device=device
            )  # (n, step+1)

            tgt_mask = make_causal_mask(tgt.size(1), device)
            # Use only the first n slices of expanded memory
            if use_cov:
                # Request per-step attention from the last decoder layer (averaged heads).
                # step_attn: (n, T_out, src_len) — row T_out-1 is the current step.
                dec_out, step_attn = self.decoder(
                    tgt, memory_exp[:n], tgt_mask=tgt_mask, return_cross_attn=True
                )
                last_attn = step_attn[:, -1, :].tolist()  # [(src_len,)] × n
            else:
                dec_out = self.decoder(tgt, memory_exp[:n], tgt_mask=tgt_mask)
                last_attn = None

            logits = self.output_proj(dec_out[:, -1, :])  # (n, V)
            log_probs = torch.log_softmax(logits, dim=-1)  # (n, V)

            # Block EOS until min_len_ratio * src_len tokens have been generated.
            # This implements a soft coverage constraint: the model must produce
            # at least ~half as many tokens as the source before it can stop.
            if step < min_out_len:
                log_probs[:, eos_id] = -1e9

            # Progressive EOS boost: +6 at expected length, growing by +3 per step beyond
            elif eos_boost and step >= eos_boost_after:
                boost = 6.0 + (step - eos_boost_after) * 3.0
                log_probs[:, eos_id] += boost

            # Expand each active beam into top-V candidates, keep best beam_size.
            # Each candidate carries (lp, ids, done, new_cov) where new_cov is the
            # updated cumulative coverage after appending this step's attention.
            candidates: list[tuple[float, list[int], bool, list[float] | None]] = []
            for i, (lp, ids, _) in enumerate(active):
                parent_idx = active_beam_idx[i]
                # Build the set of token ids banned by n-gram blocking + repeat guard
                banned: set[int] = set()
                if no_repeat_ngram > 1 and len(ids) >= no_repeat_ngram - 1:
                    ngram_prefix = tuple(ids[-(no_repeat_ngram - 1) :])
                    for start in range(len(ids) - (no_repeat_ngram - 1)):
                        if tuple(ids[start : start + no_repeat_ngram - 1]) == ngram_prefix:
                            if start + no_repeat_ngram - 1 < len(ids):
                                banned.add(ids[start + no_repeat_ngram - 1])
                # Ban any non-special token appearing ≥2× in the last 6 positions
                recent = ids[-6:] if len(ids) >= 6 else ids
                for tok_id in set(recent):
                    if (
                        repeat_guard
                        and tok_id not in (bos_id, eos_id)
                        and recent.count(tok_id) >= 2
                    ):
                        banned.add(tok_id)
                # Ban immediate repetition of last token (prevents BPE fragment doubling: "c c at" → "ccat")
                if repeat_guard and len(ids) > 1 and ids[-1] not in (bos_id, eos_id):
                    banned.add(ids[-1])

                banned.difference_update(repeat_exempt_ids or ())
                top_lps, top_ids = log_probs[i].topk(
                    min(log_probs.size(-1), beam_size + len(banned))
                )
                added = 0
                for j in range(len(top_ids)):
                    tok = int(top_ids[j].item())
                    if tok in banned:
                        continue
                    new_lp = lp + float(top_lps[j].item())
                    done = tok == eos_id
                    new_ids = ids + [tok]
                    if use_cov:
                        # All siblings from the same parent share the same parent
                        # coverage and the same attention at this step — they diverge
                        # only from the next step onward.
                        parent_cov = beam_covs[parent_idx]
                        attn_step = last_attn[i]  # type: ignore[index]
                        new_cov: list[float] | None = [
                            parent_cov[s] + attn_step[s] for s in range(src_len)
                        ]
                    else:
                        new_cov = None
                    candidates.append((new_lp, new_ids, done, new_cov))
                    added += 1
                    if added >= beam_size:
                        break

            # Re-rank by length-normalised log-prob + optional coverage penalty.
            # Coverage penalty (Wu et al. 2016):
            #   cp = β × Σ_s log(min(Σ_t α_{t,s}, 1.0))
            # → positive contribution that shrinks toward 0 as coverage improves;
            #   beams that neglect source positions are penalised more.
            def _score(c: tuple[float, list[int], bool, list[float] | None]) -> float:
                lp, ids, _, cov = c
                n_tokens = max(1, len(ids) - 1)  # exclude BOS
                lp_norm = lp / ((n_tokens**length_penalty) or 1)
                lp_norm += word_penalty * n_tokens
                if use_cov and cov is not None:
                    cp = coverage_penalty * sum(
                        math.log(min(cov[s], 1.0)) if cov[s] > 1e-9 else -5.0
                        for s in range(src_len)
                    )
                    return lp_norm + cp
                return lp_norm

            candidates.sort(key=_score, reverse=True)
            top = candidates[:beam_size]

            # Unpack top candidates back into parallel lists
            beams = [(lp, ids, done) for lp, ids, done, _ in top]
            if use_cov:
                beam_covs = [cov for _, _, _, cov in top]  # type: ignore[misc]

            # Move finished beams to pool
            still_active: list[tuple[float, list[int], bool]] = []
            still_covs: list[list[float]] = []
            for idx, b in enumerate(beams):
                if b[2]:  # done
                    cov_b = beam_covs[idx] if use_cov else None
                    finished.append((_score((b[0], b[1], b[2], cov_b)), b[1]))
                else:
                    still_active.append(b)
                    if use_cov:
                        still_covs.append(beam_covs[idx])
            if still_active:
                beams = still_active
                if use_cov:
                    beam_covs = still_covs
            # else: all finished — keep beams as-is; next iteration will have
            # active=[] and we'll break immediately.

            if len(finished) >= beam_size:
                break

        if finished:
            finished.sort(key=lambda x: x[0], reverse=True)
            best = finished[0][1]
        else:
            best = max(beams, key=lambda b: b[0])[1]

        # Strip BOS (and EOS if present)
        out = best[1:]
        if out and out[-1] == eos_id:
            out = out[:-1]
        return out

    def translate(
        self,
        src_ids: torch.Tensor,
        beam_size: int = 1,
        max_len: int = 100,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        word_penalty: float = 0.0,
        coverage_penalty: float = 0.0,
    ) -> list[int]:
        """Translate, using greedy (beam_size=1) or beam search (beam_size>1)."""
        if beam_size <= 1:
            return self.translate_greedy(src_ids, max_len=max_len, bos_id=bos_id, eos_id=eos_id)
        return self.translate_beam(
            src_ids,
            beam_size=beam_size,
            max_len=max_len,
            bos_id=bos_id,
            eos_id=eos_id,
            word_penalty=word_penalty,
            coverage_penalty=coverage_penalty,
        )

    @torch.no_grad()
    def translate_with_attention(
        self,
        src_ids: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 100,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        **decoding_options,
    ) -> dict:
        """
        Translate and return the cross-attention matrix + per-token confidence.

        After beam search produces the best output sequence, we do one clean
        teacher-forced forward pass with `return_cross_attn=True` to get the
        full (T_out × T_src) attention matrix without the incremental complexity
        of tracking it inside the beam loop.

        Returns a dict with:
            out_ids:     list[int] — translated token IDs
            attention:   list[list[float]] — (T_out, T_src) attention matrix
                         row i sums to ~1.0 (softmax over source positions)
            entropy:     list[float] — per-output-token attention entropy
                         0 = very confident (peaked), 1 = very uncertain (flat)
        """
        self.eval()
        device = src_ids.device

        # 1. Beam search to get the best output sequence
        out_ids = self.translate_beam(
            src_ids,
            beam_size=beam_size,
            max_len=max_len,
            bos_id=bos_id,
            eos_id=eos_id,
            **decoding_options,
        )
        if not out_ids:
            return {"out_ids": [], "attention": [], "entropy": []}

        # 2. One teacher-forced forward pass with the known output to get clean attention
        src_emb = self.src_embed(src_ids) * math.sqrt(self.d_model)
        memory = self.encoder(src_emb)  # (1, S, d)

        tgt = torch.tensor([[bos_id] + out_ids], dtype=torch.long, device=device)
        T_out = tgt.size(1)
        tgt_mask = make_causal_mask(T_out, device)

        dec_out, avg_attn = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, return_cross_attn=True
        )  # avg_attn: (1, T_out, T_src)

        # Drop the BOS position — first row attends to source for the first output token
        # avg_attn shape: (1, T_out, T_src); out_ids has T_out - 1 tokens (no BOS)
        attn_matrix = avg_attn[0, :-1, :].cpu().tolist()  # (T_out-1, T_src)

        # Per-token entropy: H = -sum_s(a_s * log(a_s)), normalised by log(T_src)
        import math as _math

        T_src = avg_attn.size(-1)
        log_norm = _math.log(max(T_src, 2))
        entropy = []
        for row in attn_matrix:
            h = -sum(p * _math.log(max(p, 1e-9)) for p in row) / log_norm
            entropy.append(round(min(h, 1.0), 4))

        return {
            "out_ids": out_ids,
            "attention": [[round(v, 4) for v in row] for row in attn_matrix],
            "entropy": entropy,
        }

    # Parameter count helper

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training helpers


def nmt_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int = PAD_ID,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Label-smoothed cross-entropy over non-padding target positions.

    Smoothing spreads `smoothing` probability mass uniformly across the
    vocabulary instead of placing 1.0 on the correct token.  This prevents
    the model from becoming overconfident on training patterns and
    may improve generalisation; its effect requires held-out evaluation.

    Args:
        logits:     (B, T, V_tgt) — raw model outputs
        target_ids: (B, T) — shifted right target (first token is t1, last is EOS)
        smoothing:  label-smoothing epsilon (0 = hard cross-entropy)

    Returns:
        scalar mean loss
    """
    B, T, V = logits.shape
    log_probs = nn.functional.log_softmax(logits.reshape(B * T, V), dim=-1)
    flat_targets = target_ids.reshape(B * T)

    with torch.no_grad():
        # Uniform smooth distribution: (smoothing / V-1) everywhere, (1 - smoothing) on gold
        smooth_dist = torch.full_like(log_probs, smoothing / max(1, V - 1))
        smooth_dist.scatter_(1, flat_targets.unsqueeze(1).clamp(min=0), 1.0 - smoothing)
        smooth_dist[:, pad_id] = 0.0  # never assign mass to PAD

    loss = -(smooth_dist * log_probs).sum(dim=-1)
    mask = flat_targets != pad_id
    return loss[mask].mean()


def collate_pairs(
    pairs: list[tuple[list[int], list[int]]],
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (src_ids, tgt_ids) pairs into tensors.

    Returns:
        src:       (B, S) padded source IDs
        src_lens:  (B,)   real source lengths
        tgt_in:    (B, T) decoder input  — [BOS, t1, …, tn]
        tgt_out:   (B, T) decoder target — [t1, …, tn, EOS]
        tgt_lens:  (B,)   real target lengths (excluding BOS)
    """
    src_lens = torch.tensor([len(s) for s, _ in pairs])
    tgt_lens = torch.tensor([len(t) - 1 for _, t in pairs])  # -1: exclude BOS or EOS

    max_s = int(src_lens.max())
    max_t = max(len(t) for _, t in pairs)

    src = torch.full((len(pairs), max_s), pad_id, dtype=torch.long)
    tgt_in = torch.full((len(pairs), max_t - 1), pad_id, dtype=torch.long)
    tgt_out = torch.full((len(pairs), max_t - 1), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(pairs):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_in[i, : len(t) - 1] = torch.tensor(t[:-1], dtype=torch.long)  # BOS … tn
        tgt_out[i, : len(t) - 1] = torch.tensor(t[1:], dtype=torch.long)  # t1 … EOS

    return src, src_lens, tgt_in, tgt_out, tgt_lens


# Checkpoint helpers


def save_checkpoint(
    model: Seq2SeqNMT,
    src_vocab: BPEVocab,
    tgt_vocab: BPEVocab,
    path: Path | str,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_vocab": {"token_to_id": src_vocab._token_to_id, "merges": src_vocab._merges},
            "tgt_vocab": {"token_to_id": tgt_vocab._token_to_id, "merges": tgt_vocab._merges},
            "config": model._config,
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[Seq2SeqNMT, BPEVocab, BPEVocab, dict]:
    """Load a saved NMT checkpoint.

    Returns:
        model, src_vocab, tgt_vocab, meta_dict
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    src_vocab = BPEVocab()
    src_vocab._token_to_id = ckpt["src_vocab"]["token_to_id"]
    src_vocab._id_to_token = {int(v): k for k, v in src_vocab._token_to_id.items()}
    src_vocab._merges = [tuple(m) for m in ckpt["src_vocab"]["merges"]]
    src_vocab._merge_rank = {tuple(m): i for i, m in enumerate(src_vocab._merges)}

    tgt_vocab = BPEVocab()
    tgt_vocab._token_to_id = ckpt["tgt_vocab"]["token_to_id"]
    tgt_vocab._id_to_token = {int(v): k for k, v in tgt_vocab._token_to_id.items()}
    tgt_vocab._merges = [tuple(m) for m in ckpt["tgt_vocab"]["merges"]]
    tgt_vocab._merge_rank = {tuple(m): i for i, m in enumerate(tgt_vocab._merges)}

    cfg = ckpt["config"]
    model = Seq2SeqNMT(
        src_vocab_size=cfg["src_vocab_size"],
        tgt_vocab_size=cfg["tgt_vocab_size"],
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 3),
        d_ff=cfg.get("d_ff", 512),
        max_src_len=cfg.get("max_src_len", 256),
        max_tgt_len=cfg.get("max_tgt_len", 256),
        dropout=cfg.get("dropout", 0.1),
        tie_embeddings=cfg.get("tie_embeddings", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metadata = ckpt.get("meta", {})
    if metadata.get("decoding_policy") == "preserve_numeric_fragments":
        model._repeat_exempt_ids = {
            i for i, token in tgt_vocab._id_to_token.items() if token.removesuffix("</w>").isdigit()
        }
    return model, src_vocab, tgt_vocab, metadata
