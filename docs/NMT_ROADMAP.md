# LinguaForge NMT — Technical Roadmap

Phase 7 of LinguaForge is a full neural machine translation pipeline built
entirely from scratch — no HuggingFace, no pretrained weights, no fairseq.

---

## Current Status

| Capability | Status |
|---|---|
| Transformer seq2seq (BPE tokeniser, encoder-decoder, beam search) | ✅ done |
| IBM Model 2 lexical fallback with diagonal alignment prior | ✅ done |
| Stupid-backoff trigram LM for reranking (word sense disambiguation) | ✅ done |
| Reproducible 90/10 train/test split (seed=42, persisted to disk) | ✅ done |
| Corpus-level BLEU evaluation (implemented from scratch) | ✅ done |
| Standalone eval script with attention heatmap + JSON report | ✅ done |
| Back-translation pipeline (Sennrich et al. 2016) | ✅ done |
| Named-entity passthrough (NER masking + restore) | ✅ done |
| Attention coverage penalty (Wu et al. 2016 / Tu et al. 2016) | ✅ done |
| Full training run on 380k+ OPUS pairs | ⏳ in progress (PID 9447) |

---

## How to Run Everything

### 1. Train from scratch

```bash
# Full training — 200 epochs, up to 100k pairs, gradient accum × 4
.venv/bin/python3 scripts/phase7_train.py

# Fast smoke-test (50 epochs, small model, 10k pairs)
.venv/bin/python3 scripts/phase7_train.py --fast

# Custom settings
.venv/bin/python3 scripts/phase7_train.py \
    --epochs 300 --batch 128 --accum 4 \
    --max_pairs 100000 --save_every 25 \
    --out data/models/nmt_es_en.pt
```

Training automatically:
- Creates/loads a reproducible 90/10 train/test split (`data/splits/es_en_test_seed42.tsv`)
- Saves intermediate checkpoints every `--save_every` epochs
- Runs BLEU evaluation on the held-out test set at the end
- Saves a JSON report to `results/eval_es_en.json`

### 2. Evaluate a checkpoint

```bash
# Evaluate against the held-out test split
python evaluate_nmt.py --src es --tgt en \
    --checkpoint data/models/nmt_es_en.pt

# With attention heatmap for a specific sentence
python evaluate_nmt.py --src es --tgt en \
    --checkpoint data/models/nmt_es_en.pt \
    --attention "el gato curioso exploró la casa vieja"

# With named-entity protection
python evaluate_nmt.py --src es --tgt en \
    --checkpoint data/models/nmt_es_en.pt \
    --protect-entities

# Evaluate faster on a subset
python evaluate_nmt.py --src es --tgt en \
    --checkpoint data/models/nmt_es_en.pt \
    --max-test 500
```

Report saved to: `results/eval_es_en.json`
Attention heatmap: `results/attention_es_en.png`

### 3. Translate a sentence

```bash
.venv/bin/python3 scripts/phase7_train.py \
    --translate "me gusta el mundo" \
    --load data/models/nmt_es_en.pt
```

Or from Python:
```python
from speech_engine.nmt import load_checkpoint
model, src_vocab, tgt_vocab, _ = load_checkpoint("data/models/nmt_es_en.pt")
src_ids = src_vocab.encode("me gusta el mundo")
out_ids = model.translate(torch.tensor([src_ids]), beam_size=4)
print(tgt_vocab.decode(out_ids))
```

### 4. Back-translation (synthetic data augmentation)

```bash
# Extract monolingual English from the real corpus
cut -f2 data/corpora/es_en_real.tsv > data/monolingual_en.txt

# Train a reverse EN→ES model first (flip the corpus columns)
# (requires manually creating an en_es corpus TSV)

# Generate synthetic ES←EN pairs
python backtranslate.py \
    --src en --tgt es \
    --input data/monolingual_en.txt \
    --output data/synthetic_en_es.tsv \
    --checkpoint data/models/nmt_en_es.pt \
    --limit 100000

# Merge synthetic with real data
cat data/corpora/es_en_real.tsv data/synthetic_en_es.tsv > data/combined_es_en.tsv
```

### 5. Coverage penalty (beam search)

The coverage penalty (Wu et al. 2016) penalises beam hypotheses that
ignore source positions, reducing undertranslation. Enable it at
inference time — no retraining needed:

```python
out_ids = model.translate_beam(
    src_ids,
    beam_size=4,
    coverage_penalty=0.2,   # β in [0.1, 0.4] is typical
)
```

Higher β → more pressure to attend to all source positions.
Default is 0.0 (disabled) so existing callers are unaffected.

---

## Architecture Overview

```
Source sentence (ES)
        │
    BPE tokenise  (speech_engine/bpe.py, 4000 vocab)
        │
  [ BOS, tok1, tok2, …, EOS ]
        │
 src_embed (nn.Embedding) + sinusoidal pos encoding
        │
 TransformerEncoder  (3 layers × 256-dim × 4 heads)
        │
      memory (B, S, 256)
        │
  cross-attention ────┐
        │             │
 TransformerDecoder   │  (3 layers, learned pos encoding)
 (B, T, 256)  ←──────┘
        │
  output_proj (256 → tgt_vocab_size)  [weights tied to tgt_embed]
        │
 Target BPE tokens (EN)
```

**BPE vocabulary:** 4000 tokens, trained jointly on source+target text.

**Weight tying:** `output_proj.weight = decoder.token_emb.weight`
(Press & Wolf 2017) — reduces parameters and improves quality.

**Beam search features:**
- Length penalty (Wu et al. α=0.6)
- Min-length ratio (prevents premature EOS, default 0.5 × src_len)
- n-gram blocking (no 3-gram repetition)
- Repeat guard (no token twice in last 6 positions)
- Progressive EOS boost (encourages completion near expected length)
- Attention coverage penalty (Tu et al. 2016, β configurable)

**Lexical fallback (IBM Model 2 + LM):**
When the NMT model is unavailable or for low-level word lookup,
`speech_engine/lexical_fallback.py` uses IBM Model 2 alignment
with a diagonal prior (fast_align approach) plus a stupid-backoff
trigram LM for context-sensitive word sense disambiguation.

---

## BLEU Scores — What to Expect

| Training pairs | Epochs | Expected BLEU (ES→EN) |
|---|---|---|
| 600 curated | 600 | ~8–12 (memorisation only) |
| 10k | 50 (fast) | ~2–5 (smoke test, underfitting) |
| 100k | 200 | ~12–20 (target range with this model size) |
| 380k | 200 | ~15–25 (with enough compute) |

Context: newstest2013 EN→DE BLEU scores by system size
- Edinburgh SMT (2014): 20.7
- Small Transformer (6M params): ~22 on WMT14 EN→DE
- This model (≈3M params, much smaller data): 12–20 is realistic

BLEU improvements beyond 20 require:
- More training data (1M+ pairs)
- Larger model (d_model=512, 6 layers)
- Label smoothing (already at 0.1)
- Back-translation augmentation
- Ensemble decoding

---

## Next Steps (Research-Level)

### Near-term (high ROI)
1. **More training data** — download CCAligned or OPUS-100 for additional
   ES→EN pairs. Back-translation from the English side adds more without
   new human labels.
2. **Larger model** — scale to `d_model=512, n_layers=6, n_heads=8`.
   Expected +3–5 BLEU at the same data size.
3. **More languages** — the architecture is language-agnostic; adding FR→EN
   or DE→EN requires only a new corpus TSV and a separate training run.

### Medium-term
4. **Subword regularisation** — BPE dropout during training (Provilkov et al.
   2020): randomly merge/split BPE units to improve robustness to segmentation.
5. **Multi-source ensemble** — average logits from 3–5 independently-trained
   checkpoints at inference time. Usually +1–2 BLEU with zero extra training.
6. **Relative position encoding** — replace sinusoidal encoding with Shaw et al.
   (2018) relative positions for better generalisation to long sentences.

### Research-level
7. **Multilingual single model** — prefix source sentences with `<ES>`, `<FR>`
   etc. and train one model over all language pairs simultaneously (m-BART style).
   Enables zero-shot transfer to unseen pairs.
8. **Distillation** — use the beam search output of a large model as training
   targets for a smaller, faster model (Kim & Rush 2016 seq-level distillation).
9. **Non-autoregressive decoding** — predict all target tokens in one forward
   pass (Gu et al. 2018 NAT), 4–10× faster at ~1 BLEU cost.
10. **Minimum Bayes Risk decoding** — instead of returning the highest-scoring
    beam, return the hypothesis with the highest expected BLEU against other
    beams (Kumar & Byrne 2004). Free +0.5–1 BLEU on top of beam search.

---

## Key References

| Paper | What it contributes |
|---|---|
| Vaswani et al. 2017, "Attention Is All You Need" | The Transformer architecture |
| Wu et al. 2016, "Google's NMT System" | Length penalty, coverage penalty |
| Tu et al. 2016, "Modeling Coverage for NMT" | Attention coverage penalty |
| Sennrich et al. 2016 (BPE) | Byte-pair encoding for open vocabulary |
| Sennrich et al. 2016 (back-translation) | Back-translation for data augmentation |
| Press & Wolf 2017 | Weight tying (output proj ↔ target embedding) |
| Papineni et al. 2002 | BLEU metric |
| Lin & Och 2004 | Sentence-level BLEU smoothing |
| Dyer et al. 2013 (fast_align) | Diagonal prior for IBM Model 2 |
| Brants et al. 2007 | Stupid-backoff LM |
