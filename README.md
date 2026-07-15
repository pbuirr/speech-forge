# LinguaForge

A from-scratch multilingual NLP engine — speech recognition, machine translation, linguistic analysis, and spaced-repetition learning — built entirely in PyTorch without pretrained weights or HuggingFace.

> **Hard constraint:** no Whisper, no HuggingFace model hub, no pretrained weights of any kind. PyTorch primitives (`nn.Linear`, `nn.LSTM`, `nn.MultiheadAttention`, autograd) are used the way you'd use NumPy. Every architecture, training loop, decoder, and linguistic rule is original.

---

## What's implemented

| Phase | Component | Highlights |
|---|---|---|
| 1 | Audio preprocessing | Resample → mono → normalize → log-mel spectrogram; matches torchaudio output |
| 2 | Tokenizer + dataset pipeline | BPE from scratch (Sennrich et al. 2016); sliding-window chunker; no sentencepiece |
| 3 | CTC speech recognition | BiLSTM acoustic model + CTC loss; greedy and beam-search decoder |
| 4 | Transformer encoder-decoder | Multi-head attention, positional encoding, label-smoothing cross-entropy |
| 5 | Linguistic engine | Language ID (148 langs / 26 scripts, **95.5% accuracy**); dialect detection (190 dialects); BPE morphology; POS tagger; grammar checker |
| 6 | Spaced-repetition learning | SM-2 scheduler (Anki-compatible); flashcard deck; exercise generator |
| 7 | Neural machine translation | Seq2seq Transformer; BPE tokenizer; beam search with coverage penalty + word penalty; IBM Model 2 + trigram LM fallback; back-translation; **27.7 BLEU on 38k ES→EN test pairs** |
| 8 | Voice activity detection | Short-time energy + ZCR + spectral flatness + 4-state FSM; **F1 = 0.94**; numpy only |

---

## Quick start

```bash
# 1. Install PyTorch for your platform first
#    Mac (MPS):  pip install torch torchaudio
#    CUDA:       pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#    CPU only:   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install the package
pip install -e .

# 3. Launch the web UI (language detection, grammar, morphology, translation, flashcards)
python3 webapp/server.py
# → open http://127.0.0.1:8765
```

---

## Translation demo

```python
from speech_engine.nmt import load_checkpoint

model, src_vocab, tgt_vocab, _ = load_checkpoint("data/models/nmt_es_en.pt")
src = src_vocab.encode("el gato exploró la casa", add_bos=True, add_eos=True)

import torch
src_t = torch.tensor([src])
ids   = model.translate_beam(src_t, beam_size=4)
print(tgt_vocab.decode(ids))
# → "the cat explored the house"
```

---

## Language detection demo

```python
from speech_engine.language_detector import LanguageDetector

ld = LanguageDetector()
print(ld.detect("el gato exploró la casa"))   # → ('es', 0.998)
print(ld.detect("le chat a exploré la maison")) # → ('fr', 0.997)
print(ld.detect("die Katze erkundete das Haus")) # → ('de', 0.995)
```

---

## Training your own NMT model

```bash
# Download OPUS corpora (ES/FR/DE → EN)
python3 scripts/download_real_corpus.py

# Filter for quality
python3 scripts/filter_corpus.py \
    --input data/corpora/es_en_real.tsv \
    --output data/corpora/es_en_filtered.tsv

# Train ES → EN
python3 scripts/phase7_train.py \
    --epochs 150 --batch 128 --accum 4 \
    --vocab 4000 --max_pairs 100000 \
    --out data/models/nmt_es_en.pt

# Evaluate BLEU
python3 evaluate_nmt.py \
    --checkpoint data/models/nmt_es_en.pt \
    --beam 4 --n 200
```

---

## Repository layout

```
speech_engine/      Core implementations (BPE, NMT, language detector, VAD, …)
scripts/            Training scripts for each phase
webapp/             Local browser UI — no framework, stdlib http.server only
data/
  samples/          Small audio samples and spectrograms
  toy_dataset/      Minimal WAV dataset for smoke-testing
results/            BLEU reports and attention heatmaps
docs/               Per-phase technical notes
```

---

## Key design decisions

**Why no pretrained weights?** The goal is to be able to explain every component from first principles. "I implemented BPE tokenization, CTC decoding, and a Seq2Seq Transformer with beam search" is a different claim than "I called the Whisper API."

**Why IBM Model 2 as a fallback?** Classic statistical MT provides a robust, interpretable fallback when the neural model produces low-confidence output. The diagonal alignment prior is a direct implementation of the Brown et al. (1993) / Och & Ney (2003) approach.

**Why a shared BPE vocabulary for multilingual?** Latin-script languages (ES/FR/DE) share large portions of their subword inventory. A shared 16k-token BPE vocab covers all three with only ~10–15% more tokens than a monolingual 4k vocab, enabling knowledge transfer across language pairs (Johnson et al. 2017).

---

## References

- Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units
- Vaswani et al. (2017) — Attention Is All You Need
- Provilkov et al. (2020) — BPE-Dropout: Simple and Effective Subword Regularization
- Loshchilov & Hutter (2019) — Decoupled Weight Decay Regularization (AdamW)
- Platanios et al. (2019) — Competence-based Curriculum Learning
- Johnson et al. (2017) — Google's Multilingual Neural Machine Translation System
- Brown et al. (1993) — The Mathematics of Statistical Machine Translation

---

See [`ROADMAP.md`](ROADMAP.md) for the full technical roadmap and known limitations.
