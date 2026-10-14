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
| 5 | Linguistic engine | Language/dialect ID; BPE subwords; corpus-backed lemmas/features and contextual POS tagging; heuristic fallback; grammar checker |
| 6 | Spaced-repetition learning | SM-2 scheduler (Anki-compatible); flashcard deck; exercise generator |
| 7 | Neural machine translation | Original Transformer/BPE and statistical fallback; current ES→EN model: **29.92 BLEU on 1,993 short sentences, 14.71 on 400 reserved news article sentences, 9.37 on 400 fresh separate-publisher sentences**; experimental, with a short-sentence regression (custom evaluation) |
| 8 | Voice activity detection | Short-time energy + ZCR + spectral flatness + 4-state FSM; **F1 = 0.94**; numpy only |

---

## Quick start

The locally trained Spanish→English checkpoint is installed on this development
machine and verified through the browser. It remains experimental, especially on
longer text. Training, independent evaluation, limitations and reproduction commands
are recorded in [Real model validation](docs/REAL_MODEL_VALIDATION.md).
The latest [domain adaptation and speech report](docs/DOMAIN_ADAPTATION.md) records
news improvements, short-sentence tradeoffs and an acoustic pilot whose WER remains
too high to enable in the default application.
Checkpoints and downloaded corpora are ignored by Git; a fresh clone requires training
or restoring your own checkpoint. Real local speech recognition still needs trained weights.

For the local web application's CTC checkpoint configuration, WAV requirements,
and verified limitations, see [Local model integration](docs/LOCAL_MODEL_INTEGRATION.md).
Implemented architectures require trained weights; their presence in this repository
does not by itself establish usable speech or translation accuracy.

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
python3 scripts/translation/download_parallel_corpus.py

# Filter for quality
python3 scripts/translation/filter_corpus.py \
    --input data/corpora/es_en_real.tsv \
    --output data/corpora/es_en_filtered.tsv

# Train ES → EN
python3 scripts/translation/train_nmt.py \
    --epochs 150 --batch 128 --accum 4 \
    --vocab 4000 --max_pairs 100000 \
    --out data/models/nmt_es_en.pt

# Evaluate BLEU
python3 scripts/translation/evaluate_nmt.py \
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

See [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) for the current direction,
inspected limitations, and development preferences. Recorded quality scores above
are not independently reproduced by the regression suite.

## Regression checks

```bash
python -m pytest -q
node tests/test_web_rendering.cjs
```

These cover morphology/subword separation, resource import and validation,
contextual predictions, Unicode offsets, language isolation, HTTP routing,
flashcard evidence/persistence, and rendering. Corpus observations, context
predictions, and heuristics remain distinct; uncalibrated confidence is null.

The UI uses `/api/analyze` to coordinate language detection, word analysis, and
grammar feedback using the same text, language selection, and word spans. Grammar
findings link to the affected tokens; existing individual endpoints remain usable.
Flashcards can reuse that analysis rather than repeat inference. No rule findings
does not prove grammatical correctness. This shared pipeline does not yet perform
joint multilingual learning; corpus-based parsing is evaluated on English and Turkish.

Learning exercises reuse word spans and Unicode-aware matching across languages.
Duplicate/cross-language distractors and unbounded shuffling have been corrected.
Failed grammar checks remain visible and cannot produce a clean result or score.
See [learning evidence](docs/LEARNING_EVIDENCE.md) for reproducible cases and limits.

`python scripts/analysis/prepare_english_resource.py --calibrate` also fits the project's
local POS reliability model on development data, using the original tagger's
forward–backward scores. The UI and saved cards expose POS-specific review hints.
See [POS reliability](docs/POS_RELIABILITY.md) for measurements and limits; these
estimates do not certify lemmas or meaning.

Prepare the optional original count-based English model and reproduce its held-out
evaluation with `python scripts/analysis/prepare_english_resource.py`. The app loads the
compiled local resource on restart. See [linguistic resources](docs/LINGUISTIC_RESOURCES.md)
for sources, licensing, metrics, and limits. The measured English POS accuracy is
90.2% versus 68.5% for heuristics on gold-tokenized EWT test data; this does not
establish multilingual accuracy, semantic understanding, or speech/NMT quality.

See [`ROADMAP.md`](ROADMAP.md) for the earlier technical roadmap.

Experimental original dependency parsing is available as an optional locally trained linguistic resource. See [method, evaluation and reproduction](docs/DEPENDENCY_ANALYSIS.md). It reuses shared POS/features and source offsets; syntax predictions are uncalibrated and do not establish meaning.

Latest local translation quality, decoding/data comparisons and remaining semantic failures: [translation reliability](docs/TRANSLATION_RELIABILITY.md).

Latest source-meaning experiments, real CTC progress and application integration: [meaning and speech](docs/MEANING_AND_SPEECH.md).
## Measured speech quality

The [speech capacity and supervision report](docs/SPEECH_SCALE.md) records the
larger licensed-audio comparison, checkpoint compatibility and generated-text
language-routing investigation.

The [diverse-supervision and connected-evidence report](docs/SPEECH_DIVERSITY.md)
compares matched acoustic training with new dialect recordings and documents how
generated-input uncertainty carries into analysis, translation and learning.

The original local CTC remains experimental. The latest speaker-disjoint evaluation
and locally trained decoder comparison are in [SPEECH_QUALITY.md](docs/SPEECH_QUALITY.md).
Better aggregate error rates do not make transcripts authoritative learning answers.
