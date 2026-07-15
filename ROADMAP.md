# LinguaForge Roadmap

LinguaForge is a from-scratch multilingual speech recognition and
linguistic learning engine. "From scratch" is a hard constraint, not a
slogan: no Whisper, no HuggingFace model hub, no pretrained weights of
any kind. PyTorch's *primitives* (`nn.Conv2d`, `nn.LSTM`, `nn.Linear`,
`nn.CTCLoss`, autograd) are used the way you'd use NumPy — the
architectures, training loops, decoders, and linguistic rules on top of
them are all original.

The point of that constraint is the resume story: "I implemented CTC
decoding, a Transformer encoder-decoder, a 106-language detector, and an
SM-2 scheduler" is a meaningfully different claim than "I called the
Whisper API." Every phase below should hold up to that bar.

## Why these reference projects

The table maps each phase onto the open-source project(s) that solve the
same problem at production scale. They're **reference points for what
the component needs to do** — never dependencies, never code to copy.
Studying how a mature project (e.g. pyannote for diarization, MFA for
forced alignment) frames its problem is useful; importing it would
defeat the purpose of the project.

## Try it

`webapp/` is a local browser UI over Phases 5 + 6 — language/dialect
detection, grammar checking, morphology breakdown, and a flashcard deck
with live SM-2 review. Stdlib `http.server` only, no Flask/FastAPI, no
frontend framework or build step:

```bash
.venv/bin/python3 webapp/server.py
# open http://127.0.0.1:8765
```

## Status

| Phase | Component | From-scratch analog of | Status |
|---|---|---|---|
| 1 | Audio preprocessing (resample, mono, normalize, log-mel features) | librosa/torchaudio's preprocessing layer | ✅ done |
| 2 | Tokenizer + dataset pipeline | Whisper/NeMo's data loaders | ✅ done |
| 3 | CTC ASR model (`nn.LSTM` + `nn.CTCLoss`, greedy/beam decode) | whisper.cpp, NeMo, ESPnet, SpeechBrain | ✅ done |
| 4 | Transformer encoder-decoder (attention, positional encoding) | the architecture inside Whisper / seamless_communication | ✅ done |
| 5 | Linguistic engine — language ID (148 langs/26 scripts), dialect detection (190 dialects), BPE morphology, POS tagging, grammar checker | fastText (`lid.176`), HPLT/OpenLID-v3, lingua-py, spaCy/Stanza/Flair | ✅ done — 95.5% lang / 100% dialect accuracy on the test suite |
| 6 | Learning engine — SM-2 spaced repetition, flashcard decks, exercise generation | Anki's scheduler (same SM-2 core) | ✅ done |
| 7 | Machine translation — BPE tokeniser + Transformer seq2seq (reuses Phase 4 encoder-decoder), beam-search decoding, checkpoint save/load, translate tab in web UI | NLLB-200, Marian, Argos Translate, Bergamot | ✅ done — trained on 380k+ es→en pairs (OPUS + curated); IBM2 lexical fallback with LM reranking; reproducible 90/10 test split; BLEU evaluation; NER passthrough; attention visualisation; coverage penalty; back-translation pipeline |
| 8 | Voice activity detection — short-time energy + ZCR + spectral flatness (Wiener entropy) + 4-state FSM (SILENCE→ONSET→SPEECH→HANGOVER); batch + streaming APIs; `/api/vad` endpoint | silero-vad, py-webrtcvad, pyannote-audio | ✅ done — F1=0.94 on synthetic benchmark; numpy only |
| 9 | Forced alignment — Viterbi alignment over the Phase 3 CTC posteriors, for pronunciation feedback | Montreal Forced Aligner, aeneas | ⬜ planned |
| 10 | Structured extraction — dates/numbers/durations from parsed/translated text | Duckling, Microsoft Recognizers-Text, langextract | ⬜ planned, rule-based, no model needed |
| 11 | OCR — CRNN + CTC over scanned text (reuses the Phase 3 loss) | Tesseract, PaddleOCR, EasyOCR, doctr | ⬜ stretch goal |

Not on the roadmap: NER/PII redaction (Presidio), dependency parsing
(spaCy/Stanza's full pipeline), dialogue management (DeepPavlov),
general-purpose LLM serving (`transformers`) — useful prior art if a
future phase needs them, but no current phase requires them.

## Known limitations (tracked honestly, not hidden)

- **Language ID accuracy ceiling on short text.** Da↔nb and mk↔bg are
  exact ties on short sentences that lack each language's distinguishing
  function words — sister languages with near-identical core vocabulary.
  Architectural limit, not a bug; see `speech_engine/language_detector.py`.
- **No support yet for clan/regional dialects within already-low-resource
  languages** (e.g. Somali clan varieties, Dinka's Agar/Rek/Bor/Padang/
  Northeastern dialects, or Nuer/Shilluk at all). The `dialect_of` +
  regex-pattern mechanism used for en-GB/fr-CA would work technically,
  but needs real labeled lexical data per dialect to seed accurately —
  inventing markers without a source risks getting it wrong or
  stereotyping. Needs either contributed text samples or a sourced
  wordlist before it's worth building.
- ~~POS tagger defaults to NOUN for unrecognized open-class words~~ —
  **fixed.** Added AUX closed-class entries (conjugated forms of
  avoir/être, haber/estar/ser, haben/sein/werden, etc.) for fr/es/de/ru/ar,
  and finite-conjugation suffix rules for Spanish preterite/imperfect and
  French past participles (`speech_engine/morphology.py`, `CLOSED_CLASS`
  and `SUFFIX_RULES`). French "a" now tags AUX, Spanish "exploró"/French
  "exploré" now tag VERB.
- ~~English suffix-stripping over-applies in `_extract_en_morphemes`~~ —
  **fixed** via a curated `EN_OPAQUE_WORDS` exception list (Latinate
  "ex-"/"-ous" words with no real free-morpheme split, e.g. "explore",
  "curious") checked before any stripping is attempted. Not a general
  stem-plausibility check — there's no dictionary to validate against —
  so new false positives outside the list will still need to be added by
  hand as they're found.
- **Residual, unfixed ambiguity:** French `-ment` words can be either a
  noun (gouvernement, département) or an adverb (lentement,
  silencieusement) — both commonly end in `-ement`, so no suffix rule
  ordering can get both right without a dictionary. `SUFFIX_RULES`
  currently resolves the NOUN reading first; spotted while fixing the
  above but left alone since it's a genuine two-way ambiguity, not a
  bug — same class of irreducible case as French/Spanish irregular
  adjectives (vieille/vieja) not matching any suffix pattern at all.
- **NMT translation quality** is limited by model size and training data.
  The Phase 7 Transformer is a research-scale model (≈3M parameters); a
  commercial system uses 12B+. BLEU scores in the teens are expected at
  this scale — the architecture is correct and the infrastructure (split,
  BLEU eval, back-translation, coverage penalty) is in place for further
  training runs. `FlashcardDeck.extract_from_text` uses the NMT translator
  for auto-extracted vocabulary cards; quality improves with each training run.
