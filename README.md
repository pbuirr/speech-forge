# LinguaForge

LinguaForge is a language-learning and language-analysis system I'm building from scratch. Every model in it is my own: I wrote the architectures and training code and trained the weights locally on public, licensed datasets. No pretrained weights, no hosted APIs.

The goal is a tool that helps you learn a language from real material (videos, articles, books) by showing what each word is, how the sentence is built and what it means, then turning what you meet into practice.

## What works today

| Component | What it does | Best measured result |
| --- | --- | --- |
| Speech recognition (English) | CNN + BiLSTM acoustic model trained with CTC, character language model for decoding | **22.5% WER, 7.9% CER** on 160 held-out LibriSpeech test-clean recordings from 40 unseen speakers |
| Translation (Spanish → English) | Transformer encoder–decoder with BPE subwords | **27.7 BLEU** on clean news sentences with an older, larger model I trained (the currently installed model scores 16.2) |
| Word analysis (English, Turkish) | Corpus-backed lemmas and features, HMM part-of-speech tagger with calibrated confidence | **90.2%** POS accuracy on UD English-EWT test (gold tokens) |
| Dependency parsing (English, Turkish) | Transition-based parser trained on Universal Dependencies | **72.5 LAS** English-EWT, **45.3 LAS** Turkish-BOUN |
| Grammar checks | Pattern rules, measured for false alarms on published text | 184 error-level flags on 29,211 published sentences (down from 347) |
| Learning | Flashcards with sentence context, spaced repetition (SM-2), generated exercises | Not yet evaluated with learners |

All numbers come from data the models never trained on or were tuned on. Full history: [RESULTS.md](RESULTS.md).

## How it fits together

```
 audio ──► speech recognizer ──┐
                               ├──► one text record (offsets, language, where it came from)
 typed or pasted text ─────────┘            │
                                            ├──► analysis: tokens, lemmas, POS, syntax, grammar checks
                                            ├──► translation
                                            └──► learning: flashcards and exercises built from the analysis
```

Anything a model generated (a transcript, a translation) is marked as a prediction all the way through, so it can't silently become an answer key for a flashcard.

More detail on each part:

- [Speech recognition](docs/speech-recognition.md)
- [Translation](docs/translation.md)
- [Linguistic analysis](docs/linguistic-analysis.md)
- [Learning features and the web app](docs/learning-and-web-app.md)

## What I learned building it

A few results changed how I worked:

- **My speech model wasn't too small. It was barely trained.** Development loss was still falling when the learning-rate schedule ended (about 1.8 passes over the data). Training the same model four times longer cut test WER from 47.6% to 30.3%. Adding 360 more hours of audio and a slightly larger model brought it to 22.5%.
- **My translation test set was contaminated for an older model.** Before comparing against an older model I'd trained, I checked overlap between its training data and my test sets. It had seen 99.5% of my short-sentence test, so I only compared on news sets with no overlap.
- **Rule-based grammar checks need measuring too.** Run over 29,211 sentences from published books, where almost every flag is a false alarm, the checker flagged 22% of them, mostly for passive voice. I turned passive voice and sentence length into style hints and fixed the rules behind the rest. That took the sentences with error- or warning-level flags from 347 to 184.

## Repository layout

```
speech_engine/        the library: audio, models, training, decoding, analysis, learning
scripts/speech/       data preparation, training and evaluation for speech recognition
scripts/translation/  corpus download and filtering, training, evaluation, overlap audit
scripts/analysis/     treebank resources, POS calibration, parser training and evaluation
scripts/demos/        small self-contained demos of each component on toy data
webapp/               local web app (standard-library server + one static page)
tests/                pytest suite and Node tests for the front end
results/              JSON reports behind every number in RESULTS.md
docs/                 design notes for each component
```

## Running it

Requires Python 3.11+ and PyTorch (CUDA optional). Install PyTorch for your platform first (see pytorch.org), then:

```bash
pip install -e . pytest
python webapp/server.py        # local web app at http://127.0.0.1:8765
python -m pytest -q            # tests
python scripts/demos/ctc_training.py   # quick toy demo, no downloads
```

Model weights and datasets aren't in the repository (they're large). The scripts download the public data and retrain each model; the design notes in `docs/` list the commands behind each result.

## Limitations

- Speech recognition is English read speech only. Regional accents are much worse (66–78% WER on the Northern English and Irish English dev speakers), and noisy or conversational audio hasn't been tested.
- Translation handles formal Spanish reasonably well but still mangles names and sometimes runs on past the end of short sentences.
- Language detection is hand-written rules and is unreliable on short or machine-generated text.
- Nothing has been tested with real learners yet.

## Next

- Fine-tune the older translation model on clean data, fix its run-on short sentences, and install it.
- More languages: Spanish analysis from UD, then translation for a handful of language pairs.
- A reading/watching overlay that shows word-by-word help on subtitles and web pages.

## Data

LibriSpeech (CC BY 4.0), SLR83 UK and Ireland English dialects (CC BY-SA 4.0), VoxPopuli (CC0, test only), Universal Dependencies treebanks (licences per treebank), OPUS Tatoeba, Europarl and GlobalVoices, News Commentary (evaluation only). Licence notes are kept next to each dataset.

## AI assistance

I used AI tools while building this project. What they were used for, and how, is documented in [docs/ai-assistance](docs/ai-assistance/).
