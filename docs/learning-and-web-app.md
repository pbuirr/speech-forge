# Learning features and the web app

## Web app

`webapp/server.py` is a standard-library HTTP server with a JSON API; `webapp/static/` is one HTML page with plain JavaScript and no build step. It runs fully offline.

```bash
python webapp/server.py                       # http://127.0.0.1:8765
python webapp/server.py --asr-checkpoint checkpoints/best.pt --asr-tokenizer data/tokenizer.json
python webapp/server.py --nmt-device cuda
```

| Endpoint | Does |
| --- | --- |
| `POST /api/analyze` | language, word analysis, dependencies and grammar notes for a text |
| `POST /api/translate` | translation, plus a word-by-word gloss when the model output fails its checks |
| `POST /api/transcribe` | speech recognition for a 16 kHz 16-bit WAV (up to 30 s / 2 MB) |
| `POST /api/extract`, `/api/add_card`, `/api/review`, `/api/exercise` | flashcards and practice |
| `POST /api/conjugate`, `/api/lookup_form` | Spanish conjugation tables and reverse lookup |
| `GET /api/languages`, `/api/speech` | which models are installed |

Text that came from a model (a transcript or a translation) carries a marker through every step (`speech_engine/input_evidence.py`). The UI labels it, and it's never used as the answer key for an exercise.

## Learning

- **Flashcards** (`flashcard.py`). Cards can be added by hand or extracted from analysed text: content words become cards, and each card keeps its original sentence and character offsets. Without a translation, the back of an extracted card shows the word's analysis instead of a gloss.
- **Scheduling** (`srs.py`). SM-2: grades below 3 reset a card; otherwise the interval goes 1 day, 6 days, then multiplies by the ease factor.
- **Exercises** (`exercise.py`). Fill-in-the-blank uses the stored offsets rather than splitting on spaces, so it works for punctuation-heavy text and for authored spans in Chinese and Japanese. Multiple choice picks same-POS distractors from the deck. Word order rebuilds the one reference sentence.
- **Answer matching** (`text_matching.py`). Unicode-equivalent spellings count as the same answer, but accents and spelling differences still count as wrong.

On 12 hand-written fill-in-the-blank cases across 10 languages, the earlier exercise code produced 3 correct prompts and the current code produces all 12 (`scripts/analysis/evaluate_learning_exercises.py`). That checks the software; nothing here has been tested with real learners.

## Tests

```bash
python -m pytest -q                 # Python tests
node tests/test_web_rendering.cjs   # front-end rendering and async flows
node tests/test_local_speech_ui.cjs
node tests/test_generated_input_ui.cjs
```
