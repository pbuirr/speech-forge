# Linguistic analysis

What the app shows about a sentence: language, each word's lemma, part of speech and features, the sentence's dependency structure, and grammar notes. Everything works offline from resources built out of Universal Dependencies (UD) treebanks.

`speech_engine/understanding.py` runs these components over one shared text record, so their results all point at the same character offsets in the original text.

## Components

**Language detection** (`language_detector.py`). Unicode script first (Thai, Hangul, Georgian and similar scripts decide the language on their own), then distinctive characters and frequent n-grams within the script, then dialect markers (colour/color, vosotros/ustedes). This is hand-written rules and it's the weakest part of the pipeline: on short text or recognition output it's often wrong. Generated text is never routed automatically; the user picks or confirms the language.

**Lexicon** (`linguistic_resources.py`). Compiled from a treebank's training split. For each word form it stores every attested lemma, UPOS tag and feature set, plus where it came from. CoNLL-U has no morpheme boundaries, so the lexicon doesn't invent any. BPE subwords are shown separately and aren't presented as morphemes.

**POS tagging** (`morphology.py`, `unknown_words.py`, `pos_uncertainty.py`). A bigram HMM over the lexicon's candidate tags. Unknown words get tag evidence from case variants, suffixes and word shape. Forward–backward marginals give a confidence score per tag, and isotonic calibration fitted on the dev split turns that score into an estimated accuracy, with separate curves for seen and unseen words.

**Dependency parsing** (`dependency.py`). An arc-standard transition parser trained with an averaged perceptron and a static oracle, using POS, features, frequent word forms and attachment distance. It always produces a single-rooted tree, but can't represent non-projective structures.

**Grammar checks** (`grammar.py`). Pattern rules: English agreement, a/an, common confusions (then/than, should of, whose/who's), wordy phrases and punctuation, plus a few Spanish and French rules. Passive voice and long sentences are style hints and don't lower the score.

## Results

| Task | Data | Result |
| --- | --- | --- |
| POS, gold tokens | UD English-EWT test | 90.2% (68.5% with the earlier rule-based tagger) |
| POS, raw text | UD English-EWT test | 87.8% accuracy, 92.9% token F1 |
| Parsing LAS, predicted POS/features | EWT / PUD / Turkish-BOUN | 72.5 / 70.5 / 45.3 |
| Parsing LAS, gold POS/features | EWT / PUD / Turkish-BOUN | 79.0 / 77.6 / 57.9 |
| Grammar false alarms | 29,211 published sentences | 184 flagged (was 347) |

PUD is a different domain from EWT and was never used for training or tuning. Punctuation is excluded from parser scores, and test sentences that also appear in training or dev were removed.

**Grammar precision.** Running every rule over published book text, where almost any flag is a false alarm, showed 347 sentences with error- or warning-level flags. The main causes were agreement rules firing on contractions like "wasn't", confusion rules flagging correct uses of "to", "effect" and "might of", French spacing before punctuation, and Spanish accent rules firing on statements like "creo que sí". After fixing those and turning passive voice and sentence length into style hints, it's 184. Most of what's left is old spelling in older books ("an hysterical") and dialect in dialogue. Each fix has a regression test in `tests/test_grammar_precision_probes.py`. Recall on real learner errors hasn't been measured yet.

## Limitations

- English (EWT) and Turkish (BOUN) have lexicons, taggers and parsers built from UD; English is the more thoroughly evaluated. Other languages fall back to simple POS rules and have no lemmas.
- The gap between gold and predicted POS (79.0 vs 72.5 LAS on EWT) shows tagging errors carry over into the parse.
- Parser output has no calibrated confidence.

## Reproducing

```bash
python scripts/analysis/prepare_english_resource.py --calibrate
python scripts/analysis/train_dependency.py --help
python scripts/analysis/evaluate_dependency.py --help
python scripts/analysis/evaluate_sentence_pipeline.py --help
```

UD English-EWT and the derived resources are CC BY-SA 4.0. The licence and README files are kept next to the compiled resources in `data/linguistics/`.
