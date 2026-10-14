# Results

Every experiment that changed a decision, newest first. Development sets are used for choosing; test sets are run once on the chosen model.

## Speech recognition

**Test set:** 160 LibriSpeech test-clean recordings, 40 speakers never seen in training, chosen before any model was evaluated on it.
**Dev set:** 175 recordings — 80 MiniLibriSpeech (13 speakers) and 95 SLR83 regional English. The regional half comes from one anonymous speaker code per region, so per-dialect numbers are really per-speaker numbers.
**Decoding:** "+ LM" means beam search (width 10) with a character 5-gram model built only from training transcripts, weight 0.2.

| Date | Model | Training | Dev WER + LM | Test WER + LM | Test CER + LM | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-13 | Full CNN–BiLSTM, 4.8M params | 465 h, ~400k draws, SpecAugment + speed perturbation | **48.2%** | **22.5%** | **7.9%** | Selected |
| 2026-10-13 | Full CNN–BiLSTM, 4.8M params | 102 h, 96 cycles, SpecAugment + speed perturbation | 54.5% | — | — | Beat wide model on dev; scaled up |
| 2026-10-13 | Wide CNN–BiLSTM, 3.0M params | 102 h, 96 cycles, SpecAugment + speed perturbation | 58.1% | — | — | Rejected: augmentation hurt at this size |
| 2026-10-12 | Wide CNN–BiLSTM, 3.0M params | 102 h, 96 cycles | 55.8% | 30.3% | 10.5% | Selected at the time |
| 2026-10-12 | Wide CNN–BiLSTM, 3.0M params | 102 h, 24 cycles | 68.4% | 47.6% | 17.4% | Baseline |

**Main finding.** The 24-cycle baseline was undertrained, not too small: dev loss fell every cycle until the learning-rate schedule ended (1.8 passes over the data). Four times the training on the same model and data took dev loss from 1.196 to 1.035 and test WER from 47.6% to 30.3%.

**Still weak.** Regional speech: on the 465 h model, dev WER is 50.0% (Scottish), 66.4% (Northern) and 78.1% (Irish), each from a single speaker code. All results use one training seed.

## Translation (Spanish → English)

Scores are on sentences with no exact or near-duplicate overlap with either model's training data (checked with `scripts/translation/check_bitext_overlap.py`). BLEU here is lowercase and whitespace-tokenised, so it isn't directly comparable with published BLEU.

| Model | Training data | News BLEU (400) | News char-F2 | News Commentary BLEU (400) | News Commentary char-F2 |
| --- | --- | --- | --- | --- | --- |
| Older model: Transformer, ~7.5M params, 4k BPE | ~965k pairs: Europarl, OpenSubtitles, Tatoeba | **27.7** | **60.4** | **23.8** | **56.4** |
| Installed model: Transformer, 1.2M params, 2k BPE | ~60k pairs: Tatoeba, GlobalVoices | 16.2 | 48.2 | 9.2 | 40.2 |

**Overlap audit.** The older model had seen 1,984 of the 1,993 sentences in my Tatoeba test, so Tatoeba can't be used to compare it. News overlap was 2 of 1,040 and News Commentary 63 of 45,829; those pairs were removed.

**Known problems with the older model.** On short constructed sentences it sometimes keeps going after the sentence is finished ("Laura sent the report to pedro pedro"), and it mangles some names. Next step: fine-tune it on clean current data and build a clean short-sentence test.

## Word analysis and parsing

| Task | Data | Result |
| --- | --- | --- |
| POS tagging, gold tokens | UD English-EWT test | 90.2% accuracy (68.5% with the old heuristic tagger) |
| POS tagging, raw text | UD English-EWT test | 87.8% accuracy, 92.9% token F1 |
| Dependency parsing (LAS) | English-EWT / English-PUD / Turkish-BOUN | 72.5 / 70.5 / 45.3 |

## Grammar checks

Measured on 29,211 sentences of published book text (LibriSpeech transcripts), where almost any flag is a false alarm.

| Version | Sentences flagged as errors or warnings |
| --- | --- |
| Before | 347 |
| After fixing agreement, confusion, French and Spanish rules; passive voice and sentence length moved to style hints | 184 |

Most remaining flags are old-fashioned spelling in older books ("an hysterical") and dialect in dialogue. How many real learner errors the rules catch hasn't been measured yet.
