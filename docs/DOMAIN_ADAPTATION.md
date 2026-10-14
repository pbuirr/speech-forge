# Real-world translation and speech validation

Latest active-model comparison: [TRANSLATION_RELIABILITY.md](TRANSLATION_RELIABILITY.md).
The measurements below remain the record of this earlier adaptation.

2026-10-11. Original local models, no pretrained weights, hosted inference or new
dependencies. This report supplements the earlier short-sentence experiment in
[REAL_MODEL_VALIDATION.md](REAL_MODEL_VALIDATION.md).

## Translation diagnosis

`scripts/translation/diagnose_translation.py` and `results/nmt_domain_diagnosis.json` describe
the original checkpoint's distribution shift. Tatoeba test sentences average 6.76
words, compared with 19.60 in GlobalVoices and 24.70 in a separate News Commentary
sample. Unseen whitespace word forms increase from 5.62% to 21.71%/20.91%; subword
pieces per word increase from 1.56 to 2.03/2.07. Actual unknown BPE IDs are only
0.61%/0.31% in news. This supports improving lexical/contextual training rather
than assuming that a much larger vocabulary or model is necessary.

Short GlobalVoices examples still score only 9.21 raw BLEU versus 34.65 for short
Tatoeba examples, so sentence length alone cannot explain the gap. Strata are
descriptive, not causal estimates. Names, unfamiliar constructions, fragmentation,
long-distance dependencies and sentence alignment quality remain limitations.

## Adaptation data and comparisons

Used the cached OPUS GlobalVoices 2018Q4 archive, SHA256
`bfc160f021af8c667182cc7425ccabfab34b29943e4d02ebe21b6bf229f8d642`.
[Global Voices attribution policy](https://globalvoices.org/about/global-voices-attribution-policy/)
describes the site's CC BY 3.0 policy and exceptions; the archive directs users to
original source licenses. Preserve OPUS/CASMACAT/Global Voices attribution and
check rights for any redistribution or further use. Raw data remains ignored by Git.

`prepare_news_adaptation.py` reads the release's article metadata, validates
alignment counts after removing blank article separators, and keeps connected
article/source/target groups together. The archive omits its final XML root tag;
only that known omission is repaired, with line counts checked before use.
Entire connected groups touching any previous Tatoeba partition or any of the
1,000 previously downloaded GlobalVoices examples are excluded (606 groups).
Reserved news development/test partitions contain 1,005/1,040 loadable pairs.
Article IDs and archive hashes are retained in `data/training/es_en_news/provenance.json`.

Same original 1,204,736-parameter architecture, BPE vocabularies and initial
checkpoint; batch 64, eight epochs, seed 731. No parameter enlargement or new BPE
fitting. Checkpoints are selected by news development NLL, then compared on the
same 200 news and 200 short-sentence development examples. Single-seed experiments
do not establish robustness to seeds or every training corpus. Counts differ slightly
between recorded archive rows and loadable data because the legacy TSV loader
treats leading `#` as comments; the actual saved partition hashes bind each run.

| Adaptation | News development raw BLEU | Short development raw BLEU |
|---|---:|---:|
| Original checkpoint | 4.26 | 32.26 |
| 39,975 news + 20,000 short replay; LR 0.0008 | 15.35 | 26.12 |
| 24,000 news + 35,864 short replay; LR 0.0008 | 13.09 | 29.93 |
| Same balanced data; LR 0.0002 | 8.50 | 28.24 |
| Same lower-rate run, shuffled from first epoch | 8.43 | 27.56 |

The higher replay proportion reduces forgetting. Lower learning rate alone did not
improve this comparison. These runs inherited sorted length curriculum for the first
20% of epochs (one of eight), followed by shuffled training. Warm-start adaptation
can explicitly shuffle from the first epoch for controlled comparison. The default
preserves the original 20% curriculum; tested alternatives do not automatically become defaults.

Separate publisher evaluation uses OPUS News Commentary v16, provided as research
training data by WMT. [WMT21 licensing information](https://www.statmt.org/wmt21/translation-task.html)
permits research use and requests citation; other uses require consulting original
owners. No News Commentary data is used for training or development selection.
Exact normalized overlap with the adaptation partitions is removed. The original
checkpoint scores 3.25 BLEU on 400 sampled News Commentary sentences.

Selected the balanced LR 0.0008 checkpoint using development results, before final
test evaluation. Neither lower learning rate nor fully shuffled adaptation helped
the measured tradeoff; the original curriculum default is retained. Same model
architecture and vocabulary throughout. Four adaptation runs took about 4.3–5.8
minutes each on the RTX 3050; one run shared hardware with the speech pilot, so
those timings are not a controlled performance comparison.

| Final held-out application evaluation | Original BLEU | Adapted BLEU | Adapted character F2 |
|---|---:|---:|---:|
| Previously reserved GlobalVoices articles, 400 examples | 5.36 | 13.65 | 45.78 |
| Separate News Commentary publisher data, 400 examples | 3.25 | 9.00 | 40.10 |
| Full Tatoeba short-sentence test, 1,993 examples | 33.14 | 30.27 | 53.24 |

This is a real news improvement with a short-sentence regression, not an accuracy
gain on every domain. News remains unreliable in absolute terms. For example, outputs
still invent or garble words and omit important qualifications. Heuristic review flags
fell to roughly 10–11% in news despite remaining poor quality: fluent-looking output
can evade those checks. Do not interpret fewer flags as calibrated confidence or use
unreviewed generated translations as authoritative learning answers.

Final reports: `results/nmt_globalvoices_adapted.json`, `nmt_newscommentary_adapted.json`,
`nmt_tatoeba_adapted.json`. These preserve the evaluated checkpoint SHA256
`87e7454dce8f931c5bb8e32860c41f2346c220f51456488bfcb31f592867258f`.
The previous installed weights are archived before replacement; metadata-only export
changes the installed file hash while preserving evaluated weights.

```powershell
python -X utf8 -u -B scripts/translation/train_local_translation.py --corpus data/training/es_en_news_balanced/train.tsv --splits-dir data/training/es_en_news_balanced --load data/models/archive/nmt_es_en-7be111d843035f82.pt --output-dir data/training/es_en_news_replay --epochs 8 --batch 64 --lr 0.0008 --curriculum-epochs 1 --dataset "GlobalVoices article-disjoint + Tatoeba training replay"
python -X utf8 -u -B scripts/translation/evaluate_local_translation.py --checkpoint data/training/es_en_news_balanced_adapt/best.pt --pairs data/corpora/es_en_newscommentary_independent.tsv --output results/news_recheck.json --limit 400 --device cpu
```

## Shared reliability and application changes

Display cleanup preserves repeated words such as “had had” and “that that,” which
can be grammatical. Missing source numeric anchors remain separate, escaped review
warnings rather than being appended to the translation. Grouped digit formatting
is recognized; this remains a heuristic and does not certify numeric equivalence.
The API retains original neural output. Cached historical predictions can be
rescored with `rescore_translation_display.py`; their inference timings/review flags
remain historical. These metrics are custom lowercase whitespace BLEU and character
F2, not percentages of semantic correctness or calibrated confidence.

The application now reloads atomically replaced translation checkpoints on the next
request, refreshing model metadata and invalidating stale statistical fallback.
Fallback continues to use only the checkpoint's declared training partition.
Language routing, shared text analysis and learning remain the existing application.

Validation: 156 Python tests and JavaScript rendering/flow/speech checks pass. Installed
model was exercised through the real HTTP handler on three constructed inputs,
including a year mistranslation (`2027`→`27`) retained in generated text with a separate
review warning. A news-style constructed sentence produced incorrect added words
without a review flag, confirming a false negative rather than usable general accuracy.
Statistical fallback was precomputed from the new training partition (about 20s cold);
it remains an imperfect separate literal suggestion. Warm CPU inference median/p95
on 200 short development examples is recorded in `nmt_adapted_cpu_latency.json`.

Interacted with the running Chrome app: selected Spanish, translated a news-style
sentence, inspected integrated results, opened/closed the corrected “Open flashcards”
link, and confirmed no cards were saved. The footer is more readable. The user-owned
server initially reported its cached original model. After the user restarted it,
Chrome verified replacement-model inference and the new numeric-warning backend:

- `La conferencia empezó en 2026 y terminó en 2027.` produced `The conference began
  in 2026 and ended in 27.`; the separate warning identified `2027`, a heuristic review
  badge appeared, and the imperfect literal fallback remained explicitly separate.
- The station question produced `Where is the station of train?`; the previous
  numeric warning/fallback cleared. This confirms awkward short-sentence regression,
  not correct idiomatic English.
- The public-health sentence produced `The government announced new media to protect
  the public health health salt.` without a review badge. The generic experimental
  limitation label remained visible; this is a reproduced heuristic false negative.
- Explicit Turkish retained analysis and clearly reported missing tr→en weights,
  listing only es→en. Automatic detection still incorrectly returned English with
  zero evidence; the selection controlled analysis and disagreement warnings appeared.
- Default local CTC stayed disabled with the missing-checkpoint/tokenizer message.
  No real-audio browser transcription was performed. No cards were added or changed.

Browser verification of the replacement is now complete for these workflows. This
establishes functioning integration and warning presentation, not reliable semantic
translation or usable speech recognition. Automatic file-replacement reload was
regression-tested; no extra live model swap was performed in this browser pass.

## Real speech pilot

Downloaded the official [Mini LibriSpeech SLR31](https://openslr.org/31) regression
subset, licensed CC BY 4.0, derived from LibriSpeech (Panayotov et al.). Official MD5
checksums verified before safe extraction. Archive SHA256 values:

- train-clean-5: `47805806c8b15f7549f3c51bd6b7da72ec0128a34d32830c8014a88658c0292d`
- dev-clean-2: `176ec501490eced2d6c1f89f4f0ddc7dfe799e649e5322f8ba49fe3ff50c8012`

Official training subset has 1,519 utterances. Official development speakers were
split between development and an internal held-out test: all three partitions are
speaker-disjoint. This is not the official LibriSpeech test-clean benchmark.
For a bounded pilot, filter complete 1–10-second recordings rather than truncating
audio/transcripts: 304 training clips (0.524 hours), 80 development, 80 test.
Model is the original CNN/BiLSTM CTC, configured with channels 16/32/64, hidden size
128, two recurrent layers. Tokenizer uses training transcripts only.

After eight epochs: held-out CER 75.45%, WER 99.74%. Extended to 40 total epochs,
best development-loss checkpoint (epoch 22): CER 55.12%, WER 97.24%. Training loss
kept dropping while development loss plateaued: more epochs alone did not yield
usable recognition. These small read-English results do not generalize to natural
conversation, noise, dialects or other languages. The pilot remains outside the
default web ASR checkpoint path; no usable speech capability is claimed or enabled.

Fixed the shared acoustic data path: keep real frame lengths, pad per batch, reject
unaligned audio truncation/empty transcripts, account for repeated-character CTC
alignment constraints, and reject non-finite training loss. CNN masks and packed
BiLSTMs avoid padded right context; evaluation decodes only valid frames. A one-epoch
resumed run verifies that path; batch-one held-out predictions are unchanged, so
no quality gain is attributed to the padding correction. Feature caches now distinguish
same-named audio and ordinary file changes. New checkpoints can bind tokenizer
fingerprints and training language; runtime rejects mismatches even at equal vocabulary size.

`CharTokenizer(normalization='unicode')` retains letters, combining marks and digits;
the policy is persisted. Legacy tokenizers retain English normalization and IDs.
This repairs vocabulary extension, not multilingual acoustic recognition. The old
speech demo now defaults to an official training split rather than test-clean.

Reproduction uses `prepare_real_speech.py` and `train_real_speech.py`; detailed pilot
reports are in `results/asr_real_initial.json`, `asr_real_extended.json` and
`asr_real_padding_validation.json`. Next speech work needs more representative audio,
longer complete clips and stronger held-out WER, rather than presenting this pilot
as reliable transcription. No commits or pushes.
