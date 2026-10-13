# Translation reliability experiments

2026-10-11. Original local Transformer, unchanged 1,204,736 parameters and learned
BPE vocabularies. No hosted inference, pretrained weights or dependencies added.
This report supersedes the active-model measurements in DOMAIN_ADAPTATION.md;
that report remains the record of earlier experiments and data licenses.

## What explains the remaining failures?

Direct inspection found visibly misaligned training pairs, such as a Spanish photo
credit paired with unrelated prose, or a rationale paired with a constitutional
citation. This demonstrates noise exists; it does not establish the percentage of
bad alignments or that every awkward output comes from noise. We did not reject
pairs just because names, dates or wording differ: legitimate translations can
paraphrase, translate number words or use different date formats.

The original sentence Transformer already attends across its source sentence.
It has no document context, discourse memory or verified semantic representation.
Its token prediction objective and beam score do not certify preservation of
negation, roles or facts. Lowercasing in the corpus loader and BPE also discards
capitalization evidence. Small training coverage and fragmented unfamiliar words
remain limitations; replacing names from a guessed entity detector would not
establish correctness, especially across writing systems and transliteration.

Repetition blocking operated on subword IDs. Target BPE encodes 2027 as
`2 · 0 · 2 · 7</w>`; a repeated fragment is not necessarily a repeated word. It can
be legitimate inside a number or across two years. We now permit numeric tokens
to repeat when the evaluated checkpoint declares that decoding policy. These are
still model predictions, not numbers copied into an alleged verified correction.

The decoder also has aggressive EOS boosting and source/output length constraints.
Ablations did not justify broadly removing them. The next-token model still makes
semantic errors with fluent output. Attention entropy and surface warning rules
are not correctness tests. Current Spanish linguistic analysis is limited and lacks
validated learned dependency evidence for this direction, so we did not force
English/Turkish POS or syntax predictions into Spanish translation constraints.

Relevant research: [length bias](https://aclanthology.org/W18-6322/),
[search versus model errors](https://aclanthology.org/D19-1331/), and
[conditional cross-entropy filtering](https://aclanthology.org/W18-6478/).
Our single-model paired-versus-mismatched-source ranking is a bounded local
experiment, not the paper's two-direction filtering algorithm or verified alignment.

## Development comparisons

Same 200 news and 200 short development pairs, seed983. Legacy checkpoint raw
news BLEU13.0857, short29.9303. Custom lowercase whitespace BLEU; not semantic
accuracy. Single seed; small differences are not established statistical gains.

| Decoder change, previous weights | News raw BLEU | Short raw BLEU |
|---|---:|---:|
| Remove progressive EOS boost | 13.1300 | not evaluated |
| Remove all repetition guards | 12.5644 | not evaluated |
| Exempt all unfinished BPE fragments | 12.7551 | 29.6671 |
| Exempt numeric BPE tokens | 13.2105 | 29.9303 |

Broad fragment exemptions and removing all guards were rejected. Numeric exemption
is a representation correction with almost unchanged aggregate development scores,
not a claim to solve all number translation. Legacy defaults remain for checkpoints
without the explicit `preserve_numeric_fragments` policy.

For alignment ranking, predict each training target with its paired source and a
deterministic mismatched training source. Rank by mismatched minus paired mean token
NLL. Exclude the bottom10% of the selected24k news; retain all35,864 short replay.
Compare with random10% removal and unfiltered continuation. No dev/test targets
were used for ranking. A model-dependent ranking can favor familiar sentences or
filter genuine rare constructions; do not infer clean-corpus labels from it.

All three start from the same installed adapted weights, seed731, four epochs,
LR0.0004, batch64, shuffled from the first epoch, unchanged architecture/vocab.
Select each checkpoint by news development NLL, then compare translations.

| Continued training | News raw BLEU | Short raw BLEU |
|---|---:|---:|
| Ranked10% removal + full replay | 14.0743 | 30.2460 |
| Matched random10% removal + full replay | 13.8377 | 29.6882 |
| Full data continuation | 14.4059 | 30.1186 |
| Full continuation + numeric exemption | 14.3840 | 30.1186 |

Filtering was not adopted: full continuation did best on news. Each run took
about130–141 seconds on RTX3050, with concurrent CPU evaluations; not a controlled
training-efficiency comparison. Selected full continuation plus numeric decoding
policy using development results before independent evaluation.

Ranking provenance: data/training/es_en_quality/ranking.json. Training inputs and
weights remain ignored by Git. Scripts validate unchanged source/target-disjoint
saved partitions. Reports `nmt_quality_*_dev.json` and `nmt_decode_*_dev.json` preserve
checkpoint hashes and decoder options.

## Held-out and fresh publisher evaluation

| Matched application outputs | Previous BLEU | Selected BLEU | Selected character F2 |
|---|---:|---:|---:|
| Reserved news test articles,400 examples | 12.7134 | 14.7110 | 47.9343 |
| Fresh News Commentary publisher sample,400 | 8.6962 | 9.3726 | 41.4563 |
| Tatoeba test,1,993 | 30.2743 | 29.9204 | 53.4366 |

The reserved news sample comes from data/training/es_en_news/test.tsv. It is **not**
the older1,000 GlobalVoices snapshot in DOMAIN_ADAPTATION.md, so the prior13.65 result
cannot be compared directly to14.71. Both models were evaluated on the same ordered
pairs here. The publisher audit uses seed2027 and excludes all400 previously
evaluated publisher sources; exact source/target training overlap was already
removed. News Commentary was never used for training or development selection.
Data provenance/licenses remain documented in DOMAIN_ADAPTATION.md.

Short-sentence quality regressed again (30.27→29.92); these gains are not universal.
Absolute news quality remains poor. Review signals remain uncalibrated.

Strict source/reference digit-group mention recall improved from87/129 to98/129
on reserved news and58/93 to75/93 on the fresh publisher sample; short data declined
from14/16 to13/16. This measures literal mentions, not numeric equivalence,
hallucination precision, semantic roles or factual correctness. It combines the
model and decoder changes, so improvements cannot all be attributed to the
numeric exemption. Reproduce with compare_translation_reliability.py;
results/nmt_numeric_{news,publisher,short}.json bind input report hashes.

Twenty manually constructed ES–EN challenge cases cover names, numbers, negation,
roles, temporal order and domain vocabulary. They were not training data or used
to select candidates. They are not a natural independent corpus. All requested
lexical anchors appear in11/20 previous outputs and12/20 selected outputs, but
that is **not** sentence accuracy: role reversal, bad grammar and omitted currency
can evade a mention check. See results/nmt_facts_{previous,selected}.json.
The station question remains awkward, names still corrupt, and the negative
conference statement still loses negation. The public-health sentence still
produces “new media” and spurious words. Warning/literal alternatives cannot
repair or verify these failures.

## Installation, shared speech path and application

Selected evaluated weight file SHA256:
`3ceaf710cad32824ac89f540a6014d7c4953e5decfed39da1b02a94caba07e74`.
Installed metadata-bound checkpoint SHA256:
`aeb6c948eb37827628cb1fcaa183278501ca77e38064375df6139200bdad75de`.
The previous checkpoint is archived as
`data/models/archive/nmt_es_en-7dcf2a17ab9d895b.pt`.
Installation now rejects evaluation reports with incompatible decoding options;
policy survives checkpoint serialization. No commits or pushes.

Standalone installed-checkpoint CPU news development evaluation: median49ms,
p95151ms,200 sentences, four Torch threads, beam4; excludes statistical fallback
preparation/browser/API overhead. Concurrent held-out evaluation timings are not
used as a performance comparison. The fallback cache rebuilds for changed training
provenance and is training-only.

Transformer speech encoding now supplies mel lengths to the shared padding-aware
CNN frontend, preserving valid frames under right padding. This reuses the CTC
frontend fix rather than changing architectures or retraining unverified weights.
A direct regression test confirms valid encoder states match with/without padding.
The poor real CTC pilot remains unchanged and disabled; no recognition-quality
improvement or successful browser speech transcription is claimed.

161 Python tests and both Node frontend suites pass; diff whitespace check clean.
Live browser results are recorded below after the user restarted Python.


After the user restarted the server, Chrome interactions verified:

- `La conferencia empezó en 2026 y terminó en 2027.` → `The conference began in
  2026, and ended in 2027.` Both years now appear; the missing-number warning clears.
  A separate heuristic review badge still appears, and the poor alternative remains
  explicitly labeled “Unverified word-by-word alternative (not a correction).”
- `La conferencia no empezó en 2026.` → `The conference began in 2026.` Negation
  is dropped. The word-by-word alternative also fails to preserve it; neither is
  a verified correction. A review badge appears but does not explain this error.
- `Laura envió el informe a Pedro.` → `Laura sent the report to ped.` Laura
  improves compared with the old output, Pedro remains corrupted; no review badge.
  Automatic detection says Polish; explicit Spanish selection controls analysis,
  with an accurate disagreement notice. Detection is not verified language identity.
- `El gobierno anunció nuevas medidas para proteger la salud pública.` → `The
  government announced new media to protect the public health health salt.`
  Still wrong, with no review badge; the experimental model limitation remains visible.

Local CTC remains disabled with its missing-checkpoint notice. No browser speech
recognition, microphone-provider transcription or flashcard write was performed.
Linguistic panels remain available alongside translation results. The numeric
success screenshot is saved in the chat outputs as
`linguaforge-numeric-decoding-fixed.jpg`. These are actual browser interactions,
not HTTP or unit-test claims.

Reproduction (same saved data and original model):

```powershell
python -X utf8 -u -B scripts/prepare_translation_quality.py --checkpoint data/models/archive/nmt_es_en-7dcf2a17ab9d895b.pt --news data/training/es_en_quality/news.tsv --replay data/training/es_en_masked/train.tsv --splits data/training/es_en_news_balanced --output data/training/es_en_quality_recheck
python -X utf8 -u -B scripts/train_local_translation.py --corpus data/training/es_en_news_balanced/train.tsv --splits-dir data/training/es_en_news_balanced --load data/models/archive/nmt_es_en-7dcf2a17ab9d895b.pt --output-dir data/training/es_en_quality_full_recheck --epochs 4 --batch 64 --lr 0.0004 --curriculum-epochs 0
python -X utf8 -u -B scripts/evaluate_local_translation.py --checkpoint data/models/nmt_es_en.pt --pairs data/corpora/es_en_newscommentary_independent.tsv --output results/publisher_recheck.json --limit 400 --sample-seed 2027 --exclude-prior-report results/nmt_newscommentary_adapted.json --device cpu
python -X utf8 -u -B scripts/evaluate_translation_facts.py --checkpoint data/models/nmt_es_en.pt --output results/facts_recheck.json
```

The next useful work is stronger source-conditioned learning and cleaner aligned
coverage with independent negation/role/name evaluation, rather than converting
warnings into alleged corrections or optimizing this challenge list. Document
context requires aligned contextual data and measured benefit before adding memory.
