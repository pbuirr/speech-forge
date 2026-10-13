# Local POS reliability

The project's count-based tagger now computes forward–backward marginals over
the same permitted paths used by Viterbi. See [Rabiner's HMM tutorial](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf)
for the algorithm. A marginal measures mass assigned by the model, conditional
on its constraints; it is not a probability of linguistic correctness. Observed
forms restrict candidate tags, so a singleton candidate can score 1 and be wrong.

The local reliability model maps the selected tag's marginal to observed POS
accuracy using twenty fixed score bins, separate seen/unseen-form curves, and
pool-adjacent-violators isotonic fitting. [Zadrozny and Elkan](https://www.cs.columbia.edu/~djhsu/coms4771-f25/handouts/zadrozny2002kdd.pdf)
describe isotonic score calibration. This implementation calibrates the binary
event that the returned POS is correct, not the entire multiclass distribution.
The binning and seen/unseen grouping are project choices evaluated below.

## Reproduce

```bash
python scripts/prepare_english_resource.py --calibrate
python webapp/server.py
```

This downloads the pinned EWT development split as well as train/test, checks
checksums, builds the original model, fits calibration locally on development
labels, and evaluates on test labels. No external inference, pretrained weights,
or new dependencies are used. The generic `scripts/calibrate_pos.py` accepts other
resources and their original training/development/test corpora. Generated
calibration files live under `data/linguistics/calibration/` and are ignored by Git.

Calibration files bind to the training-corpus checksum, a fingerprint of the actual
transition/emission counts, and scoring version. Reloading or modifying training
invalidates calibration. Developers must bump `SCORING_VERSION` when changing
scoring constraints, smoothing, or hints. Batch loading validates all files before
applying any; a mismatched file is rejected rather than silently reused.

## Measured results

English evaluation retained 24,690 scored POS decisions. Development excluded 68
exact training word sequences; test excluded 115 sequences found in training or
development. Another 169 test words lacked a matching scored returned POS and are
reported as unscored. Near duplicates are not detected. These exclusions make this
a different population from the earlier POS accuracy report.

| Metric | Raw model scores | Development-fitted estimates |
|---|---:|---:|
| POS accuracy | 90.11% | 90.11% |
| Expected calibration error, ten fixed bins | 2.16% | 1.13% |
| Binary Brier score | 0.05781 | 0.05759 |

Predictions are unchanged. Calibration error improved, while Brier improvement is
small. At the fixed estimate threshold 0.9, retained decisions cover 75.7% of scored
words with 98.2% observed accuracy. This does not establish a guarantee or a new
overall accuracy score. Unknown-form POS accuracy is only 47.7% on 2,114 words,
making unknown-word modeling the next important accuracy problem.

On 200 EWT sentences containing 4,267 words, median local inference time over
three warmed runs was 0.020 seconds for Viterbi and 0.083 seconds with marginals.
This is additional computation, not a speed improvement; hardware and inputs vary.
Reports are `results/pos_calibration_ewt.json` and
`results/pos_uncertainty_performance.json`.

## Shared behavior and limits

`pos_uncertainty` carries the selected tag, conditional marginal, competing-tag
distribution, entropy, calibration support, and source hashes through the existing
API and saved card analysis. A score is marked as applicable only when the decoder
tag matches the returned POS. The advanced view labels development estimates;
low estimated reliability adds a POS-specific review hint and `pos-uncertain` card
tag. The fixed 0.8 review threshold is a workflow choice, not a safety guarantee.
Inputs outside observed score-bin bounds receive no review decision from calibration.

General `confidence` remains null. These estimates assume correct token boundaries
and comparable data; they do not calibrate lemma, morphological features, grammar,
meaning, speech, or translation. Raw-text errors and domain shift can invalidate
their interpretation. Only English has a fitted calibration resource at present;
the implementation is shared across languages. No cards are automatically approved.


## Learned unknown-word evidence (2026-10-11)

The previous lattice assigned nearly flat unknown-word emissions, with a fixed
fourfold handwritten hint. Unknown proper names were often labeled NOUN, PRON,
or DET. The shared decoder now derives spelling evidence locally from the
training lexicon: types occurring at most ten times, weighted one unit per
spelling and shared across ambiguous tags. Unicode capitalization strata and
final character sequences of lengths zero through five provide successively
smoothed tag distributions (five pseudo-observations of the shorter context).
Posterior/prior ratios integrate this evidence with the existing HMM transitions.
This is inspired by [Brants (2000), section 2.3](https://arxiv.org/pdf/cs/0003055),
with our own type weighting and smoothing; it is not an exact TnT implementation.
These sequences are spelling features, not asserted morphemes. No dependencies,
pretrained weights, hosted inference, or changes to known-word candidate sets.

Parameters were fixed before evaluating the test sets, not selected on test
accuracy. Training remains EWT train; calibration is refitted on EWT dev only.
The scoring fingerprint is now `hmm_suffix_shape_v2`, rejecting stale calibration.
The API, shared understanding pipeline, and saved learning evidence reuse the
same contextual decisions. Training invalidates the derived spelling cache.
The algorithm accepts any corpus language; measured natural-corpus results below
are English only. Authored Turkish tests establish code behavior, not population
accuracy or learned cross-language transfer.

| Held-out evaluation | Previous | Learned |
| --- | ---: | ---: |
| EWT unknown, reliability-scored subset | 47.73% (2,114) | 64.32% (2,113) |
| EWT unknown, all returned predictions | 50.31% (2,276) | 65.69% (2,276) |
| PUD unknown, all returned predictions | 48.40% (2,399) | 75.41% (2,399) |
| EWT overall reliability-scored POS | 90.11% | 91.60% |

The original 47.7% denominator excludes returned-POS/decoder mismatches; it is
not interchangeable with all unknown words. New reports include both populations.
Reliability-score coverage is 99.32% of eligible EWT returned predictions and
99.86% for PUD. Exact train/dev sentence overlap is excluded, not near duplicates.
EWT calibrated ECE falls from the previous 1.13% to 0.81%; raw model ECE increases
from 2.16% to 2.48%, reinforcing the need to recalibrate rather than trust raw
marginals. Unchanged EWT-dev calibration yields 1.01% overall ECE on PUD; this
one domain-shift success does not establish calibration in other languages.

PUD supplies news/Wikipedia instead of EWT's web genres, with its own annotation
conventions. Repository commit `f16eba4ae7f3d161870ed320676c5088b8fa476c`, corpus
SHA and license metadata are recorded in `results/pos_unknown_pud.json`; notices
are preserved under `data/linguistics/UD_English-PUD.*`. No PUD data was trained on.

On this machine, one run of full inference including first spelling-model build
and marginals took 0.99 seconds for 24,859 EWT words and 0.95 seconds for 21,180
PUD words. These are descriptive measurements, not a controlled speed claim;
the comparison harness builds the new cache even for its baseline emulation.
Remaining errors concentrate on proper/common noun ambiguity. Gold token and
sentence boundaries are supplied; these gains do not establish improvements to
raw-text segmentation, unknown lemmas, morphology, grammar, or translation.

Reproduce with `scripts/evaluate_unknown_words.py --help`, supplying the pinned
resource, train/dev files, regenerated calibration, and a separate test corpus.
Reports: `results/pos_suffix_calibration_ewt.json`, `results/pos_unknown_ewt.json`,
`results/pos_unknown_pud.json`. Validation: 92 Python tests and both Node UI suites.


## Case variants and lexical Unicode coverage (2026-10-11)

Unknown spellings now receive smoothed POS evidence from training forms with the
same language-aware lowercase key when available, falling back to the previous
suffix/capitalization model otherwise. All learned lexical tags remain possible;
case similarity does not license transferring lemmas or morphological features.
Turkish/Azerbaijani dotted/dotless I are handled explicitly; no accent stripping,
compatibility normalization, or global casefold equivalence is introduced.
Corpus lookup remains exact and source alternatives remain intact.

A development-only comparison interpolated case evidence with suffix evidence
at weights 0, .25, .5, .75, 1. Unknown scored accuracy was respectively 64.49%,
65.60%, 66.25%, 67.10%, 67.25%; weight 1 was selected before test evaluation.
Five prior pseudo-observations preserve alternative-tag support. The comparison
used the same original train counts and excluded exact train/dev sentence overlap.
Calibration then reused development data, so development scores are selection
results, not unbiased accuracy estimates. See `results/pos_case_development.json`.

A separate shared analysis correction checks the whole surface for letters
instead of just its first character. Digit/joiner-initial lexical words therefore
retain contextual POS and uncertainty. Existing numeric-expression behavior is
preserved. This does not establish morphological segmentation of these forms.

| All returned unknown predictions | Previous suffix model | Current |
| --- | ---: | ---: |
| EWT (2,276 words) | 65.69% | 69.33% |
| PUD (2,399 words) | 75.41% | 76.03% |

EWT NOUN→PROPN errors fell 111→63, but PROPN→NOUN rose 141→144; proper-name
recognition is not solved. PUD PROPN→NOUN rose 122→132 despite improved total
unknown accuracy. Shared contextual inference, not entity classification, is being
measured. Natural-corpus accuracy remains English-only; authored Turkish tests
verify specialization and retention of dotted/dotless distinctions.

EWT scored overall accuracy is 91.86% (24,715 decisions), with reliability
coverage 99.42% of all eligible returned decisions. PUD coverage is 99.92%.
Coverage changes mean scored-subset accuracy is not a strictly identical-population
comparison; the unknown table above has fixed denominators. After refitting,
calibrated ECE is 1.08% EWT (previous .81%) and .75% PUD (previous 1.01%). Thus
calibration did not uniformly improve. General confidence remains null.
One local full-inference run including model construction and marginals took
1.00 seconds EWT and .94 seconds PUD; not a controlled speed comparison.

Scoring version `hmm_suffix_case_variants_v3` rejects older calibrators; the local
English artifact was regenerated. Reports are `results/pos_case_calibration_ewt.json`,
`results/pos_case_ewt.json`, and `results/pos_case_pud.json`. The latter retains
pinned independent-corpus provenance. Both test corpora were previously evaluated,
so these are regression/domain checks, not newly blind benchmarks. No test data
was used to train counts or select the case-evidence weight. No new dependency,
pretrained model, or hosted inference. 94 Python tests and Node UI checks pass.
