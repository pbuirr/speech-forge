# Connected learning: evidence and limits

Exercises reuse the analyzer's original-text offsets, preserved in flashcards,
instead of guessing word locations by splitting on spaces. Stored spans must
match the target, text bounds, and production token boundaries. Legacy/manual
cards use the same tokenizer without POS/model inference; unmatched targets fall
back to a definition prompt. Punctuation, newlines, and spacing remain intact.

Matching follows [Unicode canonical normalization](https://www.unicode.org/reports/tr15/)
and the [Unicode casing guidance](https://unicode.org/versions/Unicode17.0.0/core-spec/chapter-5/).
NFC accepts equivalent encodings, while conservative lowercasing preserves accent,
joiner, compatibility-character, and spelling distinctions. Turkish/Azerbaijani
dotted/dotless I require an explicit language mapping. This is not complete
locale-sensitive matching or semantic equivalence. Exact case-sensitive grading
is available on `FillInBlankExercise` when required.

Multiple-choice distractors are unique under this matching policy and use the same
base language; an identical gloss is excluded. No distractors returns `None` (HTTP
400 with an explanation). Different glosses may still be synonymous: no semantic
validation is claimed. Auto-generated card analyses remain drafts needing review.
Word-order exercises reconstruct one reference, not all grammatical alternatives.
They remain space-delimited and skip examples without a distinct permutation.

On twelve authored cloze cases covering ten languages, the original repository
implementation produced 3 expected prompts; the revised implementation produced
12. Chinese/Japanese multi-character targets use explicit authored spans; this is
not evidence of predicted word segmentation. The report includes fixture and
baseline hashes. These cases measure behavior, not population accuracy or student
learning. Grammar health checks additionally completed 18,693 applicable rule
executions on 2,077 EWT original sentences without failures or invalid offsets;
they do not establish correctness of the rule findings.

```bash
python scripts/analysis/evaluate_learning_exercises.py
python scripts/analysis/evaluate_learning_exercises.py --baseline-module PATH_TO_TRUSTED_OLD_EXERCISE_PY --grammar-corpus PATH_TO_EWT_TEST_CONLLU
python -m pytest -q
node tests/test_web_rendering.cjs
```

The recorded baseline is `git show HEAD:speech_engine/exercise.py` before any
commit of this increment. Use an exported file to compare; do not overwrite the
working implementation. The optional module is executed Python and must be trusted.


## Shared contextual learning records (2026-10-11)

`/api/extract` now runs the existing LinguisticPipeline once and uses its shared
morphology and grammar results. Each draft card carries source SHA-256, language
selection, Unicode offset units, token indices, source occurrences, per-occurrence
analysis (including alternatives and reliability), linked heuristic rule findings,
and failed-rule information. No model is rerun during extraction. The deck and
review UI show the source and label findings as heuristics; absent findings never
claim correctness. Draft cards still require review. User card text is escaped.

Deduplication separates trained/corpus POS or lemma distinctions instead of
merging solely by spelling. Repeated matching uses retain every occurrence.
Heuristic capitalization alone does not create new lexical entries; its different
predictions are preserved in occurrence evidence. This maintains Turkish case
matching and NFC equivalence. POS review tags consider every retained occurrence.
Existing single-span cloze behavior selects the specific card's first occurrence;
no word senses, named entities, dependency edges, or translations are invented.

On the first 200 original texts from each pinned EWT test and independent PUD
corpus, 4,039 source occurrences were retained versus 3,947 under emulated legacy
first-form-only extraction. Seven distinct predicted analyses were previously
merged. All 4,039 exact source-slice checks and 3,954 card cloze-span checks passed.
201 cards retained linked grammar findings. Full analysis/extraction took about
0.53 seconds PUD and 0.56 seconds EWT in one local run after resource loading.
These measure evidence retention and execution, not linguistic accuracy, grammar
quality, learning effectiveness, or general performance. Raw-text tokenization
limitations still apply. Authored Turkish, Arabic and Spanish tests verify common
source/offset transport; measured natural-corpus coverage remains English-only.

Reproduce `scripts/analysis/evaluate_contextual_learning.py --help`; reports are
`results/contextual_learning_ewt.json` and `results/contextual_learning_pud.json`.
Validation covers persistence, homograph cloze selection, multilingual offsets,
no repeated inference, real HTTP extraction, and escaped UI context rendering.
Translation and speech remain separate prediction producers; this change does
not claim a joint learned representation or semantic analysis. The shared source
and occurrence contract provides a concrete place for future verified alignments.


## Related forms and grammatical explanations (2026-10-11)

The shared morphology payload now links successive occurrences with evidenced
common lemma/POS and compares only grammatical features present at both endpoints.
No lemma is inferred from suffixes for this purpose; unavailable/heuristic stems,
unresolved POS, punctuation and symbols are excluded. Missing feature annotation
is not a negative feature value. Overlapping multivalued annotations are flagged.
A chain represents repeated lemma evidence in linear space instead of all pairs.
Links can span sentences in one input: they are form comparisons, not dependencies,
coreference, or same-sense assertions. POS-disambiguated corpus alternatives make
the comparison a prediction; all alternatives and endpoint source spans/POS
uncertainty remain attached, including when persisted into cards.

The advanced view explains observed contrasts; source-context card reviews retain
these links and their annotations. Definitions are bounded grammatical descriptions
based on the [UD feature inventory](https://universaldependencies.org/u/feat/index.html),
with [case](https://universaldependencies.org/u/feat/Case.html) and
[tense](https://universaldependencies.org/u/feat/Tense.html) interpreted cautiously.
Case is not a universal subject/object classifier, and tense is not a complete
interpretation of event time. Features and values are preserved verbatim, including
language-specific features; unknown feature names receive a generic explanation.
General confidence stays null; POS calibration does not calibrate these links,
features, explanations, or meanings. No feedback into the POS decoder was added.

Evaluating a learned dependency parser would require retaining and validating
training graphs and evaluating attachment and domain transfer. For this iteration,
reuse of existing form evidence was chosen over untrained dependency rules or a
new parser architecture: it enables useful comparisons without invented syntax.
No pretrained model, hosted inference, new dependency, or language-specific
English relation rule is introduced. Authored corpus tests cover English, Turkish,
Russian and Arabic Number/Case comparisons, abstention, overlap, UI escaping,
and propagation into saved learning evidence. These tests are not multilingual
population accuracy estimates.

Held-out gold-word evaluation excludes exact train/dev sentence overlaps:

| Corpus | Links | Endpoint lemma/POS label precision | Feature contrasts | Exact contrast label precision |
| --- | ---: | ---: | ---: | ---: |
| EWT | 1,403 | 94.44% | 98 | 88.78% |
| Independent PUD | 1,432 | 94.34% | 75 | 72.00% |

Reference repeated-lemma/POS opportunities were 1,691 EWT and 1,760 PUD. Those
are coverage denominators, not dependency edges, and predicted chains may skip
unavailable intermediate words. Precision requires both endpoint reference labels
and contrast values to match, not merely equal predicted lemmas. PUD lemmas and
features were automatically added; differences in annotation and prediction both
contribute to its lower contrast score. Results warrant visible uncertainty, not
claims of validated contextual semantics. These are gold-boundary checks and do
not establish raw-text explanation or learning effectiveness. About 1.02 seconds
EWT and .95 seconds PUD for complete inference plus comparison in one local run,
excluding resource loading. Output grows linearly in tokens, not quadratically.

Reproduce `scripts/analysis/evaluate_form_relationships.py --help`; reports in
`results/form_relationships_ewt.json` and `results/form_relationships_pud.json`.
Existing pinned corpus notices/provenance apply. Validation: 104 Python tests,
UI rendering/flow and JavaScript syntax checks. Remaining work includes contextual
feature disambiguation and calibrated, evaluated dependency/semantic interpretation;
translation and speech predictions have not been recharacterized as linguistic facts.
