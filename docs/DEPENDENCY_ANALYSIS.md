# Experimental local dependency analysis

LinguaForge now trains its own labeled arc-standard parser and integrates its
predictions with the existing source-token representation. It adds no dependency,
pretrained weights, hosted inference, or changes to grammar correction/translation.
The design follows the transition-system approach discussed by
[Nivre (2008)](https://aclanthology.org/J08-4003/); all implementation and training
are local. Statistical evidence was compared with left-chain and verb-hub baselines
before integration. This does not establish superiority over other trained parsers.

## Method and integration

- Averaged perceptron, static gold-tree oracle, three epochs. Features reuse POS,
  grammatical annotations, frequent surface forms, local POS combinations,
  attachment distance and already attached child POS. No lemma/meaning oracle.
- Training validates basic trees: consecutive words, valid heads, one root, no
  self-loops/cycles, compatible root labels. Multiword rows and empty nodes do not
  become separate basic-tree nodes. Enhanced dependencies are not modeled.
- 12,544 EWT training sentences; 12,257 projective trees used, 287 excluded from
  training. Nonprojective test trees are retained in evaluation. The projective
  decoder always produces a single-root, acyclic basic tree, not necessarily a
  correct tree. It cannot represent all structures in free-word-order languages.
- Training took 124 seconds locally. The JSON model is 4.52 MB, with 16,246 feature
  entries and 51 relation labels. Transition count is linear; current child-feature
  extraction scans the sentence and makes total feature extraction quadratic.
  Product inference skips chunks over 128 tokens and reports partial coverage.
- `LinguisticPipeline` analyzes morphology once, then predicts syntax using those
  exact tokens. Original codepoint spans and token indices are preserved. Simple
  final-punctuation chunking is a heuristic, not evaluated sentence segmentation.
- `/api/analyze`, the advanced UI and contextual learning records carry arcs,
  source/model fingerprints and explicit experimental status. Cards retain relevant
  dependent/head surfaces. Existing form evidence, grammar checks, cloze answers,
  translation and speech remain intact. No dependency is an automatically graded
  answer, grammar correction, semantic role, or factual interpretation.
- Local syntax models are loaded from `data/linguistics/syntax/` only when their
  training SHA matches the language's lexical resource. Resource replacement
  invalidates loaded syntax. Unsupported languages are explicitly unavailable.
  Shared code accepts language-specific trained resources; only English has an
  evaluated natural-corpus model. Authored Turkish tests check graph invariants,
  not multilingual parsing accuracy or cross-language learned transfer.

## Measured effectiveness

UAS means correct head; LAS requires correct head and exact basic relation label.
Gold word/sentence boundaries are supplied, punctuation is excluded from scoring,
and exact training/development sentence overlaps are excluded from test data.

| Held-out data | Verb-hub UAS | Parser UAS, production POS/features | Parser LAS, production POS/features | Parser UAS/LAS, gold POS/features |
| --- | ---: | ---: | ---: | ---: |
| EWT dev | 22.39% | 74.36% | 67.30% | 83.53% / 80.69% |
| EWT test (21,830 scored words) | 24.22% | 75.43% | 68.43% | 83.62% / 80.99% |
| Independent PUD (18,732 scored words) | 18.41% | 74.04% | 66.16% | 82.43% / 77.27% |

Both baselines are deliberately simple, not strong learned competitors. Gold
features are an oracle-input diagnostic: they expose upstream annotation errors
and do not describe product accuracy. The gap warrants joint feature/context
improvements; this model was not tuned to test scores. PUD has different domains
and annotation conventions, including automatically added morphological features.
PUD root accuracy is only 60%, despite overall UAS 74%; aggregate scores hide
important weaknesses. Exact per-relation results are retained in reports.

On 400 original raw texts (200 EWT and 200 PUD), the actual shared pipeline and
card extraction produced 8,679 source-linked syntax arcs and attached relevant
syntax to 3,954 cards. All syntax source slices, existing source occurrences and
cloze spans checked correctly, with no over-limit chunks in this sample. Processing
took 1.28s EWT and 1.31s PUD excluding resource loading in single local runs. These
are transport/behavior checks, not raw-text attachment accuracy or student learning
evidence. Full dependency evaluation runs score both gold/production inputs and
baselines; their elapsed time is not isolated parser inference time.

No syntax confidence calibration exists: `confidence` stays null. POS calibration
does not calibrate arcs, roots, semantics or explanations. Prediction mistakes,
unknown-word errors, ambiguous morphology and segmentation errors all propagate.

## Reproduction and validation

Use `scripts/train_dependency.py --help` with the pinned EWT **train** file,
language, source URL/license, three epochs and output
`data/linguistics/syntax/en.json`. Reports record corpus and model SHA-256.
The original corpus pin/notices remain those documented in POS_RELIABILITY.md.
The dependency model derives from CC-BY-SA-4.0 EWT annotations; preserve their
existing source/license notices. PUD is evaluation-only, with its existing notices.

`scripts/evaluate_dependency.py --help` reproduces gold/production feature
comparisons and per-relation scores. Reports:
`results/dependency_{development,test}_ewt.json`,
`results/dependency_test_pud.json`, `results/dependency_training_ewt.json`.
`scripts/evaluate_contextual_learning.py --with-syntax` reproduces raw-text/card
transport checks (`results/contextual_syntax_{ewt,pud}.json`).

112 Python tests plus UI rendering/flow, JavaScript syntax and diff checks pass.
Tests cover tree validation, learning/reloading, graph invariants, sentence/offset
isolation, resource binding, token limits, HTTP integration, persistence and escaped
experimental UI labels. No commits or pushes were made.


## Shared sentence boundaries and rejected beam search (2026-10-11)

The previous POS and syntax code independently ended context at every terminal
punctuation token. The shared pipeline now produces one covering set of token
ranges and original-text sentence spans. POS Viterbi/marginals and syntax consume
those identical ranges. Closing quotes, punctuation runs and dots inside adjacent
alphanumerics are handled together; Unicode Arabic and Indic endings are included.
Short period-final lexical spellings observed in each language's training resource
supply abbreviation hints, without a global English abbreviation list. These are
still heuristics: some abbreviation endings also end sentences, and punctuation-free
speech/text does not provide sufficient sentence evidence. The work is consistent
with the abbreviation ambiguity studied by [Kiss & Strunk (2006)](https://aclanthology.org/J06-4003/),
but does not implement their statistical Punkt algorithm or claim its performance.

`analyze_tokens` treats a supplied token sequence as one upstream sentence unless
explicit validated ranges are passed; it no longer silently fragments supplied
gold sentences at internal punctuation. Original text analysis computes its own
heuristic ranges before POS inference. Sentence IDs/spans flow through the API,
syntax arcs and persisted card contexts. Reviews show relevant sentence context
rather than always the entire input. Full source/examples remain saved. These
records can support future translation/transcript alignment; translation and speech
quality have not been measured or changed by this iteration.

Greedy decoding was compared with a structural beam of size 2 and 4 on the first
400 nonoverlapping EWT development sentences, keeping the locally trained weights
fixed. Head/label accuracy was 73.14%/66.15% greedy, 69.95%/62.56% beam2, and
67.09%/60.44% beam4; correct roots were 273, 223 and 201. Inference took 1.05,
2.25 and 4.34 seconds respectively. Labels do not affect future features, so the
prototype retained only the highest-scoring label for each structural transition.
Wider search over this locally trained classifier therefore did not justify its
runtime/complexity and was removed from the product. Beam-aware training or a
separate root model was not evaluated and cannot be inferred to fail from this test.
`results/dependency_beam_development.json` records the experiment; the prototype
is retained in the chat work directory as `dependency_beam_experiment.py`.

The shared boundary approach and abbreviation ablation were evaluated on development
before test. On 400 held-out records per corpus, paired into 200 source-text inputs
with one space, inference receives no gold boundaries. Exact train/dev sentence
overlap is excluded. Exact source-span F1 measures boundary ends and roots; labeled
attachment F1 also requires correct dependent/head spans and relation labels.
Reference lexical arcs remain in recall denominators if alignment fails (none did
in these samples). One EWT punctuation root is excluded by lexical scoring. This is
a controlled paired-text evaluation, not representative natural paragraphs.

| Raw-text metric | EWT previous → shared | PUD previous → shared |
| --- | ---: | ---: |
| Sentence boundary F1 | 83.71% → 92.01% | 92.27% → 99.38% |
| Root span F1 | 62.38% → 67.10% | 57.07% → 59.13% |
| Labeled attachment F1 | 64.78% → 64.71% | 63.61% → 63.64% |

Root gains reflect fewer spurious roots, not improved recall: EWT root recall
65.66%→65.16%, PUD60.00%→59.50%. Thus cleaner segmentation does not solve
underlying root selection and attachment quality is essentially unchanged. The
abbreviation-free ablation is retained in reports; development favored hints on
boundary/root F1. PUD's near-perfect boundary result is limited to this selected
sample and its punctuation/annotation conventions, not universal segmentation.
Complete local processing took 1.46→1.27s EWT and 1.75→1.56s PUD in single runs;
cache warmup/order prevents treating this as a controlled speed improvement.

On the full gold-boundary PUD evaluation, production UAS/LAS remained74.04%/66.16%.
The large gold-feature diagnostic gap remains an upstream/contextual disambiguation
priority; wider search alone did not overcome it. Parser weights were not retrained
or tuned on test data. Boundary contracts changed the POS scoring fingerprint to
`hmm_shared_sentence_boundaries_v4`, and local English calibration was refitted
on development only. Gold-word EWT POS accuracy remains91.86%, ECE1.08%; that
calibration does not validate raw-text sentence choices or syntax. The UI explicitly
states that correct token and sentence boundaries are assumptions.

Reproduce `scripts/evaluate_sentence_pipeline.py --help`; reports are
`results/sentence_pipeline_{development,test_ewt,test_pud}.json`,
`results/dependency_shared_boundaries_pud.json`, and
`results/pos_sentence_calibration_ewt.json`. Natural-data results are English-only;
authored multilingual boundary tests cover Arabic/Indic/CJK endings, exact offsets,
quotes, decimal points, explicit supplied boundaries and pipeline alignment.
122 Python tests, UI rendering/flow, JavaScript syntax and diff checks pass.
No new dependency, pretrained model, automatic semantic claim, commit or push.

## Training on shared production evidence

The gold-training/runtime mismatch is now measured separately from decoding.
On EWT development data, the original parser's LAS is 67.30% with production
inputs, 75.80% with gold POS and production morphology, 71.83% with production
POS and gold morphology, and 80.69% with both gold inputs. These are diagnostic
interventions, not independent additive effects or available product accuracy.
They establish that both upstream components matter.

`DependencyParser.fit(..., token_sequences=...)` accepts aligned upstream evidence
while preserving the original gold head/relation supervision. The training CLI's
`--resources` option obtains that evidence from the same shared analyzer used at
runtime. Source forms must match exactly; models bind to the training corpus and
POS scoring fingerprint, and incompatible resources are rejected atomically.
Existing gold-feature training and older compatible model files remain supported.
No English POS, root-position, or agreement rules were introduced.

This approach was compared with the existing gold-input training on development
data before separate test evaluation. It uses **in-sample** upstream predictions:
the tagger/lexicon were fit on the training corpus, so their training errors are
optimistic. No test annotations enter fitting. Cross-fitting, joint neural learning,
dynamic-oracle training and nonprojective decoding remain untested alternatives.
[Agić & Schluter (2017)](https://aclanthology.org/P17-2107/) demonstrate why
predicted-input adaptation and fold choices need empirical evaluation across
languages rather than an automatic ten-fold prescription.

The same code was also evaluated on Turkish BOUN, pinned to
`f828b02872123a8dfb336584eab0767c4e55c5f9`. Its manually corrected morphology
and dependency annotations, corpus origins and attribution are documented in
`data/linguistics/UD_Turkish-BOUN.README.md`; CC BY-SA 4.0 terms are retained in
the adjacent LICENSE file. Locally compiled Turkish counts contain 100,713 words
and 31,586 forms. Parser training used 7,803 sentences, of which 7,442 are
projective. English used the unchanged EWT training corpus and three epochs.
Generated lexical/model JSON files remain ignored by Git.

| Test corpus, gold word/sentence boundaries | Production UAS old → adapted | Production LAS old → adapted | Root accuracy old → adapted |
| --- | ---: | ---: | ---: |
| English EWT | 75.43% → 78.21% | 68.43% → 72.51% | 76.51% → 79.65% |
| English PUD (different domain) | 74.04% → 76.62% | 66.16% → 70.45% | 60.00% → 65.00% |
| Turkish BOUN | 56.26% → 58.68% | 42.43% → 45.31% | 55.82% → 62.51% |

All test trees, including nonprojective ones, remain in scoring; punctuation is
excluded. Exact train/dev sentence overlap is excluded. English test corpora have
been used in previous regression evaluations, so they are not fresh blind tests.
Turkish is a second natural-language validation, not evidence of universal transfer.
Resources remain specialized by language inside shared code and representations;
there is no jointly trained cross-language model. Turkish development diagnostics
show morphology is the larger gap: gold morphology with predicted POS raises LAS
43.21%→55.55%, whereas gold POS with predicted morphology raises it only to45.53%.
Replacing the parser alone cannot recover missing case/inflection evidence.

| Raw paired-text test, 400 source records each | Labeled attachment span F1 old → adapted | Root span F1 old → adapted |
| --- | ---: | ---: |
| English EWT | 64.71% → 68.17% | 67.10% → 70.97% |
| English PUD | 63.64% → 68.37% | 59.13% → 63.85% |
| Turkish BOUN | 38.51% → 41.36% | 46.15% → 51.12% |

Both versions use identical shared segmentation. Inference receives source text,
not gold words or boundaries. Five Turkish expanded reference words could not be
aligned exactly; their arcs remain in recall denominators. This controlled paired
text is not a representative paragraph or speech evaluation. Root recall now also
improves (PUD59.50%→64.25%, Turkish46.73%→51.76%). No semantics, translation,
speech or student-learning accuracy improvement is claimed.

Four alternating-order warmed runs on identical cached PUD inputs (400 sentences,
8,101 words) measured median parser latency1.247→1.292s (+3.6%). English model
storage4.52→5.26MB (+16.3%); inference algorithm and dependencies are unchanged.
Training including production evidence took127s English and67s Turkish on this
machine. Syntax confidence stays null; English POS calibration is unchanged and
Turkish POS has no accuracy calibrator. Syntax evidence is propagated through the
existing API/UI/card context rather than used as a grammar verdict or answer key.

Reproduction: download the three `tr_boun-ud-{train,dev,test}.conllu` files from
the pinned [BOUN repository](https://github.com/UniversalDependencies/UD_Turkish-BOUN/tree/f828b02872123a8dfb336584eab0767c4e55c5f9),
then compile training only with `scripts/import_linguistic_corpus.py --lang tr`.
Train each corpus using `scripts/train_dependency.py --resources data/linguistics`
with matching `--language`, `--corpus`, `--source-url`, `--license CC-BY-SA-4.0`,
`--output` and `--report`; omit `--resources` to reproduce the comparison baseline.
`scripts/evaluate_dependency.py` reproduces attachment/input ablations;
`scripts/evaluate_sentence_pipeline.py --lang tr --syntax-resources DIRECTORY`
supports controlled raw model comparisons, and `scripts/benchmark_dependency.py`
compares warmed runtime. Each command's `--help` lists required paths.
Reports are `results/dependency_{production*,input_ablation*,adaptation*}.json`,
`results/dependency_turkish_*_training.json` and `results/sentence_adaptation*.json`.
124 Python tests, Node rendering/flow, JavaScript syntax and diff checks pass.
No commit or push.
