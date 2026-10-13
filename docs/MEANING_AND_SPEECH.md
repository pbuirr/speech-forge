# Source meaning and real speech

2026-10-11. Original local models, unchanged architecture sizes and token IDs;
no pretrained weights, hosted inference or new dependencies. Translation weights
installed in the app remain the previous evaluated checkpoint. Experimental
translation candidates and speech weights stay local and ignored by Git.

## Source conditioning is not semantic correctness

Added a shared token-NLL/conditional preference objective and a linguistic contrast
evaluator. A reference is scored with its faithful source and a changed source,
using the **same target** to control target-length bias. Eight development and12
held-out cases cover polarity, argument order, names/locations, quantity and temporal
order. These are manually authored contrasts, not a representative semantic benchmark.
They were never training examples. All three models prefer the faithful source in
all12 held-out scoring contrasts; that is not12 correct generated translations.

Original model, ordinary continued training and the auxiliary source-conditioning
objective were compared. Same warm start,59,864 observed bitext pairs, seed731,
LR0.0004, two epochs, effective batch64/microbatch16; auxiliary weight0.2 versus0.
The preference encourages the observed target to score better with its paired
source than with another batch source; identical source/target pairs are excluded.
No invented bilingual training labels or English grammar rules were substituted.
False negatives and noisy observed pairings remain possible. This is not the
published contrastive preference/coreference algorithms, and it does not certify
alignment or sentence meaning.

| Held-out score diagnostic | Current | Ordinary continuation | Auxiliary objective |
|---|---:|---:|---:|
| Faithful source preferred,12 fixed-target contrasts | 12/12 | 12/12 | 12/12 |
| Mean source NLL margin | 1.2068 | 1.2115 | 1.2411 |
| Fresh publisher character F2,200 sentences | 42.1330 | 42.7039 | 42.3234 |

The publisher sample uses seed3031 and excludes801 rows matching sources from
both previous publisher audits. No publisher text was used in training/selection.
All models use the evaluated numeric repetition policy; reporting different decoder
settings as equivalent is avoided. Reports include model/data hashes and generated
texts. Custom character F2 is not factual accuracy. Single-seed differences do not
establish robustness. No candidate was installed based on these small differences.

Generation review demonstrates the gap directly:

- `La médica no encontró el documento.` preserves negation across all models.
- `Elena viajó a Berlín.` turns Berlin into “freedom” despite positive source scores.
- `Irene llamó a David.` corrupts David to “davi”; changed argument order can also
  replace Irene with “him to go.”
- `La estudiante felicitó a la profesora.` improves with the auxiliary model to
  “The student congratulated the teacher”; ordinary training still produces “happy.”
- `Todavía no han firmado el acuerdo.` improves under both continuations, so the
  benefit cannot all be attributed to the auxiliary objective.
- Names, lexical substitutions, tense, omissions and invented words remain.

See results/conditioning_generation_review.json and conditioning_* reports.
The original model already responds to source distinctions in scoring, while its
most likely generated words can be wrong. Random mismatched-source training is
insufficient evidence of robust negation, role or name understanding. Current
Spanish linguistic analysis also lacks validated rich dependency evidence; no
uncertain parse was forced into a purported translation correction. Existing English
source analysis was inspected in the browser: “not” carries Polarity=Neg evidence,
with ambiguity/prediction labels. This is not verified proposition scope.

The auxiliary run took about500 seconds; the ordinary control about170 seconds,
with partial hardware overlap. These are observed costs, not controlled throughput
ratios. A full-batch attempt approached8GB device usage and was interrupted; a
microbatch attempt was interrupted after one epoch while diagnosing loss cost.
Neither partial experiment was installed. Flattening token classification preserves
the mathematical NLL and avoids the slower spatial CUDA loss kernel: a diagnostic
of10 forward/backward calls showed0.307s versus0.028s and maximum value difference
0.0000029, under concurrent training. This kernel result is not application speedup.
The final bounded comparisons completed with the flat loss and accumulated gradients.

## Original real-audio CTC progress

Expanded the cached [Mini LibriSpeech SLR31](https://openslr.org/31) training subset
from304 clips/0.524h to all1,519 clips/5.311h. CC BY4.0; attribution and archive
checksums are retained in data/speech_real/provenance.json. Restore our own best
CNN/BiLSTM CTC weights, matching tokenizer, then train8 epochs with a new optimizer
and schedule. No audio was truncated while retaining a complete transcript.
Existing80 development/80 test clips and speaker separation remain unchanged,
verified by exact saved manifest equality. This internal test is not LibriSpeech's
official test-clean set.

| Same80 held-out clips,greedy decoding | Earlier pilot | Expanded training |
|---|---:|---:|
| Word error rate | 97.24% | 88.03% |
| Character error rate | 55.12% | 40.68% |

Model remains unsuitable for dependable everyday transcription. English read speech
only: no conversation/noise/multilingual recognition claim. Expanded checkpoint SHA:
`7f3eefe2d3d352fbf0c93e3bea723fa85d85e303790f06914aeb40700d835a5e`.
It remains at data/training/asr_expanded/best.pt, not the default checkpoint path.
Actual LocalCTCRecognizer inference on a held-out PCM WAV successfully loads it,
but “yes how many” becomes “as oman.” Inference success is not recognition quality.

Connected existing NgramLM to original CTC prefix beam search through optional
character shallow fusion. The language model learns **only** the1,519 training
transcripts; its backoff scores are ranking scores, not calibrated probabilities.
No external dictionary correction or pretrained language model. Select weight0.2,
character reward0.2 and beam10 on the first20 development clips before testing.

| Independent evaluation | Greedy WER | Selected fusion WER | Selected fusion CER |
|---|---:|---:|---:|
| Same80 held-out clips | 88.03% | 84.24% | 38.21% |
| Separate20 longer held-out clips,10–20s | 85.04% | 82.26% | 35.28% |

The long stratum uses seed2043, excludes the short evaluation by duration, and has
speakers disjoint from training. The remaining decoder settings on long data are
reported diagnostically; settings were not reselected on test results. Decoding-only
elapsed time: selected fusion16.70s for80 short clips and12.44s for20 longer clips,
excluding cached acoustic inference. Greedy is much faster. Browser inference still
uses the original greedy path; fusion is an evaluated experiment, not an implicit
production upgrade. Reports asr_expanded_greedy.json, asr_decoding_development.json,
asr_fusion_test.json and asr_long_test.json retain parameters, hashes and predictions.

Future speech restores now validate tokenizer IDs/normalization/fingerprint and
bind accepted manifests. Changed-data optimizer resumes are rejected; explicit
weight warm-start supports expanded data. Tokenizer bytes are preserved across
restore, and saved manifest hashes bind exact UTF-8 bytes. These guards were added
after the expanded run started; that checkpoint does not contain the new accepted-
manifest guard fields. Its evaluated saved partitions are separately hashed.

## Application integration and learning

Browser inspection reproduced an unnecessary English→English model error during
source analysis. Exact matching language selections now show the original text
without invoking translation, in both frontend workflow and API. Regional tags
must match exactly; this is not dialect conversion. Header says “Original text”
and the note explicitly says no translation/correction was performed. Source
negation, names and spacing are not rewritten. Linguistic analysis remains connected.
Model language is now displayed separately from language-detection evidence for
local speech; a configured English model does not establish the audio's language.

Reviewed learning integration: generated NMT text is not automatically promoted to
card answers. Vocabulary cards retain source examples, annotation/prediction evidence
and review tags. No card writes or authoritative translated answers were created
in this iteration. Recognition text remains provisional and editable; the user must
select analysis after review rather than being automatically given graded answers.

169 Python tests pass, plus both frontend suites. Regression checks cover source-loss
gradients/padding, actual optional training path, exhaustive CTC paths/fusion defaults,
restore provenance and same-language analysis without NMT. Diff whitespace check clean.
No commits or pushes.

Browser follow-up, 2026-10-12: three held-out PCM WAVs were uploaded and transcribed
through Chrome using the expanded checkpoint. Actual outputs were
“i ho mis noli yopospas y min io hon”, “as oman”, and
“tan dotelinef fir wit ma ser”. The corresponding references are in
data/speech_real/browser_checks/references.json. Successful execution does not
establish usable accuracy. The review prompt and English model-language selection
worked; explicitly requested analysis preserved the original transcript and labelled
its disagreement with automatic language detection. A subsequent transcript cleared
stale analysis. No microphone provider, authoritative learning answer or card save
was used. Screenshot evidence is retained in the chat outputs.
