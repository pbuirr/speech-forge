# Acoustic quality investigation

2026-10-12. Original locally trained 807,553-parameter CNN/BiLSTM CTC;
no new dependencies, pretrained weights, hosted recognition or grammar correction.

## Diagnosis and comparisons

Data: cached [Mini LibriSpeech, SLR31](https://openslr.org/31), CC BY 4.0.
Archive checksums, attribution and speaker inventory are retained in
data/speech_real/provenance.json. This resource is a regression subset, not a
representative multilingual training corpus. Training: 1,519 clips, 5.311 hours,
28 speakers. Existing short development/test partitions have 80 clips each,
13/12 speakers respectively, all disjoint from training and each other.

results/asr_data_audit.json checks accepted partitions: all files are 16 kHz,
no normalized transcript changes, no unrepresented characters, and no impossible
CTC alignments, including adjacent repeated labels. Minimum alignment slack is
62 training frames and 60 test frames after CNN downsampling. This checks feasible
alignment lengths, not whether learned alignments are linguistically correct.
Development word OOV rate is 13.32%; characters remain representable, so the model
can generate unseen words without dictionary replacement. Read-speech, accents and
speaker diversity remain limited; no conversational/noisy/other-language claim.

The original expanded checkpoint's training sample (first 80 saved training rows)
has 76.33% WER / 29.72% CER versus 87.98% / 40.74% on development. Both weak fitting
and generalization matter. Sampling is deterministic, not a full training metric.
The preceding eight-epoch run's development loss still fell at the end.

CNN BatchNorm includes padded cells before later masks. The sequential saved
batch-eight training order has about 20.54% padding; shuffled training batches
need not have the same fraction. Recompute running moments from training audio
only with dropout disabled: batch one gives development WER 87.89%, versus 87.98%
original; batch eight gives 88.24%. Tiny gains and CER regression do not justify
adopting recalibration. See results/asr_normalization_diagnosis.json.

An optional training correction computes CNN normalization moments from real
time/frequency cells, retaining the same learned affine parameters, checkpoint
keys, convolution sizes and recurrent architecture. Population variance is used
for normalization and sample variance for running estimates, following
[PyTorch BatchNorm conventions](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm2d.html).
Default legacy behavior is preserved. Padding-invariance and finite-gradient
regressions cover the corrected training path. Different normalization cannot
silently resume the old optimizer; explicit warm-start remains supported.

Compare ordinary and corrected-normalization continuations: same expanded warm
start, all accepted training clips, seed839, batch8, 12 additional epochs, peak
LR0.0005, warmup30, cosine floor0.000001. Tokenizer bytes and accepted manifests
are bound to each checkpoint. Features are reused from the same deterministic
cache. Candidate training skips test inference; best checkpoints are selected by
development CTC loss, then decoder alternatives by all80 development WER.
Single-seed findings do not establish robust statistical superiority.

## Measurement and application safeguards

The full development decoder comparison shows 860 substitutions, 110 deletions,
47 insertions for original greedy recognition. These describe one minimum-edit
alignment, not the linguistic cause of an error. Character backoff fusion learns
only training transcripts using the existing shared NgramLM. It influences CTC
search rather than applying an authoritative grammar correction to a transcript.
Legacy checkpoints still use greedy inference; evaluated decoder metadata can
explicitly configure fusion. Runtime validates the training manifest fingerprint.

CTC sums compatible frame alignments and removes repeated labels/blanks as
described by [Graves et al.](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
No transcript timing labels are invented. Error counts, CPU acoustic/feature
time, decoding time and real-time factor are recorded by
scripts/evaluate_speech_decoding.py. CER counts Unicode code points including
spaces; WER uses whitespace-delimited words, suitable here for English. Other
writing systems need appropriate token-unit evaluation, not automatic English
word segmentation assumptions. The dataset now rejects silently dropped
unrepresented normalized characters instead of scoring altered references.

SpecAugment is a promising future regularizer
([Park et al.](https://arxiv.org/abs/1904.08779)), but it is not measured or adopted
in this comparison. More data, richer features and larger models are likewise
not credited with unmeasured gains. Existing acoustic preprocessing and tokenizer
are held constant to isolate the compared changes.

## Results and selected package

Development comparisons (CPU, four threads):

| Candidate | Greedy WER | Greedy CER | Selected fusion WER | Fusion CER |
|---|---:|---:|---:|---:|
| Previous expanded checkpoint | 87.98% | 40.74% | 83.91% | 38.33% |
| Ordinary continuation | 86.42% | 38.21% | 82.01% | 36.17% |
| Padding-aware continuation | 84.78% | 37.51% | 81.23% | 35.96% |

Selected padding-aware candidate and beam10/LM weight0.2/character reward0.2
on development before test inference. Both continuations used the same original
warm start; selecting minimum CTC-loss checkpoints chose ordinary epoch8 and
padding-aware epoch12. This is a comparison of training/selection strategies,
not a proof that every observed difference is attributable to one operation.
No model parameter count increase. Reports asr_*_development_decoders.json.

| Evaluation | Previous greedy WER / CER | Selected greedy WER / CER | Selected fusion WER / CER |
|---|---:|---:|---:|
| Same80 held-out clips | 88.03% / 40.68% | 84.41% / 37.35% | 81.31% / 34.95% |
| Fresh120 utterances | 87.26% / 40.31% | 84.62% / 37.20% | 80.08% / 35.20% |

Previous checkpoint with fusion on fresh120: WER84.29%, CER38.22%. Fresh sample
uses seed127, durations1–29s, excludes100 previously evaluated short/long paths,
and was fixed before candidate test inference. It comes from the same withheld
speaker pool and corpus, not an independent domain/corpus. Source/manifest hashes
are retained in asr_fresh_test_selection.json and evaluation reports. No decoder
settings were reselected on these results. Same80 fusion errors comprise797
substitutions,91 deletions,56 insertions: quality remains unsuitable for daily use.

Observed CUDA epoch-loop cost: ordinary continuation approximately8.2 minutes;
padding-aware continuation542.5 seconds, approximately9 minutes. Startup/evaluation
are excluded. Early diagnostic/test activity partially overlapped hardware, so this
is not a controlled throughput comparison. TestCPU feature/acoustic work and fusion
decoding take13.62s for409s of short audio; fresh120 take26.99s for763s audio.
Separate evaluation processes briefly overlapped; do not infer precise speedups.

Direct packaged PCM inference on the three prepared recordings, four threads:
first request0.213s including model/LM initialization; warm medians0.076s,0.058s,
0.063s over three repeated calls each. These are short recordings, not a load test.
Reports asr_selected_pcm_runtime.json and asr_upload_preprocessing_audit.json.
The upload and file-training preprocessing produce exactly identical features on
these WAVs. New outputs remain bad: “yes how many” → “as homan”; the negated
request becomes “then do  ine tr wit measer”, losing the negation. Aggregate gains
do not guarantee that an individual utterance improves or preserves its meaning.

Packaged model: data/models/asr_en_quality.pt with matching
asr_en_quality.tokenizer.json and evaluation metadata. Source acoustic checkpoint
SHA85c89af52c92039122a8266e72562ff677fc08bc70e9f00da969175b9698c852;
packaged SHA bc19cc682d083a1391ca07f9e488effa4643fdd7d572e89fc436d8872a289108.
Original checkpoints are unchanged. Installation validates source checkpoint,
tokenizer and evaluation hashes and chooses the decoder by development WER.
Runtime verifies its training-only LM manifest fingerprint. It is explicitly
configured for experimental use, not silently enabled as a dependable default.

173 Python tests pass, including normalization padding invariance, Unicode reference
coverage, edit-count consistency, source-bound packaging and local fusion wiring.
No commits or pushes. After the user's server restart, the selected package was
actually tested through Chrome on all three prepared held-out WAVs. Outputs
matched the direct PCM runtime, including the two spaces after “do” in the third
transcript. The review prompt, English source selection, explicit original-text
analysis and clearing of prior analysis on a new transcript were verified.
Detection disagreement was labelled uncalibrated; no correction was performed.
See APPLICATION_BROWSER_TESTING.md for the observed outputs and limitations.

## Generalization and controlled feature-masking follow-up

2026-10-12. Preserve the installed package: neither new continuation is a robust
replacement. A targeted regularizer has mixed results, and the harder evaluation
exposes a much larger acoustic generalization gap than the short clean test.

Downloaded official [LibriSpeech test-other](https://openslr.org/12), CC BY 4.0,
with publisher MD5 verification, local SHA256, attribution and extraction filter.
scripts/prepare_speech_challenge.py fixes four duration-eligible random utterances
per speaker (seed127) after excluding every speaker in all previous raw partitions.
132 recordings, 33 new speakers, 827.80 seconds. Manifest and archive fingerprints
are in data/speech_challenge/provenance.json. This is a harder official subset and
different audiobook recordings, still the same English corpus family. Recording
conditions are not separately annotated; it is not conversational, multilingual,
or a measured accent/dialect coverage sample. This set is evaluation-only.

Compare two six-epoch continuations from exactly the same saved masked-normalization
checkpoint: all1,519 accepted training clips, identical tokenizer/manifests, seed839,
batch8, peak LR0.0003, warmup30, cosine floor0.000001; 1,140 updates each. Control
changes no features. Treatment applies one random frequency mask (0..8 of80 bins)
and one time mask (0..20 frames, also capped at10% of actual duration), filled with
the utterance mean. No time warping or changed transcript/timing lengths. This is
a bounded [SpecAugment-inspired](https://arxiv.org/abs/1904.08779) policy, not the
paper's complete policy. Separate RNG leaves shuffle/dropout randomness untouched;
masking occurs after deterministic cache loading, only in training, leaving padding
and caches unchanged. Default behavior remains unchanged. Different augmentation
cannot silently resume the previous optimizer. Original architecture/capacity and
feature representation remain fixed. Training loops: control341.31s, masking348.80s
on RTX3050; incidental CPU work means these are not controlled throughput estimates.

Best checkpoints use development CTC loss. Control loss1.2955, masking1.2390
(both epoch6). Development fusion WER ties, so lower CER explicitly breaks the tie
before candidate challenge/test inference; results/asr_mask_selection.json records
the choice. The installed decoder settings remain fixed: beam10, character LM
weight0.2/reward0.2, trained only on the same training transcripts.

All following cells are WER / CER percentages with that decoder:

| Evaluation | Installed quality model | Matched control | Feature masking |
|---|---:|---:|---:|
| Development80 | 81.23 / 35.96 | 80.71 / 35.52 | 80.71 / 34.88 |
| Earlier held-out80 | 81.31 / 34.95 | 80.36 / 34.55 | 81.40 / 34.92 |
| New-speaker test-other132 | 90.57 / 48.66 | 90.75 / 48.45 | 89.87 / 47.43 |
| Same132 + synthetic10dB white noise | 96.41 / 71.96 | 96.67 / 71.19 | 96.62 / 69.29 |

Synthetic noise uses fixed Gaussian draws, seed127+row, RMS scaled to the stated
SNR after existing waveform preprocessing. It is a controlled stress test, not
genuine environmental recordings, independent audio, or a training augmentation.
Masking improves noisy CER while slightly worsening WER versus installed weights;
fewer missing characters alone does not establish accurate words or meaning.

New-speaker word substitutions/deletions/insertions: installed1691/288/87,
control1685/289/96, masking1705/254/91. Masking primarily reduces deletions;
substitutions do not improve. Under noise: installed1145/1048/6 versus
masking1395/788/21, showing the tradeoff explicitly. Full character edit counts,
per-speaker errors and timings are in results/asr_*_other*.json. Masked fusion
processes827.80s audio in about32.34s CPU feature/acoustic+decode time (four threads,
cached features, some evaluations overlap; no speedup or production-load claim).

Speaker-cluster bootstrap (2,000 resamples, seed127) gives masking-minus-installed
WER difference -0.70 percentage points, percentile95% interval[-1.57,+0.18].
The interval includes no improvement; resampling does not account for training-seed
uncertainty or new-domain effects. Against the matched control, the difference is
-0.88 points with interval[-1.75,-0.05], supporting a conditional masking benefit
over this continuation, not superiority to installed weights across conditions.
Do not promote this modest single-seed result to robust overall gain.
The old clean-test WER regresses and the control's new-speaker WER worsens. Both
experimental checkpoints remain saved; installed package hash is unchanged.

scripts/analyze_speech_errors.py exposes deterministic minimum-edit alignments,
character confusions and whitespace errors. English lexical probes are opt-in:
negation forms, number words and a small list of name-spelling probes. These
measure exact aligned spelling, not negation scope, factual accuracy or verified
named entities. Installed new-speaker negation forms:6/31 exact matches,18
substitutions,7 deletions; masking7/31,20 substitutions,4 deletions. Number words
remain1/27 exact matches for both; only3 name-spelling probe occurrences exist,
far too few for a name accuracy claim. Installed space alignment has313 deletions
and515 substitutions involving spaces; masking289/499. Other languages require
appropriate segmentation, lexical profiles and native-script evaluation. Current
English tokenizer cannot emit digits; these number results concern spoken words.

The shared minimum-edit alignment and masking infrastructure accept other scripts
without English-specific corrections. Inference-only experiments do not establish
that masking preserves tonal or dialect information; validate before adopting it
for those languages. An independent exhaustive tiny CTC path-sum regression checks
blank/repeated-label prefix-search accounting. It passes, providing no evidence of
that basic decoder bug; narrow search and weak acoustic evidence still limit output.

Browser checks used the installed package and a fixed new-speaker recording:
“personally as you call it and all” → “pasmy as e cos and do”. The broader sample
also loses Margaret, Bessy and “don't” in one reference and “not” in another. No
fully correct new-speaker transcript is claimed. The new app behavior tracks
speech-derived input through edits and analysis, displays its uncertainty above
downstream results, and withholds automatic vocabulary cards. Clearing input
starts independent typed-text analysis and restores existing learning suggestions.
English→Spanish unavailability retains the speech warning. These flows were
actually exercised in Chrome; no cards saved or browser speech provider invoked.
Experimental checkpoints were evaluated offline, not substituted into the server.

176 Python tests, both Node suites and git diff --check pass. Original weights,
working translation/linguistic components and dependencies are preserved; no
commits or pushes. The evidence favors diversifying licensed acoustic supervision
and targeted robustness evaluation over repeated clean-benchmark continuation.
