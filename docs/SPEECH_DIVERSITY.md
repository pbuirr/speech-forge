# Speech supervision and connected evidence

2026-10-12. A bounded real-data experiment, not a dependable recognition release.
All acoustic/translation weights are our locally trained implementations. No new
dependencies, pretrained weights, hosted inference, commits or pushes.

## Choice of improvement

The previous masking experiment did not close the gap to unseen recordings.
The current acoustic model has807,553 parameters and5.31 hours of audiobook
supervision. Its sampled training WER was already poor, so neither generalization
nor model capacity is established as sufficient. More epochs on the same voices,
character-LM tweaks or grammar-based correction cannot substitute for acoustic
evidence that was never learned.

[Conformer](https://arxiv.org/pdf/2005.08100) combines convolution and attention;
its reported small model has10M parameters and is evaluated with a roughly970-hour
LibriSpeech corpus. Those results are not predictions for our6.56-hour pilot.
[Deep Speech2](https://proceedings.mlr.press/v48/amodei16.html) also demonstrates
that end-to-end architectures can handle English, Mandarin, accents and noise;
it does not establish that our reduced configuration has enough capacity/data.
The original larger CNN/BiLSTM defaults remain available:4,779,233 parameters,
19.12MB of float32 parameter storage versus3.23MB for the current model. This is
resource inspection only, not an accuracy, GPU-memory or runtime comparison.
The existing Transformer implementation is likewise not a verified better model.

| Alternative | Practical assessment |
|---|---|
| More diverse labeled speech | Directly addresses missing voices, phonetic contexts and domains; selected for a controlled pilot |
| Larger original CNN/BiLSTM or own Conformer | Plausible capacity improvement; needs matched quality/cost experiments, not architectural replacement by assumption |
| Different acoustic scaling/features | Worth testing for channel robustness; current log-Mel settings remain fixed here to isolate supervision |
| Decoding with new transcript statistics | Cheap ablation measured below; cannot recover unsupported acoustic details reliably |
| Multilingual corpus expansion | [FLEURS](https://huggingface.co/datasets/google/fleurs) is a candidate; no claim of acquisition or trained multilingual speech |

## Audited dialect supervision

[SLR83](https://openslr.org/83), attributed to Demirsahin et al./Google,
CC BY-SA4.0, contains accent-elicitation scripts and regional English recordings.
[The paper](https://aclanthology.org/2020.lrec-1.804/) describes phonological
coverage, names and local pronunciations. Acquired only Irish male, Northern
female and Scottish female archives: HTTPS origin, ZIP CRC, recorded local SHA256;
no publisher checksum was available at the inspected resource. Original license,
index and references remain in data/speech_dialects.

Explicit metadata records corpus, language, dialect, speaker policy, prompt,
raw/prepared text, source and license. Anonymous numeric speaker codes are
conservatively joined across regional/gender prefixes rather than automatically
counting every prefix as a new human. Codes04310 and03397 are reserved for
development/test; other codes train. The human identities cannot be independently
verified. Development/test each have one conservative code, spanning three
regional recording groups. The small identity pool limits broad speaker claims;
gender, region and individual voice are confounded in this subset.

SLR83 prompt IDs use fixed hashed train/dev/test buckets; no prompt-ID or exact
prepared-reference overlap across those partitions. No recordings are truncated
while retaining complete transcripts.52 references are excluded because their
letters/digits would be lost by the existing English tokenizer; no invented
digit-to-spoken-word expansion is applied. This preserves the original vocabulary
and makes its numeric/native-script limitations explicit.
The same lexical-loss rejection now applies in the shared ASRDataset, so other
ingestion paths cannot silently discard reference numerals or unsupported letters.

Added676 training clips,1.2475h: Irish92, Northern254, Scottish330. Total available
training2195 clips/6.5582h. Fixed new development95 clips and test89, plus the
unchanged80/80 legacy partitions. LibriSpeech still uses utterance prompt IDs;
the unseen-prompt claim applies to SLR83 only. Metadata, accepted manifests and
tokenizer hashes are checkpoint-bound. The generic metadata and balancing helpers
accept other languages/scripts; this training runner explicitly remains an English
pilot and rejects silently mislabeling other-language corpora as English.

## Controlled alternatives

Same locally trained acoustic warm-start, optimizer, seed839, original architecture,
80-bin log-Mel features, batch8, peak LR0.0004/warmup30/cosine floor0.000001.
Eight epochs,190 updates per epoch:1,520 optimizer updates for each strategy.
Uniform replacement sampling uses a separate RNG. Equal updates do not imply
equal audio hours or unique recordings per epoch.

Control trains only on the existing MiniLibriSpeech; pooled training adds SLR83
with uniform recording sampling. Balanced training gives equal mass at each
corpus→language→dialect→speaker level, then divides mass among recordings.
It prevents small groups disappearing in pooled data, not a universally optimal
weighting rule. Here MiniLibriSpeech gets half; each of three SLR83 regions gets
one sixth. Available data and decoder LM are identical for pooled/balanced arms.
Best epochs by the same development CTC-loss policy are1/8/6 respectively;
selected checkpoints therefore have different numbers of executed updates despite
equal completed training budgets. Training loops410.47/389.07/369.44s on RTX3050;
cache warm-up and overlapping CPU work preclude causal speedup claims.

Development175, training-only character fusion, WER/CER percentages:

| Strategy | Pooled WER/CER | Legacy WER | SLR83 WER |
|---|---:|---:|---:|
| Installed acoustic/LM | 88.99 / 44.48 | 81.23 | 96.01 |
| Matched continuation control | 88.67 / 44.68 | 81.49 | 95.15 |
| Diverse pooled training | 87.27 / 42.01 | 81.31 | 92.65 |
| Diverse balanced training | 87.19 / 41.43 | 82.09 | 91.79 |

Installed acoustics with only the new LM gives SLR83 WER95.86: almost unchanged.
Diverse pooled acoustics with the fixed original LM gives92.81. Thus the measured
gain is mainly acoustic rather than new LM text alone. Greedy results are also
reported. No beam/LM parameter sweep: beam10, weight0.2, character reward0.2.
LMs use training transcripts only; no dictionary or grammar correction.
Balanced is selected by pooled development WER before candidate test inference;
results/asr_diversity_selection.json records the decision. Irish development
remains near99% WER; averages cannot certify accent-wide improvement.

Independent evaluations after selection, WER/CER percentages:

| Evaluation | Installed | Matched control | Balanced candidate |
|---|---:|---:|---:|
| New corpus, unseen SLR83 prompts89 | 90.23 / 45.42 | 89.67 / 45.27 | 84.83 / 39.69 |
| Existing clean held-out80 | 81.31 / 34.95 | 82.00 / 35.92 | 80.36 / 34.83 |
| Previously fixed harder LibriSpeech132 | 90.57 / 48.66 | not repeated | 89.83 / 48.34 |
| Same132, synthetic10dB noise | 96.41 / 71.96 | not repeated | 96.36 / 70.69 |

The harder LibriSpeech challenge has33 speakers absent from original training;
SLR83 is a genuinely different corpus, still elicited/read English. Synthetic white
noise is a controlled perturbation, not independent environmental recordings.
Recognition remains poor. No spontaneous speech, non-native-accent, tonal-language,
or multilingual recognition accuracy claim follows from these results.

Combined169 test word substitutions/deletions/insertions change from1728/236/116
to1637/285/79: fewer substitutions and insertions, more deletions. Full per-corpus
character/space/English lexical-probe diagnostics and paired speaker-cluster
resampling are in results/asr_diversity_test_errors.json. The small conservative
SLR83 identity pool does not support a robust multi-speaker confidence claim.
Candidate inference processes924.55s audio in about30.99s on four CPU threads,
with feature caches and some concurrent evaluations; not streaming/load validation.

Separate experimental package: data/models/asr_en_diversity.pt, matching tokenizer
and evaluation siblings. Source SHA151817964f66a4dde2ca9a71fe3e63da90dbedfb5c66c49997045343ff9f289a;
package SHA f19bde5e745f655d63a1c843f5424af5451bc7c5743d4bdbae68abd792b91203.
Installed asr_en_quality.pt remains unchanged. Packaging binds the actual evaluated
LM manifest, including fixed-LM ablations, rather than silently substituting data.

## Whole-platform input fidelity

Generated input now carries canonical input_evidence through shared understanding,
morphology, grammar, syntax serialization, detection and translation. Client
confidence/verified flags cannot certify generated input. Local translations also
declare predicted output evidence. Core learning extraction and automatic-card
HTTP endpoints reject declared generated input before persistence; this is no
longer only a frontend save-button policy. Provenance is declared source metadata,
not an authentication mechanism or a way to identify externally pasted model text.

Browser testing reproduced “El gato duerme.”→“The cat doubt.” in the existing local
NMT. Swap now carries model-output uncertainty into analysis and withholds learning
suggestions. Fixed a reproduced ordering bug: capture result visibility before
language pickers hide it, and never copy a hidden stale translation over newly
edited text. Retested real translation/swap/reanalysis and fresh-text preservation;
unsupported reverse translation remains explicit. No cards or browser-provider
audio requests were made. The mistranslation was not heuristically corrected.

184 Python tests and three Node suites pass, including corpus leakage, lexical
normalization loss, balanced sampling, canonical source evidence, core/API learning
guards, decoder-data packaging and swap behavior. Direct packaged PCM inference
is verified on one fixed test recording per region; outputs remain poor.
After restarting with the diversity filenames, the browser uploaded all three
fixed held-out regional WAVs. Each transcript exactly matches direct packaged PCM
inference. First and third transcripts were analyzed; English remains selected,
same-language results explicitly show original text without correction, and the
review warning persists. The Scottish transcript incorrectly triggers Scottish
Gaelic text detection (uncalibrated 70%); analysis still uses selected English and
shows the disagreement. Expanded vocabulary withholds automatic learning cards.
Fresh transcription clears previous analysis. No cards were saved, no browser
speech provider invoked, and no console errors appeared. Displayed 7.77/9.47/7.00
seconds are recording durations, not inference timings. These checks verify the
workflow, not usable transcription quality, streaming or multilingual speech.
