# Speech capacity, supervision and language routing

2026-10-12. Training/development comparison complete; paused for external review.
No new quality release or independent candidate validation is claimed here.

The 6.56-hour supervision pool is too small to establish broad recognition.
A matched 24-epoch compact model trained from scratch reaches training CTC loss
0.65, but development loss bottoms at 1.588 in epoch9 and then worsens. Its full
6,576-update loop takes about27 minutes. More optimization on these voices alone
does not close the generalization gap. Previous installed models are preserved.

The training runner now supports original compact/full sizes and a wide variant:
compact CNN16/32/64, BiLSTM256x2, 2,967,809 parameters. Optional fourfold temporal
subsampling reduces recurrent sequence length while retaining the same log-Mel
representation. Old checkpoints default to twofold subsampling. Loading weights
with the wrong stride is rejected, even though parameter shapes could match.
Padding masks and CTC frame lengths are tested for odd/even lengths; all existing
train/dev/test references fit the fourfold alignment bound including repeated
characters. [Conformer](https://www.isca-archive.org/interspeech_2020/gulati20_interspeech.pdf)
also uses fourfold convolutional subsampling; this does not make our model a
Conformer or predict its published accuracy. Attention architectures remain
worth evaluating with adequate supervision, rather than adopting paper scores.

The completed small-data wide4 arm has development WER89.90%, CER45.46%,
versus compact2 WER86.49%, CER42.43%, using the same training-only text model.
Its training loss falls to0.458 but its best development loss is1.954 (epoch8).
Median output-frame/required-character-alignment slack is1.77 at stride4 versus
3.54 at stride2. This is a necessary-alignment diagnostic, not proof of causation:
width and stride changed together. The expanded-data capacity comparison therefore
keeps stride2 and compares compact versus wide with the same supervision and budget.

Untrained resource probe, one11.09s training recording, precomputed features,
five warm CPU runs/four threads with concurrent work: compact median14.31ms,
wide4 median15.22ms, original full4 median24.52ms. This measures acoustic-forward
cost only, not decoding, cold latency, recognition quality or streaming.

[LibriSpeech](https://openslr.org/12/) train-clean-100 acquisition uses official
sources/mirrors and publisher MD5 verification plus local SHA256. Extraction
rejects non-file entries/path traversal and caps declared archive size. Its audio
is English audiobook speech, not a replacement for conversational/noisy or
multilingual supervision. Preparation preserves licensing/source metadata,
deduplicates MiniLibriSpeech utterances and reserves evaluation/challenge speakers.
Lexical normalization losses and impossible fourfold CTC alignments are excluded
explicitly. Existing development/test partitions remain fixed.

Preparation completed:29,211 training recordings,101.8156 hours,255 declared
speaker identities;1,519 MiniLibriSpeech duplicates removed, four duration
exclusions. Existing dev175/test169 accepted manifest hashes match the controls.
The archive publisher MD5 is2a93770f6d5c6c964bc36631d331a522; local SHA256 is
d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2.

A new test-clean sample is frozen before candidate evaluation:160 utterances from
40 speakers,0.3228h, four hash-ranked eligible recordings per speaker. Selection
does not inspect model results. Its manifest SHA256 is
2217c2dff84076258b2d31b1bb24013a0c6d5b333fed6c3128e343ae6efe3adf.

The expanded compact2 run is complete: greedy dev WER 81.56%, CER 37.07%;
fixed small-text trigram fusion WER 79.59%, CER 35.58%. The matched small-data
compact control has greedy WER 89.16% and fused WER 86.49%. The expanded loop
took 3,528.9 seconds; resource contention varied, so wall time is not an isolated
hardware benchmark. There are 52,608 replacement draws, only 1.80 pool exposures,
covering 24,368 distinct recordings (83.42%). Irish English receives only 149
draws. These sampled cycles must not be described as 24 full 102-hour passes.

A character-context ablation reuses the shared backoff model with optional order
five; legacy order three is unchanged. With compact weights and the same small
text training set, dev WER falls from 79.59% to 74.83%; using all expanded training
transcripts gives 73.84%, CER 34.43%. Expanding trigram text alone gives 79.67%.
Backoff scores remain unnormalized rankings; there is no forced lexicon or grammar
correction. [Brants et al.](https://aclanthology.org/D07-1090/) motivates efficient
longer-context backoff; our character-fusion result is a separate measured result,
not a reproduction of their translation experiments. Decoder order is bound to
matching development/test reports and packaged inference configuration.

The final expanded wide2 model (2,967,809 parameters) has greedy dev WER 76.55%,
CER 33.08%; fixed-small-text trigram fusion gives WER 73.76%, CER 31.76%.
Expanded-text order-five fusion gives WER 68.38%, CER 30.43%, versus compact2
73.84%/34.43% under the same supervision, update budget and decoder. Word errors
are 1,362 substitutions, 158 deletions and 145 insertions over 2,435 reference
words. Regional dev WER remains 81.39%, including Irish 91.90%. Warm CPU pipeline
RTF is 0.0548 on this run (four threads, concurrent evaluation); cold loading and
LM construction are excluded. Selection is recorded in
`results/asr_scale_selection.json` before candidate independent inference.
Checkpoint SHA256 is
5bb993e76118d4bb6aa0422e22b59462bc85f1063173b0ca6e633e83ca359fb2.
One training seed does not establish robust architecture superiority. Existing
models remain installed/preserved. Fresh/parliament candidate evaluation and new
model browser testing are pending.

A separate, frozen [VoxPopuli](https://github.com/facebookresearch/voxpopuli)
parliamentary challenge contains 26 gold-transcribed clips from 26 published
speaker IDs, 15 accent categories, three 2019 sessions, 0.0596 hours. Annotation
selection precedes candidate inference and ignores the publisher's decoded/WER
columns. Forty-eight invalid/overlapping boundaries and 76 duration failures are
excluded from the 2019 candidate pool, with no guessed repairs. Own HTTP range
indexing retrieves 596 MB of selected sessions from the 6.96 GB uncompressed tar;
local member checksums and archive identity are retained, without claiming an
unavailable whole-archive publisher checksum. Data is publisher-licensed CC0.
This tiny, session-clustered challenge is not a representative accent benchmark
and is never used for training or model/decoder selection.

## Research alternatives

[MLS](https://www.openslr.org/94/) provides CC BY 4.0 audiobook speech in eight
languages. It can add multilingual supervision, but audiobook expansion alone
does not resolve conversational or recording-domain mismatch.
[Google FLEURS](https://huggingface.co/datasets/google/fleurs/blob/main/README.md)
provides CC BY 4.0 parallel read speech across 102 languages, around ten training
hours per language, and publisher-defined speaker-separated splits. It is useful
for multilingual acoustic language-ID evaluation in comparable recording settings.
Its listed fields do not expose individual speaker IDs: preserve the publisher's
splits and distinguish that assertion from independently audited speaker identities;
do not invent speaker IDs from utterance IDs. Neither corpus is acquired/trained
here, and neither establishes noisy conversational recognition.

The Conformer paper's small model uses about10M parameters, 960h supervision,
wordpiece/transducer modeling and a substantially different training recipe.
Its results justify comparing an own attention/convolution encoder when resources
allow; they do not justify transferring paper WER expectations to our character
CTC, copying a stride without evaluation, or using third-party pretrained weights.

New CTC checkpoints also preserve Torch/Python RNG and explicit sampler/loader
generator state for resume. A CPU uninterrupted-versus-resumed dropout/sampling
regression produces identical parameters. Old checkpoints still load with an
explicit legacy-resume note; they cannot recover random state never saved.
This does not promise deterministic CUDA execution or identical results after
changing the training schedule, data or device.

## Connected uncertainty

On169 existing English held-out records, text identification is wrong on11
references and36 generated transcripts;13 wrong transcripts have uncalibrated
scores at least0.65. Heuristic character/word-marker rankings on distorted text
cannot identify the original spoken language. This is an English-conditioned
diagnostic, not multilingual acoustic language-ID evaluation.

Shared analysis now uses an explicitly selected language, otherwise the declared
generated source language. Generated input without either requests a language
selection instead of silently routing via distorted text. Detection retains its
rankings for inspection but labels scope and disallows automatic generated-text
routing. UI labels transcript/text hypotheses and withholds the misleading
percentage display for generated inputs. This does not certify configured language
or fix the detector's underlying statistical accuracy. Source metadata remains
declared provenance, not authentication.

Local inference records the actual loaded checkpoint bytes' SHA256; identity
flows through canonical input evidence. The interface names the checkpoint and
explicitly labels recording duration. Core/API learning protections remain in
place. New frontend language-label behavior was actually browser-tested on the
fixed Scottish WAV; final backend/checkpoint test will follow evaluation.

Translation output also retains one canonical original generated-source record,
including the originating speech checkpoint identity. Reusing translated text
does not accumulate unbounded provenance or promote predictions to verified text.
Actual English ASR and Spanish-to-English NMT currently lack a compatible trained
cross-language speech pair; API lineage regressions do not establish Spanish ASR.

207 Python tests and three Node suites pass.
Reports live under
results/asr_capacity_* and results/asr_transcript_language.json. No pretrained
weights, hosted inference, automatic commits or pushes.

CTC fusion now retains language-model scores only for surviving beam prefixes.
An exhaustive four-frame weighted-path oracle passes for character orders 3 and 5.
On a fixed 400-frame synthetic-logit probe, output is identical and peak traced
Python heap falls from 874,872 to 78,992 bytes. This excludes acoustic inference
and LM construction; it is a memory regression probe, not an accuracy result.
See `results/asr_beam_cache_probe.json`.
