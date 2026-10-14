# Real local model validation

Historical initial translation validation. The latest selected checkpoint,
independent news improvements, short-sentence regression and real acoustic pilot
are recorded in [Domain adaptation](DOMAIN_ADAPTATION.md); that report supersedes
the installed checkpoint hash and current-capability figures below.

Validated on 2026-10-11 using the original LinguaForge Transformer, locally learned
BPE and existing application. No pretrained weights, hosted inference or new dependencies.
The installed model is experimental Spanish→English, not a general multilingual translator.

## Data and training

[OPUS Tatoeba v2023-04-12](https://opus.nlpl.eu/Tatoeba.php), English/Spanish,
community translations, [Tatoeba attribution and licensing](https://en.wiki.tatoeba.org/articles/show/faq)
(CC BY 2.0 FR). Archive SHA256:
`a26e7675cae64c93ce45e6cae48f38424bf26e0923cec9d7881b1fa2bb636227`.
Filtered sampled snapshot: 39,849 unique accepted pairs, 2–20 whitespace words and
at most 200 characters per side. Sampling and filtering bias coverage toward short
sentences; this is not representative of all Spanish usage.

Connected groups of normalized source OR target text stay in one partition:
35,864 training, 1,992 development, 1,993 test, seed 731. This prevents exact
normalized text overlap, not all semantic paraphrase leakage. BPE vocabularies
are learned only from training. Existing statistical fallback also uses training only.
Saved partitions and hashes are in `data/training/es_en_masked/training.json`.

Model: 1,204,736 parameters, width 128, four attention heads, two layers, FF width
256. Initial 15 epochs followed by 35 warm-start epochs on identical partitions
with corrected source-padding masks. Best checkpoint selected by development
token-weighted NLL, 1.7371. RTX 3050 8GB; processes took about seven minutes total,
including BPE fitting but excluding downloads/evaluation. Additional training and
the mask fix are confounded; the gain cannot be attributed to the mask alone.
Warm-start preserves weights/vocabularies but creates a new optimizer. Training
state is saved; exact interrupted-run resume is not implemented.

## Measured quality and efficiency

Beam four selected against greedy on 200 development examples (32.25 versus
31.44 application BLEU). Initial model scored 23.70 on the same development sample.
Test data was not used for fitting or decoder selection.

| Evaluation | Examples | Application BLEU | Character F2 | Exact match |
|---|---:|---:|---:|---:|
| Held-out Tatoeba | 1,993 | 33.14 | 55.21 | 9.93% |
| Independent GlobalVoices news | 400 | 5.36 | 26.98 | 0.25% |

News source: [OPUS GlobalVoices 2018Q4](https://opus.nlpl.eu/GlobalVoices.php),
Global Voices/CASMACAT; observe [Global Voices attribution policy](https://globalvoices.org/about/global-voices-attribution-policy/)
and archive licensing. Evaluation excluded normalized overlap with all Tatoeba
partitions (none found among the sampled news candidates). Copy-source news baseline
was 1.15 BLEU. These are lowercase whitespace BLEU and custom no-space character
F2, not standardized SacreBLEU scores or percentages of semantic correctness.
Single references penalize valid alternatives. Hallucinated additions, names,
numbers and long passages remain weaknesses; review flags are uncalibrated heuristics.

Matched 200-development-example inference, four CPU threads, same installed
checkpoint and beam four: CPU median 10.95 ms/p95 22.24 ms; CUDA median 31.06 ms/
p95 57.09 ms. Excludes startup and browser/network work. CPU is the server default;
`--nmt-device cuda` or `auto` remains available. Reports preserve raw model outputs,
application postprocessing, references, hashes and latency in `results/nmt_*.json`.
Literal fallback suggestions are excluded from main translation scores.

## Actual application validation

After the user restarted the server, interacted with Chrome at localhost:8765:

- Explicit Spanish→English: `¿dónde está la estación de tren?` produced
  `Where is the train station?`, together with analysis/vocabulary panels and model limitations.
- `mi hermano menor está mirando la tele.` produced
  `My younger brother is watching tv.`. Removed a cleanup heuristic that previously
  deleted short final words such as TV, CD, sky and shy; covered by regression tests.
- Opened actual neural attention output. Replaced misleading certainty language
  with attention concentration and a warning that attention is not verified alignment
  or a probability of correctness.
- Turkish `Evlerimizden geldik.` retained analysis while translation clearly
  reported missing tr→en weights and listed only es→en as available. Automatic language
  detection incorrectly favored English with zero evidence; selected Turkish controlled
  analysis, and the disagreement/weak-evidence warnings were visible. Detection remains weak.
- Local CTC transcription remained disabled with accurate missing checkpoint/tokenizer
  requirements. No real speech transcription or WER/CER measurement was completed.

142 Python tests and JavaScript rendering/async-flow/local-speech checks pass.
No new flashcards were saved during this validation; no claim of new learning quality
or speech accuracy. HTTP tests additionally cover source-token limits, model metadata,
training-only fallback and existing routing. Source masking has padding-invariance tests.

## Restore and reproduce

Installed checkpoint: `data/models/nmt_es_en.pt`, 5,134,443 bytes, SHA256
`7be111d843035f82af6057e6e24eb910304b3acc0ab1e22122afd000396c9618`.
Training checkpoints/corpora are local ignored artifacts, not committed binaries.
Saved partition hashes bind the experiment; replay uses saved splits because the
original sampled input was reconstructed from accepted partitions after an independent
download overwrote its filename. Downloader now supports explicit `--output`.

```powershell
python -X utf8 -u -B scripts/translation/train_local_translation.py --corpus data/corpora/es_en_real.tsv --splits-dir data/training/es_en_masked --output-dir data/training/es_en_replay --epochs 50 --dataset "OPUS Tatoeba v2023-04-12; CC BY 2.0 FR"
python -X utf8 -u -B scripts/translation/evaluate_local_translation.py --checkpoint data/models/nmt_es_en.pt --pairs data/training/es_en_masked/test.tsv --output results/nmt_recheck.json --device cpu
python -u -B -m webapp.server --port 8765
```

The fresh 50-epoch command reuses the data but is not an exact replay of the two
optimizer schedules. For the original warm-start stage, use `--load
data/training/es_en_local/best.pt --epochs 35`. CUDA numerical determinism is not promised.
ASR currently has only synthetic tone demo data and no trained acoustic checkpoint.
Its English-oriented character normalization and speaker-disjoint real-data evaluation
need work before training or claiming multilingual recognition. Existing demo use of
LibriSpeech test-clean for training must not be treated as held-out quality evaluation.
