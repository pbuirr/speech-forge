# Speech recognition

English speech-to-text with a CNN + BiLSTM acoustic model trained with CTC, decoded with prefix beam search and a character n-gram language model. Everything is trained locally from the data below.

## Model

```
audio (16 kHz) → log-Mel (80 × T, 10 ms frames)
  → 3 conv blocks (Mel 80 → 10, time ÷2)
  → 3 bidirectional LSTM layers (256 per direction, residual + LayerNorm)
  → linear → characters (CTC)
```

- Code: `speech_engine/features.py`, `model_ctc.py`, `train.py`, `decode.py`, `lm.py`.
- Sizes: `compact` (0.8M parameters), `wide` (3.0M) and `full` (4.8M). The selected model is `full`.
- Output units are characters (a–z, space, apostrophe, hyphen), so there's no fixed word vocabulary.
- Decoding: beam width 10, character 5-gram stupid-backoff LM built from the training transcripts only, LM weight 0.2, per-character reward 0.2. On dev this beats greedy decoding by about 6 WER points (54.0% → 48.2%).

## Data

| Use | Data | Notes |
| --- | --- | --- |
| Training | LibriSpeech train-clean-100 + train-clean-360, Mini LibriSpeech, SLR83 dialect recordings | 133,192 recordings, 465 h; audiobook speech plus a small amount of UK/Irish regional speech |
| Dev (choosing) | 80 Mini LibriSpeech + 95 SLR83 recordings | 175 recordings; speakers never in training |
| Test (reporting) | 160 LibriSpeech test-clean recordings, 4 each from 40 speakers | frozen by hash before any candidate was scored |
| Challenge only | LibriSpeech test-other sample, 26 VoxPopuli parliamentary clips | never used for training or choosing |

Every split is speaker-disjoint. The test sample was picked by hashing file names, not by looking at model output, and its manifest hash is recorded in each report.

## Training

- Learning rate: linear warmup, then cosine decay (peak 1e-3).
- Each "cycle" draws a fixed number of utterances with replacement. Cycles aren't full passes over the data.
- Batches are bucketed by length to cut down on padding.
- SpecAugment: 2 frequency masks and 2 time masks, filled with the utterance mean.
- Speed perturbation at 0.9 / 1.0 / 1.1. The slowed and sped-up features are cached separately.
- Mixed precision is available on CUDA (`--amp`); the selected run didn't use it.
- Checkpoints save model, optimizer, sampler and RNG state, so a resumed run matches an uninterrupted one exactly (there's a test for this).

The selected run: 96 cycles × 4,166 utterances (about 400k draws, roughly 3 passes over the data), batch size 16, seed 839, on an RTX 3050 (8 GB). Those settings are also saved in the run's `evaluation.json`.

## What changed the results

1. **Training length.** The first model's dev loss was still dropping when the learning-rate schedule ended. Training the same model 4× longer took test WER from 47.6% to 30.3%.
2. **Augmentation needs capacity.** SpecAugment and speed perturbation made the 3.0M model worse on dev (55.8% → 58.1%) but helped the 4.8M model (54.5%). The smaller model was already short on capacity.
3. **More data.** Adding train-clean-360 (102 h → 465 h) with the 4.8M model gave 48.2% dev and 22.5% test WER.

The full table is in [RESULTS.md](../RESULTS.md).

## Limitations

- English read speech only. On regional speech the dev WER is 50% (Scottish), 66% (Northern English) and 78% (Irish). Each of those comes from a single speaker code, so they're really per-speaker numbers.
- Noisy, spontaneous and conversational speech hasn't been tested on real recordings.
- Every result comes from one training seed.
- The dev set is mostly regional speech, so dev WER runs much higher than test WER. It's for comparing models, not for reporting accuracy.

## Reproducing

```bash
python scripts/speech/acquire_librispeech.py --archive train-clean-100
python scripts/speech/acquire_librispeech.py --archive train-clean-360
python scripts/speech/download_speech_dialects.py
python scripts/speech/prepare_scaled_speech.py --splits train-clean-100 train-clean-360 --output data/speech_scale/460h
python scripts/speech/prepare_fresh_speech_test.py

python scripts/speech/train_real_speech.py --data data/speech_scale/460h \
    --metadata data/speech_scale/460h/metadata.jsonl --output data/training/asr_460_full_aug \
    --architecture full --epochs 96 --samples-per-epoch 4166 --batch-size 16 \
    --train-limit 200000 --eval-limit 200 --train-max-duration 18 --eval-max-duration 18 \
    --bucket-batches --spec-augment --speed-perturb --device cuda --skip-test

# choose decoding on dev, then score the frozen test set once
python scripts/speech/evaluate_speech_decoding.py --directory data/training/asr_460_full_aug \
    --split dev --lm-order 5 --mode compare --output results/asr_460_full_aug_dev_lm5.json
python scripts/speech/evaluate_speech_decoding.py --directory data/training/asr_460_full_aug \
    --manifest data/speech_scale/fresh_test/test.tsv --metadata data/speech_scale/fresh_test/metadata.jsonl \
    --lm-order 5 --mode lm02 --output results/asr_460_full_aug_fresh160.json
```

Run any script with `--help` for the full option list.
