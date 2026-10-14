# Translation

Spanish → English with a Transformer encoder–decoder over BPE subwords. The architecture, BPE and decoding are in `speech_engine/nmt.py`, `bpe.py` and `attention.py`; training and evaluation scripts are in `scripts/translation/`.

## Models

| | Older model (`nmt_es_en_v3`) | Installed model |
| --- | --- | --- |
| Size | ~7.5M parameters | 1.2M parameters (d_model 128, 2 + 2 layers, 4 heads) |
| BPE vocabulary | 4k | 2k |
| Training data | ~965k pairs: Europarl, OpenSubtitles, Tatoeba | ~60k pairs: Tatoeba, GlobalVoices news |
| Clean news BLEU | **27.7** | 16.2 |
| Clean News Commentary BLEU | **23.8** | 9.2 |

Decoding is beam search (width 4) with a length penalty, a minimum/maximum output length relative to the source, an EOS boost and a 3-gram repeat block. Numbers, names and URLs can be swapped for placeholders before translation and put back afterwards (`speech_engine/ner.py`). When the model's output fails coverage or entropy checks, the app also shows a word-by-word gloss from IBM Model 1 trained on the same corpus (`lexical_fallback.py`).

The older, larger model is clearly better, but it isn't installed yet: on short constructed sentences it sometimes keeps going after the sentence is done ("…to pedro pedro"). The plan is a short fine-tune on clean current data, a fresh short-sentence test, then install it.

## Evaluation

- **BLEU** here is lowercase, whitespace-tokenised, single-reference corpus BLEU from `speech_engine/bleu.py`. **char-F2** is character 6-gram F-score with spaces removed. Neither is directly comparable with published sacreBLEU numbers.
- **Overlap audit.** Before comparing the two models, `check_bitext_overlap.py` checked every test set against the older model's 965k training pairs (exact match after normalisation, plus word 8-gram near-duplicates). The older model had seen 1,984 of the 1,993 sentences in my Tatoeba test, so Tatoeba can't be used to compare it at all. News had 2 overlapping pairs out of 1,040 and News Commentary 63 out of 45,829; those were removed, and 400 sentences were sampled from what was left.
- Decoding settings are chosen on news dev. Test sets are only used for the final comparison.

## Limitations

- Sentence-level only, with no document context.
- Names and rare words are often mangled, and lowercasing in the data loader loses capitalisation.
- Only es→en has evaluated weights. The multilingual ES/FR/DE→EN trainer (`train_multilingual.py`) exists but hasn't been evaluated under the same overlap-checked protocol.

## Reproducing

```bash
python scripts/translation/download_parallel_corpus.py --src es --tgt en
python scripts/translation/train_local_translation.py --corpus data/corpora/es_en_real.tsv --output-dir data/training/es_en_local --epochs 50

python -X utf8 scripts/translation/check_bitext_overlap.py \
    --training data/corpora/es_en_filtered.tsv \
    --test news=data/training/es_en_news/test.tsv \
    --test newscommentary=data/corpora/es_en_newscommentary_independent.tsv \
    --output-dir results/legacy_nmt_overlap_full

python scripts/translation/evaluate_local_translation.py --checkpoint data/models/nmt_es_en_v3.pt \
    --pairs results/legacy_nmt_overlap_full/news_clean.tsv --limit 400 --output results/nmt_legacy_v3_news_clean.json

python scripts/translation/install_local_translation.py --checkpoint <checkpoint> \
    --destination data/models/nmt_es_en.pt --reports <dev and test reports>
```
