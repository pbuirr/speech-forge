# Corpus-backed analysis

The morphology pipeline accepts corpus-attested lemmas, UD POS tags, and
morphological features. It retains the source URL, declared license, training
checksum, and up to three sentence/token/line references per interpretation.
These annotations are evidence of observed uses, not verified analyses of every
new sentence. BPE pieces stay separate; a lemma does not imply morpheme boundaries.

The importer follows the lexical fields in the [CoNLL-U specification](https://universaldependencies.org/format.html).
It checks columns, consecutive word IDs, UPOS tags, and feature syntax; it skips
multiword-token wrapper rows and empty nodes. It is not a dependency-tree validator
or a tokenizer. Unknown lemmas and explicitly foreign-language tokens are excluded
from the lexical index. Case is preserved and lookup normalizes Unicode to NFC.

## Reproduce the English resource

From the repository root:

```bash
python scripts/analysis/prepare_english_resource.py
python webapp/server.py
```

Preparation downloads the public English EWT train/test split from pinned commit
`4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`, checks its corpus SHA-256 values, builds
`data/linguistics/en.json`, preserves the upstream README/license, and writes
`results/linguistic_eval_ewt.json`. The compiler consumes only training annotations.
The app loads local compiled resources before serving requests and performs all
analysis offline. Compiled JSON and downloaded corpora are ignored by Git.

The annotations/database are CC BY-SA 4.0, with underlying text rights described
in the upstream README. Attribution and terms are retained alongside the resource.
Review source terms before redistributing derived resources.

## Import another annotated corpus

```bash
python scripts/analysis/import_linguistic_corpus.py --input train.conllu --lang es --source-url https://example.org/dataset --license DATASET-LICENSE --output data/linguistics/es.json
python scripts/analysis/evaluate_linguistic_resource.py --resource data/linguistics/es.json --test test.conllu --context --output results/linguistic_eval_es.json
```

Supply the actual source and license. Import support does not establish language
coverage: use separate training and evaluation splits. Same-file evaluation is
rejected by checksum; different checksums alone cannot detect sentence overlap.
There is one resource per base language; duplicate language files fail visibly.
Resources load atomically. Dialect-specific profiles and merging corpora are future work.

## How decisions work

The exact-form index keeps all distinct lemma/POS/feature interpretations and
their observation counts. A language-specific bigram HMM uses training emissions
and transitions; Viterbi decoding considers the complete token sequence, resetting
at sentence-final punctuation. It can select a candidate POS, but cannot distinguish
two meanings with the same POS. Unknown words receive statistical/heuristic POS
predictions; they do not acquire invented corpus lemmas.

Fields common to every remaining candidate are preserved. Disputed fields remain
unresolved. Selecting alternatives by sentence context is labeled `corpus_context`,
with `predicted` lemma/POS status when those fields depended on the selection.
All original candidates remain in API output. `confidence` stays null because the
model is not calibrated. Generated flashcards retain the analysis/provenance and
require review; existing manually authored cards remain compatible.

Text analyses preserve original Unicode code-point offsets in half-open `span`
intervals. Combining marks stay with their bases. `analyze_tokens` accepts upstream
or gold word boundaries and leaves offsets null. Chinese/Japanese raw-text
tokenization remains character-oriented rather than a linguistic word segmenter.

## Measured English results

The pinned EWT test split contains 25,079 eligible word tokens after excluding
missing lemmas. Using gold word/sentence boundaries:

| Measure | Result |
|---|---:|
| Previous heuristic POS accuracy | 68.5% |
| Contextual POS accuracy | 90.2% |
| Lemma decision coverage | 82.7% |
| Lemma accuracy on decisions | 98.3% |
| Heuristic lemma accuracy on those same decisions | 68.9% |

These are one English treebank evaluation, not multilingual, semantic, or raw-text
accuracy claims. Abstentions stay in coverage denominators. Corpus lookup alone
finds candidates for 90.9% of test words, while complete unambiguous analyses cover
30.5%. The report includes exact-match metrics, checksums, and sample failures.
The contextual model can still make grammatical mistakes; deeper syntax and
context-sensitive word senses are not implemented by this change.
# Raw-text evaluation

Run `python scripts/analysis/evaluate_linguistic_resource.py --resource data/linguistics/en.json --test PATH_TO_HELD_OUT_CONLLU --raw-text --output results/raw_text_eval_ewt.json` to measure the production analyzer on original sentence text. Sentence boundaries remain supplied; word boundaries do not. Exact token-span precision, recall, F1, and UPOS accuracy over all gold words are reported. Incorrect boundaries count as POS errors. Missing original text or non-concatenative word expansions are explicitly excluded, with counts. Unicode NFC and whitespace are normalized only for scoring, not production offsets.

On the pinned EWT test split, 2,077 sentences and 25,094 words aligned without exclusions. English clitic rules improved token F1 from 91.25% to 92.89% and raw-text POS accuracy from 85.62% to 87.75%. These are before/after measurements on the same test corpus, not evidence of generalization across domains or languages. The earlier gold-word POS result measures a different, easier input condition. The raw-text evaluation includes all annotated POS labels, including words with missing lemmas.

English tokenization splits suffixes `n't`, `'s`, `'re`, `'ve`, `'ll`, `'d`, and `'m` (including curly apostrophes), preserving surface slices such as `ca` + `n't`. This is a deterministic syntactic-word convention, not proof of a morphological or semantic interpretation. Other languages retain their existing tokenization. Internal ZWNJ and ZWJ followed by a letter or mark are preserved for scripts that use them. URLs, abbreviations, compound conventions, and sentence segmentation still need work.

