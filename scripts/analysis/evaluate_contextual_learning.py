"""Measure source/evidence retention, not linguistic accuracy or learning gains."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.flashcard import FlashcardDeck
from speech_engine.morphology import MorphologyEngine
from speech_engine.grammar import GrammarChecker
from speech_engine.language_detector import LanguageDetector
from speech_engine.understanding import LinguisticPipeline
from speech_engine.exercise import generate_fill_in_blank
from speech_engine.text_matching import answer_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--resources', required=True)
    parser.add_argument('--lang', required=True)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--output', required=True)
    parser.add_argument('--with-syntax', action='store_true', help='Load optional original local dependency models')
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError('limit must be positive')
    raw = Path(args.corpus).read_bytes()
    texts = re.findall(r'^# text = (.*)$', raw.decode('utf-8'), re.M)[:args.limit]
    engine = MorphologyEngine(); engine.load_resources(args.resources)
    if args.with_syntax:
        from speech_engine.dependency import load_dependency_models
        load_dependency_models(engine, Path(args.resources) / 'syntax')
    checker = GrammarChecker(); pipeline = LinguisticPipeline(engine, checker, LanguageDetector())
    counts = dict(texts=len(texts), cards=0, retained_occurrences=0, legacy_retained_occurrences=0,
                  distinct_analyses_lost_by_legacy_dedup=0, source_slice_checks=0,
                  source_slice_failures=0, cloze_span_failures=0, cards_with_rule_findings=0)
    counts.update(syntax_arcs=0, syntax_span_failures=0, cards_with_syntax=0, skipped_syntax_sentences=0)
    start = perf_counter()
    for text in texts:
        result = pipeline.analyze(text, args.lang)
        counts['skipped_syntax_sentences'] += len(result.syntax.get('skipped_sentences', []))
        for arc in result.syntax['arcs']:
            counts['syntax_arcs'] += 1
            left, right = arc['span']
            counts['syntax_span_failures'] += text[left:right] != result.morphology.analyses[arc['token_index']].surface
        cards = FlashcardDeck().extract_from_understanding(result, engine, checker)
        legacy_forms = {answer_key(c.front, args.lang) for c in cards}
        counts['cards'] += len(cards)
        counts['legacy_retained_occurrences'] += len(legacy_forms)
        counts['distinct_analyses_lost_by_legacy_dedup'] += len(cards) - len(legacy_forms)
        for card in cards:
            counts['cards_with_rule_findings'] += bool(card.analysis['context']['grammar_findings'])
            counts['cards_with_syntax'] += bool(card.analysis['context']['syntax']['arcs'])
            exercise = generate_fill_in_blank(card, engine)
            counts['cloze_span_failures'] += exercise.span != tuple(card.analysis['span'])
            for occurrence in card.analysis['occurrences']:
                counts['retained_occurrences'] += 1
                counts['source_slice_checks'] += 1
                left, right = occurrence['span']
                counts['source_slice_failures'] += text[left:right] != occurrence['surface']
    report = {'corpus_sha256': sha256(raw).hexdigest(), 'language': args.lang,
              'metrics': counts, 'seconds': perf_counter() - start,
              'limits': 'First N original text records. Measures evidence retention and offsets, not correctness of POS, lemmas, grammar findings, meanings, or student learning. Legacy counts emulate first-form-only deduplication on identical analyses.'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
