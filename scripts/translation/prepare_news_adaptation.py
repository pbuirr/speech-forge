"""Prepare article-disjoint OPUS adaptation data without consuming existing tests."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.translation.train_nmt import load_corpus
from scripts.translation.train_local_translation import canonical


def article_partitions(records, excluded, seed=419, max_train=40000):
    """Union article IDs and repeated source/target text before splitting.

    Entire connected article groups touching excluded text are discarded, including
    sibling sentences. Thus the previous news evaluation remains untouched.
    """
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        parent[find(a)] = find(b)
    for doc, source, target in records:
        union(('doc', doc), ('s', canonical(source)))
        union(('doc', doc), ('t', canonical(target)))
    blocked = {find((side, value)) for side, value in excluded if (side, value) in parent}
    groups = {}
    seen = set()
    for doc, source, target in records:
        root = find(('doc', doc))
        if root in blocked or (source, target) in seen:
            continue
        seen.add((source, target))
        if not all(3 <= len(s.split()) <= 55 and len(s) <= 650 for s in (source, target)):
            continue
        if not .4 <= len(target.split()) / len(source.split()) <= 2.5:
            continue
        if any(mark in source + target for mark in ('http', '<', '&amp;', 'www.')):
            continue
        groups.setdefault(root, []).append((doc, source, target))
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    result = {'train': [], 'dev': [], 'test': []}
    for group in ordered:
        partition = 'dev' if len(result['dev']) < 1000 else 'test' if len(result['test']) < 1000 else 'train'
        if partition == 'train' and len(result['train']) >= max_train:
            continue
        result[partition].extend(group)
    return result, len(blocked)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--exclude-splits', type=Path, required=True)
    p.add_argument('--exclude-news', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    excluded = set()
    for path in [*(args.exclude_splits/f'{name}.tsv' for name in ['train','dev','test']), args.exclude_news]:
        for source, target in load_corpus(path):
            excluded.update([('s', canonical(source)), ('t', canonical(target))])
    with zipfile.ZipFile(args.archive) as archive:
        source_lines = archive.read('GlobalVoices.en-es.es').decode('utf-8').splitlines()
        target_lines = archive.read('GlobalVoices.en-es.en').decode('utf-8').splitlines()
        if len(source_lines) != len(target_lines):
            raise ValueError('Parallel line counts differ')
        # Moses files include blank article separators absent from the link list.
        aligned = [(s,t) for s,t in zip(source_lines,target_lines) if s.strip() or t.strip()]
        source, target = map(list, zip(*aligned))
        alignment = archive.read('GlobalVoices.en-es.xml').rstrip()
        # This release omits only the closing root tag; reject incomplete groups.
        if alignment.endswith(b'</linkGrp>'):
            alignment += b'\n</cesAlign>'
        tree = ET.fromstring(alignment)
        docs = [group.attrib['fromDoc'] + '|' + group.attrib['toDoc']
                for group in tree.iter('linkGrp') for _ in group.iter('link')]
        if not len(source) == len(target) == len(docs):
            raise ValueError('Alignment metadata does not match parallel line counts')
        for name in ['README','LICENSE']:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/f'GlobalVoices.{name}.txt').write_bytes(archive.read(name))
    records = [(d, s.strip().lower(), t.strip().lower()) for d,s,t in zip(docs,source,target)]
    partitions, blocked = article_partitions(records, excluded)
    for name, rows in partitions.items():
        (args.output/f'news_{name}.tsv').write_text(''.join(f'{s}\t{t}\n' for d,s,t in rows), encoding='utf-8')
    # Replay everyday sentences to measure/prevent catastrophic forgetting.
    everyday = load_corpus(args.exclude_splits/'train.tsv')
    random.Random(419).shuffle(everyday)
    training = [(s,t) for d,s,t in partitions['train']] + everyday[:20000]
    random.Random(419).shuffle(training)
    (args.output/'train.tsv').write_text(''.join(f'{s}\t{t}\n' for s,t in training), encoding='utf-8')
    for name in ['dev','test']:
        (args.output/f'{name}.tsv').write_bytes((args.output/f'news_{name}.tsv').read_bytes())
    report = {'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'policy':'article/source/target connected groups; exclude whole groups touching prior evaluation',
        'counts':{name:len(rows) for name,rows in partitions.items()}, 'replay':min(20000,len(everyday)),
        'blocked_groups':blocked, 'articles':{name:sorted({d for d,s,t in rows}) for name,rows in partitions.items()}}
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'articles'}))


if __name__ == '__main__':
    main()
