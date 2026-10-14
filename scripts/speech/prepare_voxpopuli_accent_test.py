"""Freeze a small gold-transcribed parliamentary challenge without downloading a year.

Own data adapter; no publisher code or model weights are imported. HTTP ranges
read an uncompressed tar index and only selected session audio. Selection uses
annotations, never ASR hypotheses or the publisher's decoded/quality columns.
"""
import argparse
import ast
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import urllib.request

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speech_engine.corpus import lexical_normalization_loss
from speech_engine.audio import normalize_amplitude
from speech_engine.tokenizer import CharTokenizer, normalize_text

BASE = 'https://dl.fbaipublicfiles.com/voxpopuli'
SOURCE = 'https://github.com/facebookresearch/voxpopuli'


def rank(value):
    return hashlib.sha256(('LinguaForge-parliament-142:' + value).encode()).hexdigest()


def eligible_rows(rows, year, exclusions=None):
    tokenizer = CharTokenizer()
    accepted = []
    exclusions = exclusions if exclusions is not None else defaultdict(int)
    for row in rows:
        if (not row['session_id'].startswith(str(year)) or row['split'] != 'test'
                or row['is_gold_transcript'] != 'True'):
            continue
        spans = ast.literal_eval(row['vad'])
        if (not spans or any(not (math.isfinite(float(start)) and math.isfinite(float(end))
                and 0 <= float(start) < float(end)) for start, end in spans)
                or any(float(spans[i][0]) < float(spans[i-1][1]) for i in range(1,len(spans)))):
            exclusions['invalid_or_overlapping_boundaries'] += 1
            continue
        duration = sum(float(end) - float(start) for start, end in spans)
        if not 1 <= duration <= 18:
            exclusions['duration'] += 1
        elif lexical_normalization_loss(row['normed_text'], tokenizer):
            exclusions['lexical_normalization'] += 1
        else:
            accepted.append(row)
    return accepted


def select_rows(rows, sessions=3, speakers_per_accent=2):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['session_id']].append(row)
    chosen = []
    accents, speakers = set(), set()
    for _ in range(min(sessions, len(grouped))):
        session = min(grouped, key=lambda s: (
            -len({r['accent'] for r in grouped[s]} - accents),
            -len({r['speaker_id'] for r in grouped[s]} - speakers), rank(s)))
        chosen.extend(grouped.pop(session))
        accents.update(r['accent'] for r in chosen)
        speakers.update(r['speaker_id'] for r in chosen)
    by_accent = defaultdict(lambda: defaultdict(list))
    for row in chosen:
        by_accent[row['accent']][row['speaker_id']].append(row)
    selected, used = [], set()
    for accent, voices in sorted(by_accent.items()):
        for speaker in sorted(voices, key=rank):
            if speaker in used:
                continue
            selected.append(min(voices[speaker], key=lambda r: rank(r['id_'])))
            used.add(speaker)
            if sum(r['accent'] == accent for r in selected) == speakers_per_accent:
                break
    return sorted(selected, key=lambda r: rank(r['id_']))


class RangeTar:
    def __init__(self, url):
        self.url, self.total, self.etag = url, None, None

    def read(self, offset, length):
        request = urllib.request.Request(self.url, headers={
            'Range': f'bytes={offset}-{offset + length - 1}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=45) as response:
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if (response.status != 206 or not match
                    or tuple(map(int, match.groups()[:2])) != (offset, offset + length - 1)):
                raise ValueError('Server did not honor exact archive range')
            total, etag = int(match[3]), response.headers.get('ETag')
            if self.total is not None and (total != self.total or etag != self.etag):
                raise ValueError('Archive changed during acquisition')
            self.total, self.etag = total, etag
            data = response.read(length + 1)
            if len(data) != length:
                raise ValueError('Truncated or oversized archive range')
            return data

    def locate(self, names):
        found, offset = {}, 0
        while self.total is None or offset + 512 <= self.total:
            header = self.read(offset, 512)
            if header == b'\0' * 512:
                break
            member = tarfile.TarInfo.frombuf(header, encoding='utf-8', errors='strict')
            if member.name in names:
                if not member.isfile() or not 0 < member.size <= 256 * 1024**2:
                    raise ValueError('Invalid selected session member')
                found[member.name] = (offset + 512, member.size)
                if set(found) == set(names):
                    return found
            offset += 512 + ((member.size + 511) // 512) * 512
        raise ValueError('Selected session audio is absent from the official archive')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data/speech_scale/parliament_test')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = args.output / 'asr_en_accented.tsv.gz'
    url = BASE + '/annotations/asr/asr_en_accented.tsv.gz'
    if not annotations.exists():
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = response.read(16 * 1024**2 + 1)
        if len(payload) > 16 * 1024**2:
            raise ValueError('Unexpected annotation size')
        annotations.write_bytes(payload)
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(annotations.read_bytes()).decode()), delimiter='|'))
    exclusions = defaultdict(int)
    eligible = eligible_rows(rows, 2019, exclusions)
    selected = select_rows(eligible)
    if not selected:
        raise ValueError('No eligible gold transcripts')
    frozen = {'selection_policy': '2019, three annotation-selected sessions; maximize unseen accent coverage then speaker coverage; hash ties; at most two speakers per accent and one clip per speaker',
        'annotations_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
        'eligible_recordings': len(eligible), 'exclusions': dict(exclusions),
        'selected_ids': [r['id_'] for r in selected], 'selected_sessions': sorted({r['session_id'] for r in selected}),
        'limits': 'Small single-year, session-clustered parliamentary challenge; not a representative multilingual or accent benchmark. No test audio is used for training or selection of model/decoder.'}
    selection = args.output / 'selection.json'
    if selection.exists() and json.loads(selection.read_text(encoding='utf-8')) != frozen:
        raise ValueError('Frozen challenge selection changed')
    selection.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print(json.dumps(frozen), flush=True)
    archive = RangeTar(BASE + '/audios/original_2019.tar')
    names = {f'original/2019/{s}_original.ogg': s for s in frozen['selected_sessions']}
    located = archive.locate(names)
    prior_path = args.output / 'provenance.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else None
    if prior and (prior['archive_url'] != archive.url or prior['archive_bytes'] != archive.total
            or prior['archive_etag'] != archive.etag):
        raise ValueError('Frozen challenge source archive changed')
    known = {r['member']: r for r in prior['sessions']} if prior else {}
    provenance, paths = [], {}
    for name, (offset, size) in located.items():
        session = names[name]
        target = args.output / (session + '.ogg')
        if name in known and target.exists():
            if (known[name]['bytes'] != size or target.stat().st_size != size
                    or hashlib.sha256(target.read_bytes()).hexdigest() != known[name]['sha256']):
                raise ValueError('Frozen session audio fingerprint changed')
        else:
            with target.with_suffix('.partial').open('wb') as output:
                for start in range(0, size, 8 * 1024**2):
                    output.write(archive.read(offset + start, min(8 * 1024**2, size - start)))
            target.with_suffix('.partial').replace(target)
        paths[session] = target
        provenance.append({'member': name, 'bytes': size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        print(f'Acquired {session}: {size} bytes', flush=True)
    metadata, manifest = [], []
    for row in selected:
        path = args.output / ('clip-' + rank(row['id_'])[:16] + '.wav')
        with sf.SoundFile(paths[row['session_id']]) as audio:
            if audio.samplerate != 16000 or audio.channels != 1:
                raise ValueError('Unexpected official session audio format')
            pieces = []
            for start, end in ast.literal_eval(row['vad']):
                first, last = int(float(start) * 16000), int(float(end) * 16000)
                if last > len(audio):
                    raise ValueError('Published boundary exceeds the session audio')
                audio.seek(first)
                pieces.append(audio.read(last - first, dtype='float32'))
        waveform = np.concatenate(pieces)
        if not np.isfinite(waveform).all():
            raise ValueError('Nonfinite session audio samples')
        # Match shared preprocessing before PCM16 encoding, avoiding Vorbis
        # overshoot clipping while preserving within-recording relative levels.
        waveform = normalize_amplitude(waveform)
        sf.write(path, waveform, 16000, subtype='PCM_16')
        text = normalize_text(row['normed_text'])
        manifest.append(f'{path.resolve()}\t{text}\n')
        metadata.append({'file': str(path.resolve()), 'text': text, 'raw_text': row['original_text'],
            'language': 'en', 'speaker': 'voxpopuli:' + row['speaker_id'], 'corpus': 'VoxPopuli-accented-frozen',
            'dialect': row['accent'], 'license': 'CC0', 'source': SOURCE, 'prompt_id': 'voxpopuli:' + row['id_'],
            'session_id': row['session_id'], 'is_gold_transcript': True,
            'preprocessing': 'Published nonoverlapping VAD spans concatenated; shared peak normalization to 0.95 before PCM16 encoding',
            'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output / 'test.tsv').write_bytes(''.join(manifest).encode())
    (args.output / 'metadata.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in metadata), encoding='utf-8')
    (args.output / 'provenance.json').write_text(json.dumps({'archive_url': archive.url,
        'archive_bytes': archive.total, 'archive_etag': archive.etag, 'sessions': provenance,
        'limits': 'Local checksums for exact selected member bytes; no publisher checksum of the entire archive is claimed.'}, indent=2), encoding='utf-8')
    print(json.dumps({'clips': len(metadata), 'speakers': len({r['speaker'] for r in metadata}),
        'accents': len({r['dialect'] for r in metadata}), 'hours': sum(sf.info(r['file']).duration for r in metadata) / 3600}), flush=True)


if __name__ == '__main__':
    main()
