"""Download official CC BY 4.0 Mini LibriSpeech and create speaker-disjoint manifests."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speech_engine.dataset import load_librispeech


def speaker(path):
    return Path(path).stem.split('-')[0]


def validate_disjoint(partitions):
    seen_speakers, seen_audio = set(), set()
    for rows in partitions.values():
        speakers = {speaker(path) for path,text in rows}
        audio = {str(Path(path).resolve()) for path,text in rows}
        if speakers & seen_speakers or audio & seen_audio:
            raise ValueError('Speech partitions overlap in speakers or audio')
        seen_speakers.update(speakers); seen_audio.update(audio)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'data'/'speech_real')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = 'https://www.openslr.org/resources/31/'
    checks = urllib.request.urlopen(base+'md5sum.txt',timeout=30).read().decode()
    expected = {line.split()[-1].lstrip('*'):line.split()[0] for line in checks.splitlines()}
    hashes = {}
    for name in ['train-clean-5','dev-clean-2']:
        filename = name+'.tar.gz'; archive = args.output/filename
        if not archive.exists():
            tmp = archive.with_suffix('.part')
            print('Downloading',filename,flush=True)
            request = urllib.request.Request(base+filename,headers={'User-Agent':'LinguaForge local research data preparation'})
            with urllib.request.urlopen(request,timeout=30) as response, tmp.open('wb') as output:
                received = 0
                while chunk := response.read(64*1024):
                    output.write(chunk); received += len(chunk)
                    if received % (16*1024*1024) == 0:
                        print(f'{filename}: {received//(1024*1024)} MiB',flush=True)
            tmp.replace(archive)
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != expected[filename]:
            raise ValueError(f'Official checksum mismatch for {filename}')
        hashes[filename] = hashlib.sha256(archive.read_bytes()).hexdigest()
        marker = args.output/(name+'.extracted')
        if not marker.exists() or marker.read_text() != hashes[filename]:
            with tarfile.open(archive) as package:
                # Python data filter rejects traversal, unsafe links and special files.
                package.extractall(args.output,filter='data')
            marker.write_text(hashes[filename])
        print('Verified and extracted',filename,flush=True)
    training = load_librispeech(args.output/'LibriSpeech','train-clean-5')
    development = load_librispeech(args.output/'LibriSpeech','dev-clean-2')
    speakers = sorted({speaker(path) for path,text in development})
    if len(speakers) < 2: raise ValueError('Need multiple held-out speakers')
    dev_speakers = set(speakers[::2])
    partitions = {'train':training,
        'dev':[(path,text) for path,text in development if speaker(path) in dev_speakers],
        'test':[(path,text) for path,text in development if speaker(path) not in dev_speakers]}
    validate_disjoint(partitions)
    for name, rows in partitions.items():
        (args.output/f'{name}.tsv').write_text(''.join(f'{Path(path).resolve()}\t{text}\n' for path,text in rows),encoding='utf-8')
    report = {'dataset':'Mini LibriSpeech SLR31; derived from LibriSpeech by Panayotov et al.',
        'license':'CC BY 4.0','source':'https://openslr.org/31','sha256':hashes,
        'counts':{name:len(rows) for name,rows in partitions.items()},
        'speakers':{name:sorted({speaker(path) for path,text in rows}) for name,rows in partitions.items()},
        'policy':'official training subset; official development speakers split between development and internal held-out test',
        'limits':'small read-English regression subset; not representative multilingual/noisy speech; internal test is not official LibriSpeech test-clean'}
    (args.output/'provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
