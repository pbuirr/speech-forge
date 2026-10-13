"""Mixed-corpus adaptation with conservative speaker and prompt-disjoint splits."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import soundfile as sf
from speech_engine.corpus import lexical_normalization_loss,load_metadata,validate_partitions
from speech_engine.dataset import load_manifest
from speech_engine.tokenizer import CharTokenizer,normalize_text


def main():
    source=ROOT/'data/speech_dialects';output=source/'mixed';output.mkdir(exist_ok=True)
    tok=CharTokenizer.load(ROOT/'data/models/asr_en_quality.tokenizer.json')
    references={}
    with (source/'line_index_all.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.reader(f,skipinitialspace=True):
            if len(row)!=3:raise ValueError('Unexpected SLR83 index columns')
            prompt,file,text=row
            references[file.strip()]=(prompt.strip(),text.strip())
    dialects={'irm':'Irish English','nof':'Northern English','scf':'Scottish English'}
    rows={split:[] for split in ('train','dev','test')};records=[];exclusions=Counter()
    # Keep existing accepted evaluation/audio unchanged for matched comparisons.
    for split in rows:
        for path,text in load_manifest(ROOT/'data/training/asr_masked_norm'/(split+'.tsv')):
            rows[split].append((path,text))
            records.append({'file':path,'text':text,'raw_text':text,'language':'en',
                'speaker':'librispeech:'+Path(path).stem.split('-')[0],
                'prompt_id':'librispeech:'+Path(path).stem,'corpus':'MiniLibriSpeech',
                'license':'CC BY 4.0','source':'https://openslr.org/31','dialect':None})
    added={split:[] for split in rows}
    for archive in ('irish_english_male','northern_english_female','scottish_english_female'):
        for path in sorted((source/archive).rglob('*.wav')):
            prompt,raw=references[path.stem]
            group,code,_=path.stem.split('_')
            if group not in dialects:raise ValueError('Unexpected dialect group')
            if lexical_normalization_loss(raw,tok):exclusions['lexical_normalization_or_vocabulary_loss']+=1;continue
            if not 1<=sf.info(path).duration<=18:exclusions['duration']+=1;continue
            # Reused anonymous numeric codes are conservatively joined across
            # regional/gender prefixes. Do not assume every prefix is a new human.
            split='dev' if code=='04310' else 'test' if code=='03397' else 'train'
            bucket=int(hashlib.sha256(prompt.encode()).hexdigest()[:8],16)%10
            eligible= bucket<6 if split=='train' else 6<=bucket<8 if split=='dev' else bucket>=8
            if not eligible:exclusions['speaker_prompt_partition_intersection']+=1;continue
            text=normalize_text(raw,tok.normalization)
            added[split].append((str(path.resolve()),text))
            records.append({'file':str(path.resolve()),'text':text,'raw_text':raw,'language':'en',
                'speaker':'slr83:conservative:'+code,'speaker_group':group+'_'+code,
                'speaker_policy':'Anonymous numeric code conservatively shared across prefixes; not independently verified human identity',
                'prompt_id':'slr83:'+prompt,'corpus':'SLR83','dialect':dialects[group],
                'license':'CC BY-SA 4.0','source':'https://openslr.org/83'})
    for split in rows:rows[split]+=added[split]
    if any(not added[s] for s in added):raise ValueError('Empty dialect partition')
    meta=output/'metadata.jsonl'
    meta.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    metadata=load_metadata(meta);validate_partitions(rows,metadata)
    for split in rows:
        (output/(split+'.tsv')).write_bytes(''.join(f'{p}\t{t}\n' for p,t in rows[split]).encode('utf-8'))
        (output/('dialect_'+split+'.tsv')).write_bytes(''.join(f'{p}\t{t}\n' for p,t in added[split]).encode('utf-8'))
    report={'source':'https://openslr.org/83','license':'CC BY-SA 4.0',
        'counts':{s:len(r) for s,r in rows.items()},'dialect_counts':{s:len(r) for s,r in added.items()},
        'dialect_hours':{s:sum(sf.info(p).duration for p,_ in r)/3600 for s,r in added.items()},
        'dialect_groups':{s:dict(Counter(metadata[str(Path(p).resolve())]['dialect'] for p,_ in r)) for s,r in added.items()},
        'exclusions':dict(exclusions),'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
        'limits':'Bounded three-region elicited speech; conservative anonymous code disjointness and SLR83 prompt-disjointness, not independently verified human identity. Legacy LibriSpeech uses utterance prompt IDs; no unseen-text claim for legacy evaluation.'}
    (output/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
