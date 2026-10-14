"""Transparent error breakdowns; English lexical probes are opt-in, not semantics."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from speech_engine.evaluate import align_units,edit_counts,char_error_rate,word_error_rate

PROBES = {
    'negation_forms':set("no not never neither nor nothing nobody nowhere cannot don't doesn't didn't isn't aren't wasn't weren't won't wouldn't shouldn't couldn't haven't hasn't hadn't".split()),
    'number_forms':set('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion'.split()),
    # Spellings only: does not establish proper-name use or named-entity accuracy.
    'name_spelling_probes':set('john mary tom george henry james william elizabeth'.split())}

def summarize(rows,english=False):
    refs=[r['reference'] for r in rows]; hyps=[r['hypothesis'] for r in rows]
    word=Counter();char=Counter();confusions=Counter();boundary=Counter();critical={k:Counter() for k in PROBES} if english else {}
    for ref,hyp in zip(refs,hyps):
        word.update(edit_counts(ref.split(),hyp.split()));char.update(edit_counts(list(ref),list(hyp)))
        for r,h in align_units(list(ref),list(hyp)):
            if r!=h: confusions[(r,h)]+=1
            if r==' ' or h==' ':
                boundary['matches' if r==h else 'insertions' if r is None else 'deletions' if h is None else 'substitutions']+=1
        for r,h in align_units(ref.split(),hyp.split()):
            for key,forms in (PROBES.items() if english else []):
                if r in forms:
                    critical[key]['reference_occurrences']+=1
                    critical[key]['exact_aligned_matches' if r==h else 'deletions' if h is None else 'substitutions']+=1
                if h in forms and r!=h:critical[key]['unaligned_hypothesis_occurrences']+=1
    return {'wer':word_error_rate(refs,hyps),'cer':char_error_rate(refs,hyps),
        'word_edits':dict(word),'character_edits':dict(char),'space_alignment':dict(boundary),
        'critical_lexical_forms':{k:dict(v) for k,v in critical.items()},
        'frequent_character_errors':[{'reference':r,'hypothesis':h,'count':c} for (r,h),c in confusions.most_common(15)]}

def main():
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--mode',default='lm02');p.add_argument('--english-probes',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare',type=Path,help='Paired baseline on identical utterances and noise settings')
    a=p.parse_args()
    report=json.loads(a.report.read_text(encoding='utf-8'));rows=report['metrics'][a.mode]['rows']
    speakers=defaultdict(list)
    corpora=defaultdict(list)
    for row in rows:speakers[row.get('speaker') or Path(row['file']).stem.split('-')[0]].append(row)
    for row in rows:
        if row.get('corpus'):corpora[row['corpus']].append(row)
    result={'source_report':str(a.report),'checkpoint_sha256':report['checkpoint_sha256'],
        'aggregate':summarize(rows,a.english_probes),
        'by_speaker':{s:summarize(r) for s,r in speakers.items()},
        'by_corpus':{s:summarize(r,a.english_probes) for s,r in corpora.items()},
        'limits':'One minimum-edit alignment. Spaces are English transcription separators, not universal word boundaries. Opt-in English lexical probes measure spelling retention, not semantic scope, factual fidelity or verified proper names. No automatic transcript corrections.'}
    if a.compare:
        baseline=json.loads(a.compare.read_text(encoding='utf-8'))
        other=baseline['metrics'][a.mode]['rows']
        other_by_file={str(Path(r['file']).resolve()):r for r in other}
        identities=[str(Path(r['file']).resolve()) for r in rows]
        if len(set(identities))!=len(rows) or len(other_by_file)!=len(other) or set(identities)!=set(other_by_file):
            raise ValueError('Paired comparison requires unique, identical recording identities')
        other=[other_by_file[key] for key in identities]
        if [r['reference'] for r in rows]!=[r['reference'] for r in other] or report.get('synthetic_noise')!=baseline.get('synthetic_noise'):
            raise ValueError('Paired comparison must use identical recordings, references and conditions')
        grouped=defaultdict(lambda:np.zeros(3,dtype=np.float64))
        for candidate,old in zip(rows,other):
            ref=candidate['reference'].split()
            def errors(row):
                c=edit_counts(ref,row['hypothesis'].split())
                return c['substitutions']+c['deletions']+c['insertions']
            grouped[candidate.get('speaker') or Path(candidate['file']).stem.split('-')[0]]+=np.array([errors(candidate),errors(old),len(ref)])
        counts=np.stack(list(grouped.values()));rng=np.random.default_rng(127)
        samples=counts[rng.integers(0,len(counts),size=(2000,len(counts)))].sum(axis=1)
        differences=(samples[:,0]-samples[:,1])/samples[:,2]
        result['paired_speaker_bootstrap']={'baseline':str(a.compare),'samples':2000,'seed':127,
            'wer_difference':float((counts[:,0].sum()-counts[:,1].sum())/counts[:,2].sum()),
            'percentile_95_interval':np.quantile(differences,[.025,.975]).tolist(),
            'limits':'Speaker-cluster resampling within this fixed subset; does not cover training-seed uncertainty, new languages or all recording conditions.'}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['aggregate']),flush=True)

if __name__=='__main__':main()
