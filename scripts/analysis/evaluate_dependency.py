"""Evaluate syntax with gold and production POS/features on held-out words."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from speech_engine.dependency import DependencyParser, read_dependencies, projective
from speech_engine.morphology import MorphologyEngine
from speech_engine.uncertainty_evaluation import sentence_fingerprint, sentence_fingerprints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model','corpus','training','development','resources','output']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); model=DependencyParser.load(args.model)
    if sha256(Path(args.training).read_bytes()).hexdigest()!=model.source['training_sha256']:
        raise ValueError('Mismatched training corpus')
    checksum=sha256(Path(args.corpus).read_bytes()).hexdigest()
    if checksum==model.source['training_sha256']:
        raise ValueError('Use held-out data')
    excluded=sentence_fingerprints(args.training)
    if Path(args.corpus).resolve()!=Path(args.development).resolve():
        excluded |= sentence_fingerprints(args.development)
    engine=MorphologyEngine();engine.load_resources(args.resources)
    lexicon=engine._lexicons.get(model.language)
    if not lexicon or lexicon.source['sha256'] != model.source['training_sha256']:
        raise ValueError('Evaluation requires matching lexical resources')
    if (model.source.get('pos_model_sha256') is not None and
            model.source['pos_model_sha256'] != engine.tagger.model_fingerprint(model.language)):
        raise ValueError('Evaluation requires matching POS model')
    methods={name:Counter() for name in ['left_chain','verb_hub','gold_features','production_features',
                                        'gold_pos_production_morphology','production_pos_gold_morphology']}
    by_relation={}; excluded_count=0; start=perf_counter()
    for sentence in read_dependencies(args.corpus):
        if sentence_fingerprint(sentence.words) in excluded:
            excluded_count+=1;continue
        gold=[{'surface':w.form,'pos':w.pos,'ud_features':dict(w.features)} for w in sentence.words]
        predicted=[a.to_dict() for a in engine.analyze_tokens([w.form for w in sentence.words],model.language).analyses]
        nonprojective = not projective(sentence)
        root=next((i for i,t in enumerate(predicted) if t['pos']=='VERB'),0)
        guesses={'left_chain':[{'head_token_index':i-1 if i else None,'relation':'root' if not i else 'dep'} for i in range(len(gold))],
                 'verb_hub':[{'head_token_index':None if i==root else root,'relation':'root' if i==root else 'dep'} for i in range(len(gold))],
                 'gold_features':model.parse(gold),'production_features':model.parse(predicted),
                 'gold_pos_production_morphology':model.parse([
                     {**p,'pos':g['pos']} for p,g in zip(predicted,gold)]),
                 'production_pos_gold_morphology':model.parse([
                     {**p,'ud_features':g['ud_features']} for p,g in zip(predicted,gold)])}
        for name,arcs in guesses.items():
            for i,(word,arc) in enumerate(zip(sentence.words,arcs)):
                if word.pos=='PUNCT':continue
                head=arc['head_token_index']; correct=(head+1 if head is not None else 0)==sentence.heads[i]
                labeled=correct and arc['relation']==sentence.labels[i]
                methods[name].update(tokens=1,uas=correct,las=labeled,nonprojective_tokens=nonprojective)
                if name=='production_features':
                    count=by_relation.setdefault(sentence.labels[i],Counter())
                    count.update(tokens=1,uas=correct,las=labeled)
    def metrics(count):
        return {**count,'uas':count['uas']/count['tokens'] if count['tokens'] else None,
                'las':count['las']/count['tokens'] if count['tokens'] else None}
    report={'corpus_sha256':checksum,'model_sha256':sha256(Path(args.model).read_bytes()).hexdigest(),
            'source': model.source,
            'excluded_sentences':excluded_count,'methods':{k:metrics(v) for k,v in methods.items()},
            'production_by_gold_relation':{k:metrics(v) for k,v in by_relation.items()},'seconds':perf_counter()-start,
            'limits':'Gold words and sentence boundaries, punctuation excluded from scoring. Gold_features is an oracle-input diagnostic, not product accuracy. Production uses existing POS/feature predictions. Exact train/dev overlap excluded; no filtering of nonprojective test trees. Scores are not calibrated. Basic syntax does not establish semantic roles or correct translation.'}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['methods'],indent=2))


if __name__=='__main__':
    main()
