"""Train the original parser locally from validated dependency trees."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speech_engine.dependency import DependencyParser, read_dependencies
from speech_engine.morphology import MorphologyEngine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','language','source-url','license','output','report']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--resources',help='Train on production POS/morphology from matching local resources')
    args=parser.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    model=DependencyParser(args.language,{'training_sha256':sha256(Path(args.corpus).read_bytes()).hexdigest(),
                                        'url':args.source_url,'license':args.license})
    start=perf_counter(); sentences=list(read_dependencies(args.corpus)); evidence=None
    if args.resources:
        engine=MorphologyEngine(); engine.load_resources(args.resources)
        lexicon=engine._lexicons.get(model.language)
        if not lexicon or lexicon.source['sha256'] != model.source['training_sha256']:
            raise ValueError('Training evidence must use the same training corpus')
        evidence=[[{'surface':a.surface,'pos':a.pos,'ud_features':dict(a.ud_features)}
                   for a in engine.analyze_tokens([w.form for w in sentence.words],model.language).analyses]
                  for sentence in sentences]
        model.source['training_evidence']='production_in_sample'
        model.source['pos_model_sha256']=engine.tagger.model_fingerprint(model.language)
    else:
        model.source['training_evidence']='gold'
    report=model.fit(sentences,args.epochs,token_sequences=evidence)
    report['training_evidence']=model.source['training_evidence']
    report['seconds']=perf_counter()-start
    model.save(args.output)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
