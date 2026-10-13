"""Training-only alignment ranking with original local NMT; no verified-quality claim.
Compare paired source conditional loss against a deterministic mismatched source.
A positive margin means the paired source helps predict its target. Ranking remains
model dependent; preserve replay and compare against a matched random removal.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from speech_engine.model_objectives import sequence_token_nll
from speech_engine.nmt import load_checkpoint, collate_pairs
from scripts.phase7_train import load_corpus

@torch.inference_mode()
def losses(model, pairs, sv, tv, device):
    values=[]
    for start in range(0,len(pairs),64):
        batch=pairs[start:start+64]
        encoded=[(sv.encode(s,add_bos=True,add_eos=True),tv.encode(t,add_bos=True,add_eos=True)) for s,t in batch]
        src,_,tin,tout,_=collate_pairs(encoded,pad_id=0)
        logits=model(src.to(device),tin.to(device)); target=tout.to(device)
        values.extend(sequence_token_nll(logits,target).cpu().tolist())
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--news',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remove-fraction',type=float,default=.1);args=p.parse_args()
    if not 0 <= args.remove_fraction < 1:raise ValueError('Invalid removal fraction')
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,sv,tv,_=load_checkpoint(args.checkpoint);model.to(device).eval()
    rows=load_corpus(args.news);replay=load_corpus(args.replay)
    # Every mismatched source differs by exact text; no target is used for training selection outside train.
    rotated=[(rows[(i+len(rows)//2)%len(rows)][0],t) for i,(s,t) in enumerate(rows)]
    if len({s for s,t in rows})<2:raise ValueError('Need at least two distinct training sources')
    fallback=next(s for s,t in rows if s!=rows[0][0])
    rotated=[(s if s!=rows[i][0] else fallback if rows[i][0]==rows[0][0] else rows[0][0],t) for i,(s,t) in enumerate(rotated)]
    paired=losses(model,rows,sv,tv,device);mismatched=losses(model,rotated,sv,tv,device)
    margins=[b-a for a,b in zip(paired,mismatched)]
    keep=len(rows)-round(len(rows)*args.remove_fraction)
    ranked=sorted(range(len(rows)),key=lambda i:margins[i],reverse=True)[:keep]
    random_indices=random.Random(731).sample(range(len(rows)),keep)
    args.output.mkdir(parents=True,exist_ok=True)
    for mode,indices in [('ranked',ranked),('random',random_indices)]:
        directory=args.output/mode;directory.mkdir(exist_ok=True)
        training=[rows[i] for i in sorted(indices)]+replay
        (directory/'train.tsv').write_text(''.join(f'{s}\t{t}\n' for s,t in training),encoding='utf-8')
        for name in ['dev','test']:(directory/f'{name}.tsv').write_bytes((args.splits/f'{name}.tsv').read_bytes())
    report={'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'news_sha256':hashlib.sha256(args.news.read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        'news_count':len(rows),'kept_news':keep,'replay_count':len(replay),'seed':731,
        'limitation':'Model-dependent training ranking, not verified alignment or semantic quality; shuffled-source contrast can favor familiar sentences.',
        'lowest_margin_examples':[{'source':rows[i][0],'target':rows[i][1],'paired_nll':paired[i],'mismatched_nll':mismatched[i],'margin':margins[i]} for i in sorted(range(len(rows)),key=lambda i:margins[i])[:30]]}
    (args.output/'ranking.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='lowest_margin_examples'}),flush=True)
if __name__=='__main__':main()
