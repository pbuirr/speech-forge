"""Measure text-language confusion caused by ASR, not acoustic language ID."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from speech_engine.language_detector import LanguageDetector

def main():
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--language',required=True);p.add_argument('--decoder',default='lm02');a=p.parse_args()
    report=json.loads(a.report.read_text(encoding='utf-8'));detector=LanguageDetector()
    rows=report['metrics'][a.decoder]['rows'] if 'metrics' in report else report['variants'][a.decoder]['rows']
    measurements=[]
    for row in rows:
        ref=detector.detect(row['reference']);hyp=detector.detect(row['hypothesis'])
        measurements.append({'file':row['file'],'speaker':row['speaker'],'corpus':row.get('corpus'),
            'reference_language':ref.language,'transcript_language':hyp.language,
            'transcript_score_uncalibrated':hyp.confidence,'reference':row['reference'],'hypothesis':row['hypothesis']})
    result={'language':a.language,'examples':len(rows),'reference_errors':sum(r['reference_language']!=a.language for r in measurements),
        'transcript_errors':sum(r['transcript_language']!=a.language for r in measurements),
        'transcript_confusions':dict(Counter(r['transcript_language'] for r in measurements if r['transcript_language']!=a.language)),
        'high_score_wrong_transcripts':sum(r['transcript_language']!=a.language and r['transcript_score_uncalibrated']>=.65 for r in measurements),
        'limits':'Declared checkpoint language is known for this evaluation. Text scores are not acoustic language identification or calibrated correctness; configured-language routing does not certify audio language.',
        'rows':measurements}
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
