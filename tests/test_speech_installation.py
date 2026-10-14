import hashlib,json,sys
import pytest
import torch
from scripts.speech import install_local_speech as installer
from speech_engine.tokenizer import CharTokenizer


def test_speech_packaging_requires_matching_evaluation_and_development_selection(tmp_path,monkeypatch):
    directory=tmp_path/'training';directory.mkdir()
    tokenizer=directory/'tokenizer.json';CharTokenizer().save(tokenizer)
    manifest=directory/'train.tsv';manifest.write_text('unused.wav\thello\n',encoding='utf-8')
    checkpoint=directory/'best.pt'
    torch.save({'model':{},'config':{'tokenizer_sha256':installer.digest(tokenizer)}},checkpoint)
    report={'checkpoint_sha256':installer.digest(checkpoint),'training_manifest_sha256':installer.digest(manifest),
        'metrics':{'greedy':{'wer':.9,'cer':.5},'lm02':{'wer':.8,'cer':.4,'lm_weight':.2,'token_reward':.2}}}
    dev=tmp_path/'dev.json';test=tmp_path/'test.json'
    for path in [dev,test]:path.write_text(json.dumps(report),encoding='utf-8')
    output=tmp_path/'app.pt'
    args=['install','--directory',str(directory),'--development-report',str(dev),'--test-report',str(test),'--output',str(output)]
    monkeypatch.setattr(sys,'argv',args+['--mode','greedy'])
    with pytest.raises(ValueError,match='development WER'):installer.main()
    assert not output.exists()
    monkeypatch.setattr(sys,'argv',args+['--mode','lm02']);installer.main()
    saved=torch.load(output,weights_only=True)
    assert saved['config']['asr_decoder']['training_manifest_sha256']==installer.digest(manifest)
    assert saved['config']['asr_decoder']['lm_order']==3
    assert output.with_suffix('.tokenizer.json').read_bytes()==tokenizer.read_bytes()
    alternative=tmp_path/'other_training.tsv';alternative.write_text('other.wav\tworld\n',encoding='utf-8')
    report['lm_training_manifest_path']=str(alternative)
    report['lm_training_manifest_sha256']=installer.digest(alternative)
    for path in [dev,test]:path.write_text(json.dumps(report),encoding='utf-8')
    installer.main()
    assert torch.load(output,weights_only=True)['config']['asr_decoder']['training_manifest_sha256']==installer.digest(alternative)
    previous=installer.digest(output)
    bad={**report,'lm_training_manifest_sha256':'changed'};test.write_text(json.dumps(bad),encoding='utf-8')
    with pytest.raises(ValueError,match='LM data differs'):installer.main()
    assert installer.digest(output)==previous
    report['lm_order']=5
    dev.write_text(json.dumps(report),encoding='utf-8')
    test.write_text(json.dumps({**report,'lm_order':3}),encoding='utf-8')
    with pytest.raises(ValueError,match='context order differs'):installer.main()
    assert installer.digest(output)==previous
    test.write_text(json.dumps(report),encoding='utf-8');installer.main()
    assert torch.load(output,weights_only=True)['config']['asr_decoder']['lm_order']==5
    report['checkpoint_sha256']='changed';test.write_text(json.dumps(report),encoding='utf-8')
    with pytest.raises(ValueError,match='Evaluation'):installer.main()
