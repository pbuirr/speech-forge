import numpy as np
import pytest
import torch
from scripts.translation.prepare_news_adaptation import article_partitions
from scripts.translation.train_local_translation import canonical
from scripts.speech.prepare_real_speech import validate_disjoint
from speech_engine.dataset import ASRDataset, collate_fn
from speech_engine.tokenizer import CharTokenizer


def test_ctc_training_normalization_excludes_extra_right_padding():
    import copy
    from speech_engine.model_ctc import CTCModel
    torch.manual_seed(19)
    model=CTCModel(33,cnn_channels=(4,4,4),hidden_size=8,num_lstm_layers=1,dropout=0).train()
    model.encoder.masked_batch_norm=True
    padded=copy.deepcopy(model)
    mel=torch.randn(2,80,41); lengths=torch.tensor([41,25])
    mel[1,:,25:]=0
    first=model(mel,lengths)
    second=padded(torch.nn.functional.pad(mel,(0,30)),lengths)
    torch.testing.assert_close(first[:13],second[:13],atol=2e-5,rtol=2e-5)
    for a,b in zip(model.encoder.modules(),padded.encoder.modules()):
        if isinstance(a,torch.nn.BatchNorm2d):
            torch.testing.assert_close(a.running_mean,b.running_mean)
            torch.testing.assert_close(a.running_var,b.running_var)
    first.sum().backward();second[:first.shape[0]].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_unicode_speech_vocabulary_policy_roundtrip_and_legacy_compatibility(tmp_path):
    legacy = CharTokenizer()
    assert legacy.decode(legacy.encode('café 42')) == 'caf'
    text = 'café türkçe 日本語 42'
    tokenizer = CharTokenizer(normalization='unicode').fit([text])
    assert tokenizer.decode(tokenizer.encode(text)) == text
    for char in 'abcdefghijklmnopqrstuvwxyz':
        assert tokenizer.encode(char) == legacy.encode(char)
    path=tmp_path/'tokenizer.json'; tokenizer.save(path)
    restored=CharTokenizer.load(path)
    assert restored.normalization == 'unicode'
    assert restored.decode(restored.encode(text)) == text


def test_speech_dataset_rejects_silently_dropped_unicode_reference(tmp_path):
    import soundfile as sf
    path=tmp_path/'speech.wav';sf.write(path,np.zeros(16000,dtype=np.float32),16000)
    tokenizer=CharTokenizer(normalization='unicode')
    dataset=ASRDataset([(str(path),'türkçe')],tokenizer)
    with pytest.raises(ValueError,match='characters absent'):
        dataset[0]


def test_news_exclusion_removes_entire_connected_articles():
    records = [('a','reserved sentence here','reserved translation here'),
               ('a','another sibling sentence','another sibling translation'),
               ('b','another sibling sentence','linked article translation')]
    records += [(str(i),f'unique source number {i}',f'unique target number {i}') for i in range(2300)]
    splits,_ = article_partitions(records,{('s',canonical('reserved sentence here'))})
    assert not any(doc in {'a','b'} for rows in splits.values() for doc,s,t in rows)
    for left,right in [('train','dev'),('dev','test'),('train','test')]:
        assert {d for d,s,t in splits[left]}.isdisjoint(d for d,s,t in splits[right])
        for side in [1,2]:
            assert {canonical(row[side]) for row in splits[left]}.isdisjoint(canonical(row[side]) for row in splits[right])


def test_speech_partitions_reject_shared_speakers():
    with pytest.raises(ValueError,match='speakers'):
        validate_disjoint({'train':[('42-1-1.flac','one')], 'test':[('42-2-2.flac','two')]})
    validate_disjoint({'train':[('42-1-1.flac','one')], 'test':[('43-2-2.flac','two')]})


def test_speech_dataset_keeps_true_lengths_and_rejects_unaligned_truncation(monkeypatch):
    dataset=ASRDataset([('short.wav','hello'),('long.wav','world')],CharTokenizer(),max_frames=100)
    monkeypatch.setattr(dataset,'_load_or_compute_spec',lambda path: np.zeros((80,40 if path=='short.wav' else 120),dtype=np.float32))
    mel,tokens=dataset[0]
    assert mel.shape==(80,40)
    assert collate_fn([(mel,tokens)])['mel_lengths'].tolist()==[40]
    with pytest.raises(ValueError,match='segment audio and transcript together'): dataset[1]


def test_speech_evaluation_does_not_decode_padded_frames():
    from speech_engine.evaluate import evaluate_dataset
    from speech_engine.decode import greedy_decode
    tokenizer=CharTokenizer()
    a,b=tokenizer.encode('ab')
    class Model:
        def eval(self): return self
        def encoder_output_length(self,lengths): return lengths
        def __call__(self,mel,mel_lengths=None):
            logits=torch.full((2,2,tokenizer.vocab_size),-100.)
            logits[0,:,a]=0.; logits[1,:,b]=0.
            return logits
    batch={'mel':torch.zeros(2,80,2),'mel_lengths':torch.tensor([1,2]),
           'tokens':torch.tensor([[a,1],[a,b]]),'token_lengths':torch.tensor([1,2])}
    result=evaluate_dataset(Model(),[batch],tokenizer,greedy_decode,torch.device('cpu'))
    assert result['wer']==0 and result['cer']==0


def test_ctc_predictions_are_invariant_to_right_padding_when_lengths_supplied():
    from speech_engine.model_ctc import CTCModel
    model=CTCModel(33,cnn_channels=(4,4,4),hidden_size=8,num_lstm_layers=1,dropout=0).eval()
    mel=torch.randn(1,80,43)
    with torch.inference_mode():
        plain=model(mel)
        padded=model(torch.nn.functional.pad(mel,(0,37)),torch.tensor([43]))
    assert torch.allclose(plain,padded[:plain.shape[0]],atol=1e-5)


def test_ctc_alignment_accounts_for_repeated_characters_and_ignores_padding():
    from speech_engine.train import validate_ctc_alignment
    tokens=torch.tensor([[4,4,1,1],[5,6,6,7]])
    lengths=torch.tensor([2,4])
    validate_ctc_alignment(tokens,lengths,torch.tensor([3,5]))
    with pytest.raises(ValueError,match='repeated characters'):
        validate_ctc_alignment(tokens,lengths,torch.tensor([2,5]))


def test_speech_feature_cache_distinguishes_same_filename_and_changed_audio(tmp_path,monkeypatch):
    first=tmp_path/'a'/'clip.wav'; second=tmp_path/'b'/'clip.wav'
    for path,content in [(first,b'first'),(second,b'second')]:
        path.parent.mkdir(); path.write_bytes(content)
    dataset=ASRDataset([(str(first),'one'),(str(second),'two')],CharTokenizer(),cache_dir=tmp_path/'cache')
    calls=[]
    def compute(path):
        calls.append(path)
        return np.ones((80,len(Path(path).read_bytes())),dtype=np.float32)
    from pathlib import Path
    monkeypatch.setattr(dataset,'_compute_spec',compute)
    assert dataset._load_or_compute_spec(str(first)).shape[1]==5
    assert dataset._load_or_compute_spec(str(second)).shape[1]==6
    dataset._load_or_compute_spec(str(first)); assert len(calls)==2
    first.write_bytes(b'changed audio')
    assert dataset._load_or_compute_spec(str(first)).shape[1]==13
    assert len(calls)==3


def test_transformer_speech_encoder_preserves_valid_frames_under_padding():
    import torch
    from speech_engine.model_transformer import TransformerASR
    torch.manual_seed(48)
    model=TransformerASR(12,d_model=8,n_heads=2,d_ff=16,n_enc_layers=1,
        n_dec_layers=1,cnn_channels=(2,4,4),dropout=0).eval()
    mel=torch.randn(1,80,17)
    with torch.inference_mode():
        plain,_=model.encode(mel,torch.tensor([17]))
        padded,_=model.encode(torch.nn.functional.pad(mel,(0,14)),torch.tensor([17]))
    assert torch.allclose(plain,padded[:,:plain.size(1)],atol=1e-5)


def test_ctc_beam_matches_exhaustive_paths_and_optional_fusion():
    import itertools
    from collections import defaultdict
    from speech_engine.decode import prefix_beam_search,beam_decode
    tokenizer=CharTokenizer();blank=tokenizer.blank_id
    a=tokenizer.encode('a')[0];b=tokenizer.encode('b')[0]
    symbols=[blank,a,b]
    probabilities=np.array([[.4,.35,.25],[.2,.5,.3],[.4,.3,.3]])
    totals=defaultdict(float)
    for path in itertools.product(range(3),repeat=3):
        collapsed=[];previous=None
        for index in path:
            token=symbols[index]
            if token!=previous and token!=blank:collapsed.append(token)
            previous=token
        totals[tokenizer.decode(collapsed)]+=np.prod([probabilities[t,index] for t,index in enumerate(path)])
    logits=np.full((3,tokenizer.vocab_size),-np.inf)
    logits[:,symbols]=np.log(probabilities)
    expected=max(totals,key=totals.get)
    assert prefix_beam_search(logits,tokenizer,beam_width=100)==expected
    class Preference:
        def score(self,char,context):return 0 if char=='b' else -10
    assert prefix_beam_search(logits,tokenizer,beam_width=100,lm=Preference(),lm_weight=0)==expected
    ambiguous=np.full((2,tokenizer.vocab_size),-np.inf)
    ambiguous[:,symbols]=np.log([[.15,.45,.4],[.9,.05,.05]])
    assert prefix_beam_search(ambiguous,tokenizer,beam_width=10)=='a'
    fused=beam_decode(torch.tensor(ambiguous),tokenizer,beam_width=10,lm=Preference(),lm_weight=1)
    assert fused==['b']


def test_real_speech_warm_start_binds_data_and_preserves_tokenizer(tmp_path,monkeypatch):
    import hashlib,json,sys
    import soundfile as sf
    from scripts.speech import train_real_speech as driver
    from speech_engine.model_ctc import CTCModel
    data=tmp_path/'data';data.mkdir()
    old=tmp_path/'old';old.mkdir()
    tokenizer=CharTokenizer();tokenizer.save(old/'tokenizer.json')
    for name,speaker in [('train','101'),('dev','202'),('test','303')]:
        audio=data/(speaker+'-1-1.wav');sf.write(audio,np.zeros(16000,dtype=np.float32),16000)
        (data/(name+'.tsv')).write_text(str(audio)+'\thello\n',encoding='utf-8')
    model=CTCModel(tokenizer.vocab_size,cnn_channels=(16,32,64),hidden_size=128,num_lstm_layers=2)
    checkpoint=old/'best.pt';torch.save({'model':model.state_dict(),'config':{
        'cnn_channels':[16,32,64],'hidden_size':128,'num_lstm_layers':2,
        'tokenizer_path':str(old/'tokenizer.json'),'tokenizer_sha256':hashlib.sha256((old/'tokenizer.json').read_bytes()).hexdigest(),
        'accepted_manifest_sha256':{'train':'different'}}},checkpoint)
    output=tmp_path/'new'
    args=['train','--data',str(data),'--output',str(output),'--epochs','1']
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(sys,'argv',args+['--resume',str(checkpoint)])
    with pytest.raises(ValueError,match='selection changed'):
        driver.main()
    assert not (output/'tokenizer.json').exists()
    monkeypatch.setattr(sys,'argv',args+['--architecture','full','--warm-start',str(checkpoint)])
    with pytest.raises(ValueError,match='architecture differs'):driver.main()
    assert not (output/'tokenizer.json').exists()
    def train_stub(model,training,dev,config,device,resume_from=None):
        assert resume_from is None
        for name,digest in config.accepted_manifest_sha256.items():
            assert hashlib.sha256((output/(name+'.tsv')).read_bytes()).hexdigest()==digest
        torch.save({'model':model.state_dict(),'config':config.__dict__},output/'best.pt')
        return {}
    monkeypatch.setattr(driver,'train',train_stub)
    monkeypatch.setattr(driver,'evaluate_dataset',lambda *args:{'n_examples':1})
    monkeypatch.setattr(sys,'argv',args+['--warm-start',str(checkpoint)])
    driver.main()
    assert (output/'tokenizer.json').read_bytes()==(old/'tokenizer.json').read_bytes()
    report=json.loads((output/'evaluation.json').read_text())
    assert report['counts']=={'train':1,'dev':1,'test':1}
def test_unavailable_cuda_is_rejected_before_data_or_outputs(tmp_path, monkeypatch):
    import sys
    from scripts.speech import train_real_speech
    monkeypatch.setattr(train_real_speech.torch.cuda, 'is_available', lambda: False)
    output=tmp_path/'preserved'
    monkeypatch.setattr(sys,'argv',['train_real_speech','--device','cuda','--data',str(tmp_path/'missing'),
                                   '--output',str(output)])
    with pytest.raises(ValueError,match='CUDA.*unavailable'):
        train_real_speech.main()
    assert not output.exists()

