import pytest
from scripts.train_local_translation import canonical, split_pairs


def test_parallel_splits_group_both_sides_and_are_reproducible():
    pairs = [(f'input {i}', f'output {i}') for i in range(50)]
    pairs += [('INPUT 0!', 'different output'), ('another input', 'OUTPUT 0!')]
    split = split_pairs(pairs)
    assert split == split_pairs(pairs)
    assert sum(map(len, split.values())) == len(pairs)
    for left, right in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        for side in [0, 1]:
            assert {canonical(pair[side]) for pair in split[left]}.isdisjoint(
                canonical(pair[side]) for pair in split[right])


def test_unsplittable_parallel_data_fails():
    with pytest.raises(ValueError, match='independent'):
        split_pairs([('first source', 'same target'), ('second source', 'same target')])


@pytest.mark.parametrize('text', ['my brother watches tv.', 'she bought a cd.', 'look at the sky.', 'he is shy.', 'she had had enough.', 'i know that that is true.'])
def test_display_postprocessing_preserves_real_final_words(text):
    from speech_engine.preprocessing import postprocess_en_output
    assert postprocess_en_output(text).lower() == text


def test_original_nmt_safe_checkpoint_roundtrip(tmp_path):
    import torch
    from speech_engine.bpe import BPEVocab
    from speech_engine.nmt import Seq2SeqNMT, save_checkpoint, load_checkpoint
    source = BPEVocab().train(['hola mundo'], 25)
    target = BPEVocab().train(['hello world'], 25)
    model = Seq2SeqNMT(source.vocab_size, target.vocab_size, d_model=16,
                      n_heads=2, n_layers=1, d_ff=32, dropout=0)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(model, source, target, path, meta={'direction': 'es→en', 'epoch': 2})
    restored, sv, tv, meta = load_checkpoint(path)
    assert sv.encode('hola mundo') == source.encode('hola mundo')
    assert tv.encode('hello world') == target.encode('hello world')
    assert meta['epoch'] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_nmt_training_predictions_do_not_depend_on_batch_padding():
    import torch
    from speech_engine.nmt import Seq2SeqNMT
    torch.manual_seed(11)
    model = Seq2SeqNMT(20, 20, d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0).eval()
    with torch.inference_mode():
        plain = model(torch.tensor([[1, 4, 2]]), torch.tensor([[1, 5, 6]]))
        padded = model(torch.tensor([[1, 4, 2, 0, 0, 0]]), torch.tensor([[1, 5, 6]]))
    assert torch.allclose(plain, padded, atol=1e-5)


def test_source_anchor_review_handles_grouped_numbers_without_altering_output():
    from speech_engine.preprocessing import normalize_es_input
    prepared=normalize_es_input('Costó $1000.')
    assert not prepared.coverage_failures('It cost 1,000 dollars.')
    assert prepared.coverage_failures('It cost 20 dollars.')


@pytest.mark.parametrize('contrast_weight',[0,.2])
def test_original_training_loop_can_shuffle_warm_start_from_first_epoch(monkeypatch,contrast_weight):
    import math
    import torch
    from scripts import phase7_train
    from speech_engine.nmt import Seq2SeqNMT
    model=Seq2SeqNMT(8,8,d_model=16,n_heads=2,n_layers=1,d_ff=32,dropout=0)
    examples=[([1,4,2],[1,5,2]),([1,6,2],[1,7,2])]*2
    shuffled=[]; callback=[]
    monkeypatch.setattr(phase7_train.random,'shuffle',lambda rows: shuffled.append(len(rows)))
    def completed(epoch,model,loss,*state):
        callback.append((epoch,math.isfinite(loss))); model.eval()
    phase7_train.train(model,examples,epochs=1,batch_size=2,lr=2e-4,device=torch.device('cpu'),
                       curriculum_epochs=0,epoch_callback=completed,source_contrast_weight=contrast_weight,accum_steps=1 if contrast_weight else 2)
    assert shuffled==[4] and callback==[(1,True)] and model.training


def test_decoding_ablation_can_preserve_repeated_tokens():
    import torch
    from torch import nn
    from speech_engine.nmt import Seq2SeqNMT
    class Decoder(nn.Module):
        def forward(self, ids, memory, **kwargs):
            logits=torch.full((ids.size(0),ids.size(1),8),-100.)
            logits[:,-1,4 if ids.size(1)<3 else 2]=0.
            return logits
    model=Seq2SeqNMT(8,8,d_model=8,n_heads=2,n_layers=1,d_ff=16,dropout=0)
    model.decoder=Decoder();model.output_proj=nn.Identity()
    source=torch.tensor([[1,5,2]])
    assert model.translate_beam(source,beam_size=2,min_len_ratio=0,
        eos_boost=False,repeat_guard=False,no_repeat_ngram=0)==[4,4]
    assert model.translate_beam(source,beam_size=2,min_len_ratio=0)!=[4,4]
    assert model.translate_beam(source,beam_size=2,min_len_ratio=0,repeat_exempt_ids={4})==[4,4]


def test_alignment_ranking_loss_ignores_batch_padding():
    import torch
    from scripts.prepare_translation_quality import losses
    from speech_engine.bpe import BPEVocab
    from speech_engine.nmt import Seq2SeqNMT
    torch.manual_seed(33)
    rows=[('hola','hello'),('hola mundo otra vez','hello world again')]
    sv=BPEVocab().train([s for s,t in rows],30)
    tv=BPEVocab().train([t for s,t in rows],30)
    model=Seq2SeqNMT(sv.vocab_size,tv.vocab_size,d_model=8,n_heads=2,n_layers=1,d_ff=16,dropout=0).eval()
    batch=losses(model,rows,sv,tv,torch.device('cpu'))
    individual=[losses(model,[row],sv,tv,torch.device('cpu'))[0] for row in rows]
    assert batch==pytest.approx(individual,abs=1e-5)


def test_numeric_decoder_policy_survives_checkpoint_serialization(tmp_path):
    from speech_engine.bpe import BPEVocab
    from speech_engine.nmt import Seq2SeqNMT,save_checkpoint,load_checkpoint
    sv=BPEVocab().train(['el año 2027'],25)
    tv=BPEVocab().train(['the year 2027'],25)
    model=Seq2SeqNMT(sv.vocab_size,tv.vocab_size,d_model=8,n_heads=2,n_layers=1,d_ff=16)
    path=tmp_path/'numeric.pt'
    save_checkpoint(model,sv,tv,path,meta={'decoding_policy':'preserve_numeric_fragments'})
    loaded,_,target,_=load_checkpoint(path)
    expected={i for i,t in target._id_to_token.items() if t.removesuffix('</w>').isdigit()}
    assert expected
    assert loaded._repeat_exempt_ids==expected
    save_checkpoint(model,sv,tv,path,meta={})
    legacy,_,_,_=load_checkpoint(path)
    assert not getattr(legacy,'_repeat_exempt_ids',set())


def test_installation_requires_matching_numeric_decoding_evaluation(tmp_path,monkeypatch):
    import hashlib,json,sys
    from speech_engine.bpe import BPEVocab
    from speech_engine.nmt import Seq2SeqNMT,save_checkpoint,load_checkpoint
    from scripts.install_local_translation import main
    corpus=tmp_path/'train.tsv';corpus.write_text('año 2027\tyear 2027\n',encoding='utf-8')
    sv=BPEVocab().train(['año 2027'],25);tv=BPEVocab().train(['year 2027'],25)
    model=Seq2SeqNMT(sv.vocab_size,tv.vocab_size,d_model=8,n_heads=2,n_layers=1,d_ff=16)
    checkpoint=tmp_path/'candidate.pt'
    save_checkpoint(model,sv,tv,checkpoint,meta={'training_corpus':str(corpus),
        'split_sha256':{'train':hashlib.sha256(corpus.read_bytes()).hexdigest()}})
    report=tmp_path/'evaluation.json';report.write_text(json.dumps({
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'decoder':'beam_4','decoding_options':{'allow_repeated_digits':True},'metrics':{}}))
    destination=tmp_path/'active.pt'
    args=['install','--checkpoint',str(checkpoint),'--destination',str(destination),'--reports',str(report)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='decoding policy'):
        main()
    assert not destination.exists()
    monkeypatch.setattr(sys,'argv',args+['--allow-repeated-digits'])
    main()
    restored,_,_,metadata=load_checkpoint(destination)
    assert metadata['decoding_policy']=='preserve_numeric_fragments'
    assert restored._repeat_exempt_ids


def test_source_contrast_objective_pushes_conditioning_and_ignores_padding():
    import torch
    from speech_engine.model_objectives import conditional_contrast_loss
    paired=torch.zeros(1,2,5,requires_grad=True)
    wrong=torch.zeros(1,2,5,requires_grad=True)
    target=torch.tensor([[4,0]])
    loss=conditional_contrast_loss(paired,wrong,target,torch.tensor([True]))
    assert loss.item()==pytest.approx(.1)
    loss.backward()
    assert paired.grad[0,0,4]<0 and wrong.grad[0,0,4]>0
    assert not paired.grad[:,1].any() and not wrong.grad[:,1].any()
    empty=conditional_contrast_loss(paired,wrong,target,torch.tensor([False]))
    assert empty.item()==0 and torch.isfinite(empty)
