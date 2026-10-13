import json

import pytest

from speech_engine.flashcard import FlashcardDeck
from speech_engine.linguistic_evaluation import evaluate_lexicon, evaluate_context
from speech_engine.linguistic_resources import CorpusLexicon, read_conllu
from speech_engine.morphology import MorphologyEngine, POSTagger
from webapp import server
from test_morphology_integration import http_json


def test_english_clitics_preserve_surface_offsets_and_language_scope():
    engine = MorphologyEngine()
    text = "I can't believe she'd've said John’s."
    analyses = engine.analyze(text, 'en-US').analyses
    assert [a.surface for a in analyses] == ['I', 'ca', "n't", 'believe', 'she', "'d", "'ve", 'said', 'John', '’s', '.']
    assert all(text[a.span[0]:a.span[1]] == a.surface for a in analyses)
    assert [a.surface for a in engine.analyze("l’homme", 'fr').analyses] == ['l’homme']
    assert [a.surface for a in engine.analyze("rock’n’roll", 'en').analyses] == ['rock’n’roll']


def test_internal_script_joiners_are_not_punctuation():
    engine = MorphologyEngine()
    assert [a.surface for a in engine.analyze('می\u200cروم', 'fa').analyses] == ['می\u200cروم']
    assert [a.surface for a in engine.analyze('क्\u200dष', 'hi').analyses] == ['क्\u200dष']


def test_raw_text_evaluation_penalizes_boundaries_and_reports_exclusions(tmp_path):
    from speech_engine.linguistic_evaluation import evaluate_raw_text
    lexicon = compile_resource(tmp_path, row(1, 'e-mail', 'e-mail') + '\n')
    test = tmp_path / 'test.conllu'
    test.write_text('# text = e-mail\n' + row(1, 'e-mail', 'e-mail') + '\n'
                    + '# text = du\n' + row(1, 'de', 'de', 'ADP') + row(2, 'le', 'le', 'DET'), encoding='utf-8')
    report = evaluate_raw_text(lexicon, test)
    assert report['sentences'] == 2
    assert report['excluded_sentences'] == 1
    assert report['gold_tokens'] == 1
    assert report['predicted_tokens'] == 3
    assert report['token_f1'] == 0
    assert report['pos_accuracy_all_gold_tokens'] == 0
    with pytest.raises(ValueError, match='training source'):
        evaluate_raw_text(lexicon, tmp_path / 'train.conllu')


def row(token, form, lemma, pos="NOUN", feats="_", misc="_"):
    return f"{token}\t{form}\t{lemma}\t{pos}\t_\t{feats}\t0\troot\t_\t{misc}\n"


def compile_resource(tmp_path, text, lang="en"):
    path = tmp_path / "train.conllu"
    path.write_text(text, encoding="utf-8")
    return CorpusLexicon.from_conllu(path, lang, source_url="https://example.test/authored-fixture", license="CC0-1.0")


def test_import_preserves_provenance_and_does_not_invent_morphemes(tmp_path):
    lexicon = compile_resource(tmp_path, "# sent_id = authored-1\n" + row(1, "children", "child", feats="Number=Plur"))
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    result = engine.analyze_word("children", "en-US")
    assert result.root == "child"
    assert result.features.number == "Plur"
    assert result.morphemes == []

    assert result.to_dict()['pos_method'] == 'corpus_lookup'
    assert result.model_source['url'] == 'https://example.test/authored-fixture'
    assert result.confidence is None
    assert result.to_dict()["root_status"] == "corpus_attested"
    evidence = result.candidates[0]
    assert len(evidence["source"]["sha256"]) == 64
    assert evidence["examples"] == [{"sentence_id": "authored-1", "token_id": 1, "line": 2}]
    evidence["features"]["Number"] = "Changed"
    assert lexicon.lookup("children")[0]["features"]["Number"] == "Plur"


def test_ambiguous_lemma_and_pos_do_not_follow_frequency(tmp_path):
    text = row(1, "saw", "see", "VERB", "Tense=Past") + "\n"
    text += (row(1, "saw", "saw", "NOUN", "Number=Sing") + "\n") * 5
    lexicon = compile_resource(tmp_path, text)
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    result = engine.analyze_word("saw")
    assert result.method == "corpus_ambiguous"
    assert result.root == "saw"
    assert result.pos == "X"
    assert result.features.to_dict() == {}
    assert len(result.candidates) == 2
    assert sorted(c["observations"] for c in result.candidates) == [1, 5]
    assert FlashcardDeck().extract_from_text("saw", "en", engine) == []


def test_partial_consensus_keeps_only_shared_features(tmp_path):
    text = row(1, "read", "read", "VERB", "Tense=Past|VerbForm=Fin") + "\n"
    text += row(1, "read", "read", "VERB", "Tense=Pres|VerbForm=Fin")
    engine = MorphologyEngine()
    engine.add_lexicon(compile_resource(tmp_path, text))
    result = engine.analyze_word("read")
    assert result.root == "read"
    assert result.pos == "VERB"
    assert result.features.tense is None
    assert result.ud_features == {"VerbForm": "Fin"}
    assert result.to_dict()["root_status"] == "corpus_attested"
    assert result.method == "corpus_ambiguous"
    card = FlashcardDeck().extract_from_text("read", "en", engine)[0]
    assert "corpus-attested verb" in card.back
    assert "Tense" not in card.back


def test_no_blind_stemming_for_unknown_corpus_forms(tmp_path):
    engine = MorphologyEngine()
    engine.add_lexicon(compile_resource(tmp_path, row(1, "running", "run", "VERB")))
    result = engine.analyze_word("unbelievably")
    assert result.root == "unbelievably"
    assert result.method == "unavailable"
    assert result.morphemes == []
    assert result.to_dict()['pos_method'] == 'bigram_hmm'
    assert result.model_source['url'] == 'https://example.test/authored-fixture'


def test_reader_handles_multiword_nodes_missing_lemmas_and_codeswitching(tmp_path):
    text = "# sent_id = authored-mwt\n"
    text += "1-2\tal\t_\t_\t_\t_\t_\t_\t_\t_\n"
    text += row(1, "a", "a", "ADP") + row(2, "el", "el", "DET")
    text += "2.1\tcome\tcomer\tVERB\t_\t_\t_\t_\t0:root\t_\n"
    text += row(3, "unknown", "_", "X") + row(4, "hello", "hello", "INTJ", misc="Lang=en")
    lexicon = compile_resource(tmp_path, text, "es")
    assert lexicon.summary == {"words": 4, "missing_lemma": 1, "foreign_words": 1, "forms": 2, "analyses": 2}
    assert not lexicon.lookup("al") and not lexicon.lookup("come") and not lexicon.lookup("hello")


def test_unicode_normalization_preserves_case_and_offsets(tmp_path):
    lexicon = compile_resource(tmp_path, row(1, "café", "café"), "fr")
    engine = MorphologyEngine()
    engine.add_lexicon(lexicon)
    text = "  cafe\u0301, हूँ ‘l’école’ "
    result = engine.analyze(text, "fr")
    assert result.tokens == ["cafe\u0301", ",", "हूँ", "‘", "l’école", "’"]
    assert result.analyses[0].root == "café"
    assert lexicon.lookup("CAFÉ") == []
    for analysis in result.analyses:
        start, end = analysis.span
        assert text[start:end] == analysis.surface
    assert MorphologyEngine().analyze("か\u3099。", "ja").tokens == ["か\u3099", "。"]


@pytest.mark.parametrize("bad", [
    "1 cats cat NOUN _ _ 0 root _ _",
    row(2, "cats", "cat"),
    row(1, "cats", "cat", "INVALID"),
    row(1, "cats", "cat", feats="Number=Plur|Number=Sing"),
    row(1, "cats", "cat", feats="Number="),
    "2-1\tword\t_\t_\t_\t_\t_\t_\t_\t_\n",
])
def test_malformed_data_is_rejected_with_line_number(bad):
    with pytest.raises(ValueError, match="Line 1"):
        list(read_conllu(bad))


def test_resource_roundtrip_and_atomic_loading(tmp_path):
    lexicon = compile_resource(tmp_path, row(1, "children", "child", feats="Number=Plur"))
    resource_dir = tmp_path / "resources"
    lexicon.save(resource_dir / "en.json")
    assert CorpusLexicon.load(resource_dir / "en.json").lookup("children") == lexicon.lookup("children")
    engine = MorphologyEngine()
    summaries = engine.load_resources(resource_dir)
    assert summaries[0]["forms"] == 1
    (resource_dir / "broken.json").write_text("{}")
    fresh = MorphologyEngine()
    with pytest.raises(ValueError, match="broken.json"):
        fresh.load_resources(resource_dir)
    assert fresh.analyze_word("children").method == "english_affix_rules"


def test_invalid_compiled_entries_are_not_accepted(tmp_path):
    lexicon = compile_resource(tmp_path, row(1, "cats", "cat"))
    path = tmp_path / "resource.json"
    lexicon.save(path)
    data = json.loads(path.read_text())
    data["entries"][0]["count"] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid lexical entry"):
        CorpusLexicon.load(path)


def test_heldout_metrics_keep_unknowns_and_ambiguity_in_denominator(tmp_path):
    train = row(1, "cats", "cat", feats="Number=Plur") + "\n"
    train += row(1, "saw", "see", "VERB") + "\n" + row(1, "saw", "saw", "NOUN")
    lexicon = compile_resource(tmp_path, train)
    test_path = tmp_path / "test.conllu"
    test_path.write_text(row(1, "cats", "cat", feats="Number=Plur") + row(2, "saw", "see", "VERB") + row(3, "unknown", "unknown"))
    report = evaluate_lexicon(lexicon, test_path)
    assert report["coverage"] == 2 / 3
    assert report["accuracy_on_decisions"] == 1
    assert report["accuracy_all_tokens"] == 1 / 3
    assert report["candidate_recall"] == 2 / 3
    assert report["ambiguous_tokens"] == 1
    with pytest.raises(ValueError, match="training source"):
        evaluate_lexicon(lexicon, tmp_path / "train.conllu")


def test_http_returns_evidence_and_spans(tmp_path, monkeypatch):
    engine = MorphologyEngine()
    engine.add_lexicon(compile_resource(tmp_path, row(1, "children", "child", feats="Number=Plur")))
    monkeypatch.setattr(server, "morph", engine)
    result = http_json("/api/morphology", {"text": "children unknown", "lang": "en-US"})
    assert result["coverage"]["corpus_matched"] == 1
    assert result["coverage"]["unavailable"] == 1
    assert result["tokens"][0]["span"] == [0, 8]
    assert result["tokens"][0]["ud_features"] == {"Number": "Plur"}
    assert result["tokens"][0]["candidates"][0]["source"]["declared_license"] == "CC0-1.0"


def test_no_decisions_reports_null_accuracy(tmp_path):
    lexicon = compile_resource(tmp_path, row(1, "known", "know", "VERB"))
    test_path = tmp_path / "test.conllu"
    test_path.write_text(row(1, "unknown", "unknown"))
    assert evaluate_lexicon(lexicon, test_path)["accuracy_on_decisions"] is None


def test_trained_pos_model_uses_context_and_isolates_languages():
    tagger = POSTagger()
    verb = [("They", "PRON"), ("saw", "VERB"), ("wood", "NOUN")]
    noun = [("The", "DET"), ("saw", "NOUN")]
    tagger.fit([verb, noun] * 10, "en")
    assert tagger.tag(["They", "saw", "wood"], "en-US") == verb
    assert tagger.tag(["The", "saw"], "en") == noun
    assert not tagger.has_model("es")
    # Retraining invalidates the prepared decoder rather than using stale counts.
    tagger.fit([[('saw', 'INTJ')]] * 100, "es")
    assert tagger.tag(["saw"], "es") == [("saw", "INTJ")]
    assert tagger.tag(["The", "saw"], "en") == noun


def test_context_predictions_keep_all_evidence_and_remain_predictions(tmp_path):
    train = row(1, "They", "they", "PRON") + row(2, "saw", "see", "VERB", "Tense=Past") + row(3, "wood", "wood", "NOUN") + "\n"
    train += row(1, "The", "the", "DET") + row(2, "saw", "saw", "NOUN", "Number=Sing") + "\n"
    lexicon = compile_resource(tmp_path, train)
    path = tmp_path / "compiled.json"
    lexicon.save(path)
    engine = MorphologyEngine()
    engine.add_lexicon(CorpusLexicon.load(path))
    analysis = engine.analyze("They saw wood").analyses[1]
    assert analysis.root == "see"
    assert analysis.pos == "VERB"
    assert analysis.method == "corpus_context"
    assert analysis.to_dict()["root_status"] == "predicted"
    assert analysis.to_dict()["pos_status"] == "predicted"
    assert analysis.confidence is None
    assert len(analysis.candidates) == 2
    assert len(analysis.selected_candidate_indices) == 1
    card = FlashcardDeck().extract_from_text("They saw wood", "en", engine)[0]
    assert "predicted lemma: see" in card.back
    assert "corpus-attested verb" not in card.back
    assert card.analysis["method"] == "corpus_context"
    deck = FlashcardDeck()
    deck.add(card.front, card.back, analysis=card.analysis)
    saved = tmp_path / 'deck.json'
    deck.save(saved)
    assert FlashcardDeck.load(saved).all()[0].analysis == card.analysis
    assert engine.analyze("The saw").analyses[1].root == "saw"
    assert engine.analyze_word("saw").pos == "X"


def test_upstream_tokens_do_not_fabricate_offsets():
    result = MorphologyEngine().analyze_tokens(["hello", "world"])
    assert result.tokens == ["hello", "world"]
    assert all(a.span is None for a in result.analyses)
    with pytest.raises(ValueError):
        MorphologyEngine().analyze_tokens([" "])


def test_unknown_lexical_word_cannot_become_punctuation():
    tagger = POSTagger()
    tagger.fit([[('Hello', 'INTJ'), ('.', 'PUNCT')], [('very', 'ADV'), ('quickly', 'ADV')]])
    assert tagger.tag(['unbelievably'])[0][1] == 'ADV'


def test_context_evaluator_uses_heldout_labels_only_for_scoring(tmp_path):
    train = row(1, "They", "they", "PRON") + row(2, "saw", "see", "VERB") + "\n"
    train += row(1, "The", "the", "DET") + row(2, "saw", "saw", "NOUN")
    lexicon = compile_resource(tmp_path, train)
    test = tmp_path / "test.conllu"
    # Deliberately different gold tags must affect metrics, not decoder behavior.
    test.write_text(row(1, "They", "they", "PRON") + row(2, "saw", "see", "NOUN"))
    report = evaluate_context(lexicon, test)
    assert report['context_pos_accuracy'] == 0.5
    assert report['context_lemma_accuracy_on_decisions'] == 1.0


def test_browser_card_save_keeps_analysis_and_existing_manual_cards(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'DECK_PATH', tmp_path / 'deck.json')
    manual = http_json('/api/add_card', {'front': 'cat', 'back': 'gato'})['card']
    assert manual['tags'] == ['manual'] and manual['analysis'] is None
    analysis = {'method': 'corpus_context', 'candidates': [{'lemma': 'see'}]}
    generated = http_json('/api/add_card', {'front': 'saw', 'back': 'predicted lemma: see',
                                         'analysis': analysis, 'example': 'They saw wood'})['card']
    assert generated['tags'] == ['auto', 'needs-review']
    assert generated['analysis'] == analysis
    assert generated['example'] == 'They saw wood'
    assert len(FlashcardDeck.load(server.DECK_PATH)) == 2
