import pytest
from speech_engine.dependency import DependencyParser, read_dependencies, projective


def tree_text():
    return "1\tCats\tcat\tNOUN\t_\tNumber=Plur\t2\tnsubj\t_\t_\n2\tsleep\tsleep\tVERB\t_\tTense=Pres\t0\troot\t_\t_\n"


def test_original_parser_learns_tree_and_persists(tmp_path):
    path = tmp_path / "train.conllu"
    path.write_text(tree_text())
    sentences = list(read_dependencies(path))
    assert projective(sentences[0])
    model = DependencyParser("en")
    model.fit(sentences * 10, epochs=3)
    tokens = [
        {"surface": w.form, "pos": w.pos, "ud_features": dict(w.features)}
        for w in sentences[0].words
    ]
    arcs = model.parse(tokens)
    assert [(a["head_token_index"], a["relation"]) for a in arcs] == [(1, "nsubj"), (None, "root")]
    assert all(a["confidence"] is None and a["status"] == "predicted" for a in arcs)
    target = tmp_path / "model.json"
    model.save(target)
    assert DependencyParser.load(target).parse(tokens) == arcs
    assert model.parse([]) == []


@pytest.mark.parametrize("replacement", ["1\tnsubj", "3\tnsubj", "0\tnsubj"])
def test_dependency_validation_rejects_bad_head_or_root(tmp_path, replacement):
    path = tmp_path / "invalid.conllu"
    path.write_text(tree_text().replace("2\tnsubj", replacement))
    with pytest.raises(ValueError):
        list(read_dependencies(path))


def test_decoder_outputs_single_root_without_cycles():
    model = DependencyParser("tr")
    model.labels = ["dep", "root"]
    model.weights = {}
    arcs = model.parse([{"surface": "ev", "pos": "NOUN", "ud_features": {}}] * 30)
    assert sum(a["head_token_index"] is None for a in arcs) == 1
    for i in range(len(arcs)):
        seen = {i}
        head = arcs[i]["head_token_index"]
        while head is not None:
            assert head not in seen
            seen.add(head)
            head = arcs[head]["head_token_index"]


def test_syntax_pipeline_respects_offsets_sentence_limits_and_learning(tmp_path):
    from speech_engine.morphology import MorphologyEngine
    from speech_engine.grammar import GrammarChecker
    from speech_engine.language_detector import LanguageDetector
    from speech_engine.understanding import LinguisticPipeline
    from speech_engine.flashcard import FlashcardDeck
    from speech_engine.dependency import syntax_payload

    path = tmp_path / "train.conllu"
    path.write_text(tree_text())
    model = DependencyParser("en")
    model.fit(read_dependencies(path))
    engine = MorphologyEngine()
    engine.dependency_parsers["en"] = model
    checker = GrammarChecker()
    result = LinguisticPipeline(engine, checker, LanguageDetector()).analyze(
        "Cats sleep. Cats sleep!", "en"
    )
    arcs = result.syntax["arcs"]
    assert sum(a["head_token_index"] is None for a in arcs) == 2
    assert all(
        result.text[slice(*a["span"])] == result.morphology.analyses[a["token_index"]].surface
        for a in arcs
    )
    cards = FlashcardDeck().extract_from_understanding(result, engine, checker)
    assert all(
        c.analysis["context"]["syntax"]["status"] == "experimental_prediction" for c in cards
    )
    assert any(c.analysis["context"]["syntax"]["arcs"] for c in cards)
    assert syntax_payload(result.morphology, {}, maximum_tokens=2)["status"] == "unavailable"
    assert (
        syntax_payload(result.morphology, {"en": model}, maximum_tokens=2)["status"]
        == "partial_experimental_prediction"
    )
    assert syntax_payload(result.morphology, {"en": model}, maximum_tokens=2)["arcs"] == []


def test_model_loader_rejects_nonfinite_scores(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "version": DependencyParser.version,
                "language": "en",
                "source": {},
                "labels": ["dep", "root"],
                "forms": [],
                "weights": {"bias": {"S": float("nan")}},
            }
        )
    )
    with pytest.raises(ValueError, match="scores"):
        DependencyParser.load(path)


def test_training_on_predicted_evidence_preserves_gold_tree(tmp_path):
    path = tmp_path / "train.conllu"
    path.write_text(tree_text())
    sentences = list(read_dependencies(path)) * 10
    # Deliberately wrong POS evidence: dependency supervision remains unchanged.
    evidence = [
        [{"surface": w.form, "pos": "NOUN", "ud_features": {}} for w in s.words] for s in sentences
    ]
    model = DependencyParser("tr")
    model.fit(sentences, token_sequences=evidence)
    assert [(a["head_token_index"], a["relation"]) for a in model.parse(evidence[0])] == [
        (1, "nsubj"),
        (None, "root"),
    ]
    with pytest.raises(ValueError, match="align"):
        model.fit(sentences, token_sequences=evidence[:-1])
    evidence[0][0]["surface"] = "different"
    with pytest.raises(ValueError, match="align"):
        model.fit(sentences, token_sequences=evidence)


def test_production_training_command_binds_shared_analyzer(tmp_path, monkeypatch):
    import json
    import sys
    from scripts.analysis.train_dependency import main
    from speech_engine.linguistic_resources import CorpusLexicon

    path = tmp_path / "train.conllu"
    path.write_text(tree_text())
    resources = tmp_path / "resources"
    resources.mkdir()
    CorpusLexicon.from_conllu(
        path, "en", source_url="https://example.test/authored", license="authored"
    ).save(resources / "en.json")
    output = tmp_path / "model.json"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dependency",
            "--corpus",
            str(path),
            "--language",
            "en",
            "--source-url",
            "https://example.test/authored",
            "--license",
            "authored",
            "--output",
            str(output),
            "--report",
            str(report),
            "--resources",
            str(resources),
        ],
    )
    main()
    model = DependencyParser.load(output)
    assert model.source["training_evidence"] == "production_in_sample"
    assert len(model.source["pos_model_sha256"]) == 64
    assert json.loads(report.read_text())["projective_training_sentences"] == 1


def test_syntax_resource_binding_and_http_integration(tmp_path, monkeypatch):
    from hashlib import sha256
    from speech_engine.linguistic_resources import CorpusLexicon
    from speech_engine.morphology import MorphologyEngine
    from speech_engine.dependency import load_dependency_models
    from webapp import server
    from test_morphology_integration import http_json

    path = tmp_path / "train.conllu"
    path.write_text(tree_text())
    engine = MorphologyEngine()
    engine.add_lexicon(
        CorpusLexicon.from_conllu(
            path, "en", source_url="https://example.test/authored", license="authored"
        )
    )
    model = DependencyParser("en", {"training_sha256": sha256(path.read_bytes()).hexdigest()})
    model.fit(read_dependencies(path))
    directory = tmp_path / "syntax"
    directory.mkdir()
    model.save(directory / "en.json")
    assert load_dependency_models(engine, directory) == ["en"]
    monkeypatch.setattr(server, "morph", engine)
    payload = http_json("/api/analyze", {"text": "Cats sleep.", "lang": "en-US"})
    assert payload["syntax"]["status"] == "experimental_prediction"
    assert len(payload["syntax"]["arcs"]) == len(payload["morphology"]["tokens"])
    assert (
        http_json("/api/analyze", {"text": "evler", "lang": "tr"})["syntax"]["status"]
        == "unavailable"
    )
    model.source["pos_model_sha256"] = "0" * 64
    model.save(directory / "en.json")
    with pytest.raises(ValueError, match="POS model"):
        load_dependency_models(engine, directory)
    model.source["pos_model_sha256"] = engine.tagger.model_fingerprint("en")
    model.save(directory / "en.json")
    assert load_dependency_models(engine, directory) == ["en"]
    model.source["training_sha256"] = "0" * 64
    model.save(directory / "en.json")
    with pytest.raises(ValueError, match="source"):
        load_dependency_models(engine, directory)
