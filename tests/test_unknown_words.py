from collections import Counter
import math
from speech_engine.unknown_words import UnknownWordModel
from speech_engine.morphology import POSTagger


def test_learned_suffixes_and_case_without_language_rules():
    model = UnknownWordModel(
        {
            "cats": Counter(NOUN=1),
            "dogs": Counter(NOUN=1),
            "walking": Counter(VERB=1),
            "talking": Counter(VERB=1),
            "Mary": Counter(PROPN=1),
            "Paris": Counter(PROPN=1),
        }
    )
    tags = ["NOUN", "VERB", "PROPN"]
    assert model.scores("singing", tags)["VERB"] > model.scores("singing", tags)["NOUN"]
    assert model.scores("Susan", tags)["PROPN"] > model.scores("susan", tags)["PROPN"]
    assert all(math.isfinite(v) for v in model.scores("未知", tags).values())


def test_model_is_shared_algorithm_with_language_specific_training():
    tagger = POSTagger()
    tagger.fit([[("walking", "VERB"), ("talking", "VERB"), ("cats", "NOUN")]], "en")
    tagger.fit([[("evler", "NOUN"), ("kitaplar", "NOUN"), ("geldi", "VERB")]], "tr")
    assert tagger.tag(["gitti"], "tr")[0][1] == "VERB"
    assert tagger.tag(["singing"], "en")[0][1] == "VERB"
    first = tagger._unknown_models["tr"]
    tagger.fit([[("okudu", "VERB")]], "tr")
    tagger.tag(["gitti"], "tr")
    assert tagger._unknown_models["tr"] is not first


def test_frequent_types_do_not_overwhelm_rare_word_training():
    model = UnknownWordModel({"the": Counter(DET=10000), "cats": Counter(NOUN=1)})
    assert model.prior["DET"] == 0
    assert model.prior["NOUN"] == 1


def test_case_evidence_is_soft_and_preserves_turkic_distinctions():
    model = UnknownWordModel({"ırmak": Counter(NOUN=5), "izmir": Counter(PROPN=5)}, language="tr")
    tags = ["NOUN", "PROPN", "VERB"]
    assert model.case_key("IRMAK") == "ırmak"
    assert model.case_key("İZMİR") == "izmir"
    assert model.case_key("I") != model.case_key("i")
    assert model.scores("IRMAK", tags)["NOUN"] > model.scores("IRMAK", tags)["PROPN"]
    assert all(math.isfinite(v) for v in model.scores("IRMAK", tags).values())


def test_alphanumeric_and_joiner_initial_words_keep_model_predictions():
    from speech_engine.morphology import MorphologyEngine

    engine = MorphologyEngine()
    engine.tagger.fit([[("2cats", "NOUN"), ("\u200dname", "PROPN")]], "en")
    for surface in ["3dogs", "\u200dnewname"]:
        result = engine.analyze_tokens([surface], "en").analyses[0]
        assert result.pos not in {"NUM", "PUNCT", "SYM"}
        assert result.pos_uncertainty["applies_to_returned_pos"]
