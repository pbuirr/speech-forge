"""Regression probes for grammar-rule false alarms found in the 2026-10-12 review.

Each probe is correct text that a rule used to flag, or a true error that must
still be found. These are targeted behaviour checks, not a precision estimate.
"""
import pytest

from speech_engine.grammar import GrammarChecker, Severity


def rule_ids(text, lang='en'):
    return [e.rule_id for e in GrammarChecker().check(text, lang).errors]


@pytest.mark.parametrize('text', [
    "He wasn't there.",
    "Let it go.",
    "Does she want tea?",
    "They made it run faster.",
    "It belongs to long-standing tradition.",
    "This policy will effect real change.",
    "We met in May of last year.",
    "It cost him to many thousands of dollars in the end.",
    "I expect for the best outcome.",
    "He kept the might of his arm in reserve.",
    "They refused allies of whose coming they were not told.",
])
def test_correct_english_is_not_flagged_as_error(text):
    errors = [e for e in GrammarChecker().check(text, 'en').errors if e.severity != Severity.STYLE]
    assert errors == []


@pytest.mark.parametrize('text,rule', [
    ("He don't like it.", 'EN_SVA'),
    ("She have a car.", 'EN_SVA'),
    ("They has finished.", 'EN_SVA'),
    ("You should of called.", 'EN_CONFUSION'),
    ("We might of won.", 'EN_CONFUSION'),
    ("Whose going to the party?", 'EN_CONFUSION'),
    ("It is better then that.", 'EN_CONFUSION'),
    ("I ate to much.", 'EN_CONFUSION'),
    ("I saw a apple.", 'EN_ARTICLE_AN'),
])
def test_true_errors_still_found(text, rule):
    assert rule in rule_ids(text)


def test_style_hints_do_not_lower_the_score():
    result = GrammarChecker().check('The ball was thrown by the boy.', 'en')
    assert result.errors and all(e.severity == Severity.STYLE for e in result.errors)
    assert result.score == 1.0


def test_negative_concord_is_a_register_note_not_an_error():
    errors = GrammarChecker().check("I don't have nothing.", 'en').errors
    assert [e.severity for e in errors if e.rule_id == 'EN_DOUBLE_NEG'] == [Severity.STYLE]


def test_french_spacing_before_punctuation_is_not_flagged():
    assert 'EN_PUNCTUATION' not in rule_ids("Vraiment ? Oui ; d'accord !", 'fr')


def test_french_negation_matches_real_speech_and_suggests_correctly():
    errors = [e for e in GrammarChecker().check('Je sais pas. Il aime pas ça.', 'fr').errors if e.rule_id == 'FR_NEGATION']
    assert [e.suggestion for e in errors] == ['Je ne sais pas', "Il n'aime pas"]
    assert not [e for e in GrammarChecker().check('Je ne sais pas.', 'fr').errors if e.rule_id == 'FR_NEGATION']


def test_spanish_accent_only_on_question_words():
    text = '¿Sabes que vienes? ¿como estas? Creo que sí. ¿De donde eres?'
    errors = [e for e in GrammarChecker().check(text, 'es').errors if e.rule_id == 'ES_ACCENT']
    assert [text[slice(*e.span)] for e in errors] == ['como', 'donde']
    assert [e.suggestion for e in errors] == ['cómo', 'dónde']
