import pytest

from speech_engine.grammar import ArticleChoiceRule, GrammarChecker, GrammarError, GrammarRule
from speech_engine.understanding import grammar_payload
from test_morphology_integration import http_json
from webapp import server


class BrokenRule(GrammarRule):
    rule_id = 'AUTHORED_BROKEN'
    category = 'test'
    languages = ['en']

    def check(self, text):
        raise RuntimeError('Internal details must not be exposed')


class InvalidSpanRule(BrokenRule):
    def check(self, text):
        return [GrammarError('test', None, (0, len(text) + 1), 'error', self.rule_id, self.category)]


@pytest.mark.parametrize('rule,error_type', [(BrokenRule(), 'RuntimeError'), (InvalidSpanRule(), 'ValueError')])
def test_failed_checks_are_incomplete_without_discarding_successful_findings(rule, error_type):
    checker = GrammarChecker([rule, ArticleChoiceRule()])
    result = checker.check('a apple.', 'en')
    assert result.errors[0].rule_id == 'EN_ARTICLE_AN'
    assert result.checked_rules == ['EN_ARTICLE_AN']
    assert result.score is None
    assert 'incomplete' in result.summary()
    assert result.failed_rules == [{'rule_id': 'AUTHORED_BROKEN', 'error_type': error_type}]
    payload = grammar_payload(result, checker)
    assert payload['status'] == 'incomplete'
    assert payload['score'] is None
    assert not payload['is_clean']
    assert 'Internal details' not in str(payload)


def test_failed_rule_without_findings_cannot_be_clean(monkeypatch):
    checker = GrammarChecker([BrokenRule()])
    result = checker.check('This sentence has no returned findings.', 'en')
    assert not result.is_clean
    assert 'incomplete' in result.summary()
    monkeypatch.setattr(server, 'checker', checker)
    payload = http_json('/api/analyze', {'text': 'The cat sleeps.', 'lang': 'en'})
    assert payload['grammar']['status'] == 'incomplete'
    assert payload['morphology']['tokens']
    assert payload['grammar']['score'] is None


def test_successful_default_rules_have_valid_offsets_and_no_failures():
    checker = GrammarChecker()
    for text, lang in [('a apple.', 'en'), ('Yo tambien.', 'es'), ('Je sais pas.', 'fr'), ('猫。', 'ja')]:
        result = checker.check(text, lang)
        assert not result.failed_rules
        assert all(0 <= error.span[0] <= error.span[1] <= len(text) for error in result.errors)
