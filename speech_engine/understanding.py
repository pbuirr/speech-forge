"""Coordinate existing linguistic components around one original-text record.

This shares language choice, offsets, annotations, and evidence. It does not
train a multilingual representation or turn grammar heuristics into facts.
"""
from dataclasses import dataclass

from speech_engine.linguistic_resources import language_code
from speech_engine.linguistic_relationships import form_relationships
from speech_engine.dependency import syntax_payload


def detection_payload(result, evidence=None):
    return {"language": result.language, "name": result.name,
            "dialect": result.dialect, "script": result.script,
            "confidence": result.confidence, "is_mixed": result.is_mixed,
            "evidence_scope": "generated_text" if evidence else "supplied_text",
            "automatic_routing_allowed": evidence is None,
            "top": [{"code": m.code, "name": m.name, "confidence": m.confidence}
                    for m in result.top]}


def morphology_payload(result, engine):
    capabilities = engine.capabilities(result.lang)
    limited = capabilities['morphology'] == 'unavailable'
    tokens = [a.to_dict() for a in result.analyses]
    relationships = form_relationships(tokens)
    sentences = []
    for sentence_index, (left, right) in enumerate(result.sentence_ranges):
        span = [tokens[left]['span'][0], tokens[right-1]['span'][1]] if result.text is not None else None
        sentences.append({'sentence_index':sentence_index,'token_range':[left,right], 'span':span,
                          'text':result.text[slice(*span)] if span else None,
                          'method':result.sentence_boundary_method,
                          'status':'heuristic' if result.text is not None else 'supplied'})
        for token in tokens[left:right]:
            token['sentence_index'] = sentence_index
    for token in tokens:
        token['relationship_indices'] = []
    for link_index, link in enumerate(relationships):
        for index in link['token_indices']:
            tokens[index]['relationship_indices'].append(link_index)
    return {"text": result.text, "typology": engine.get_typology(result.lang),
            "limited": limited, "capabilities": capabilities,
            "analysis_note": "Corpus matches are observed annotations, not context-validated facts. Context choices are statistical predictions; fallback labels and affix analyses are heuristics. Ambiguous matches retain alternatives; confidence is uncalibrated.",
            "coverage": {"tokens": len(result.analyses),
                         "corpus_matched": sum(bool(a.candidates) for a in result.analyses),
                         "ambiguous": sum(a.method == 'corpus_ambiguous' for a in result.analyses),
                         "unavailable": sum(a.method == 'unavailable' for a in result.analyses)},
            "limited_note": (f"Linguistic morphology is unavailable for '{result.lang}'. "
                             "Tokens and predicted POS labels are shown; roots retain the surface form. "
                             "BPE pieces, when available, are statistical subwords rather than morphemes.") if limited else None,
            "tokens": tokens, "relationships": relationships, "sentences":sentences}


def grammar_payload(result, checker, tokens=None):
    base = language_code(result.lang)
    applicable = [(rule_id, languages) for rule_id, _, languages in checker.available_rules()
                  if '*' in languages or base in languages or result.lang in languages]
    language_specific = sum('*' not in languages for _, languages in applicable)
    errors = []
    for error in result.errors:
        indices = []
        if tokens is not None:
            left, right = error.span
            indices = [i for i, token in enumerate(tokens) if token.span is not None
                       and (token.span[0] < right and left < token.span[1]
                            or left == right and token.span[0] <= left < token.span[1])]
        errors.append({"message": error.message, "suggestion": error.suggestion,
                       "severity": error.severity, "category": error.category,
                       "rule_id": error.rule_id, "span": list(error.span),
                       "token_indices": indices, "status": "heuristic"})
    incomplete = bool(result.failed_rules)
    return {"is_clean": result.is_clean, "score": result.score if language_specific and not incomplete else None,
            "n_words": result.n_words, "errors": errors,
            "method": "pattern_rules", "limited": not language_specific or incomplete,
            "status": "incomplete" if incomplete else "limited" if not language_specific else "checked",
            "checked_rules": result.checked_rules, "failed_rules": result.failed_rules,
            "applicable_rules": [rule_id for rule_id, _ in applicable],
            "language_specific_rules": language_specific,
            "analysis_note": "Pattern-rule findings are heuristics. No findings does not establish grammatical correctness. The score is a rule penalty, not accuracy.",
            "limited_note": ("Some grammar checks failed; results are incomplete." if incomplete
                             else "Only general rules, if any, are available for this language." if not language_specific else None)}


@dataclass
class UnderstandingResult:
    text: str
    lang: str
    language_selection: str
    detection: object
    morphology: object
    grammar: object
    syntax: dict | None = None
    input_evidence: dict | None = None

    def to_dict(self, engine, checker):
        morph = morphology_payload(self.morphology, engine)
        gram = grammar_payload(self.grammar, checker, self.morphology.analyses)
        syntax={**self.syntax,'input_evidence':self.input_evidence} if self.syntax else None
        morph['syntax'] = syntax
        morph['input_evidence'] = gram['input_evidence'] = self.input_evidence
        for token in morph['tokens']:
            token['grammar_issue_indices'] = []
        for issue_index, issue in enumerate(gram['errors']):
            for token_index in issue['token_indices']:
                morph['tokens'][token_index]['grammar_issue_indices'].append(issue_index)
        return {"text": self.text, "lang": self.lang,
                "input_evidence": self.input_evidence,
                "language_selection": self.language_selection,
                "span_unit": "unicode_codepoints",
                "detection": detection_payload(self.detection,self.input_evidence),
                "morphology": morph, "grammar": gram, "syntax": syntax, "sentences":morph['sentences']}


class LinguisticPipeline:
    """Reuse caller-owned engines/resources; analyze each input only once."""
    def __init__(self, morphology, grammar, detector):
        self.morphology = morphology
        self.grammar = grammar
        self.detector = detector

    def analyze(self, text, lang=None, input_evidence=None):
        from speech_engine.input_evidence import input_evidence as canonical_evidence
        evidence=canonical_evidence(input_evidence)
        if not isinstance(text, str):
            raise ValueError('Text must be a string')
        detection = self.detector.detect(text)
        explicit=bool(lang and lang!='auto')
        if explicit:
            selected=lang;selection='explicit'
        elif evidence:
            selected=evidence.get('language')
            if not selected:raise ValueError('Select a source language for model-generated text; text detection cannot identify the original speech language')
            selection='source_configured'
        else:
            selected=detection.language;selection='detected'
        language_code(selected)  # Validate while preserving a supplied regional tag.
        selected = selected.lower()
        morphology = self.morphology.analyze(text, selected)
        syntax = syntax_payload(morphology, self.morphology.dependency_parsers)
        return UnderstandingResult(text, selected, selection,
                                   detection, morphology, self.grammar.check(text, selected), syntax, evidence)
