"""Corpus-attested lexical analyses with provenance and explicit ambiguity.

CoNLL-U supplies observed lemmas, UPOS, and features, not morpheme boundaries.
This module does not parse new sentences or establish grammatical truth. It
indexes annotations and preserves all distinct interpretations of each form.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterator

UPOS = frozenset(
    "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split()
)
_FEATURE_NAME = re.compile(r"[A-Z][A-Za-z0-9]*(?:\[[a-z0-9]+\])?\Z")
_FEATURE_VALUE = re.compile(r"[A-Z0-9][A-Za-z0-9]*\Z")


def language_code(lang: str) -> str:
    """Normalize a base language code; dialect-specific resources remain future work."""
    if not isinstance(lang, str) or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", lang):
        raise ValueError(f"Invalid language tag: {lang!r}")
    return lang.lower().split("-")[0]


def normalized_form(form: str) -> str:
    # Preserve case: Turkish I/i and proper-name distinctions are not universal.
    return unicodedata.normalize("NFC", form)


def parse_features(raw: str) -> tuple[tuple[str, str], ...]:
    if raw == "_":
        return ()
    features = {}
    for item in raw.split("|"):
        key, separator, value = item.partition("=")
        if (
            not separator
            or not _FEATURE_NAME.fullmatch(key)
            or not all(_FEATURE_VALUE.fullmatch(v) for v in value.split(","))
            or key in features
        ):
            raise ValueError(f"Malformed or duplicate feature: {item!r}")
        features[key] = ",".join(sorted(set(value.split(","))))
    return tuple(sorted(features.items()))


@dataclass(frozen=True)
class AnnotatedWord:
    form: str
    lemma: str | None
    pos: str
    features: tuple[tuple[str, str], ...]
    sentence_id: str
    token_id: int
    line: int
    language: str | None = None


def read_conllu(text: str) -> Iterator[AnnotatedWord]:
    """Read basic word annotations, skipping multiword rows and empty nodes.

    Validate the fields used for lexical import. This is not the full UD tree
    validator: dependency graphs and range coverage are outside this reader.
    """
    sentence_number = 1
    sentence_id = "1"
    previous_id = 0
    for line_number, line in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        if not line.strip():
            if previous_id:
                sentence_number += 1
            sentence_id = str(sentence_number)
            previous_id = 0
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                key, sep, value = line[1:].partition("=")
                if key.strip() == "sent_id" and sep and value.strip():
                    sentence_id = value.strip()
            continue
        columns = line.split("\t")
        if len(columns) != 10 or any(not c for c in columns):
            raise ValueError(f"Line {line_number}: expected 10 nonempty tab-separated fields")
        token_id, form, lemma, pos, _, feats, _, _, _, misc = columns
        if re.fullmatch(r"[1-9]\d*-[1-9]\d*", token_id):
            start, end = map(int, token_id.split("-"))
            if start >= end:
                raise ValueError(f"Line {line_number}: invalid multiword range")
            continue
        if re.fullmatch(r"\d+\.[1-9]\d*", token_id):
            continue
        if not re.fullmatch(r"[1-9]\d*", token_id) or int(token_id) != previous_id + 1:
            raise ValueError(f"Line {line_number}: word IDs must be consecutive from 1")
        previous_id = int(token_id)
        if pos not in UPOS:
            raise ValueError(f"Line {line_number}: unknown UPOS {pos!r}")
        try:
            features = parse_features(feats)
        except ValueError as error:
            raise ValueError(f"Line {line_number}: {error}") from error
        token_language = None
        for item in misc.split("|"):
            if item.startswith("Lang="):
                token_language = language_code(item[5:])
        yield AnnotatedWord(
            normalized_form(form),
            None if lemma == "_" else normalized_form(lemma),
            pos,
            features,
            sentence_id,
            previous_id,
            line_number,
            token_language,
        )


class CorpusLexicon:
    """Exact-form index. Annotation counts indicate observations, not probabilities."""

    schema_version = 1

    def __init__(
        self,
        lang: str,
        source: dict,
        entries: list[dict],
        summary: dict,
        transitions: list | None = None,
    ):
        self.lang = language_code(lang)
        self.source = dict(source)
        self.summary = dict(summary)
        self.transitions = transitions or []
        self._index: dict[str, list[dict]] = defaultdict(list)
        for entry in entries:
            self._index[entry["form"]].append(entry)

    @classmethod
    def from_conllu(
        cls, path: str | Path, lang: str, *, source_url: str, license: str
    ) -> CorpusLexicon:
        """Import a local corpus; caller supplies its source and declared license."""
        if not source_url.strip() or not license.strip():
            raise ValueError("Source URL and declared license are required")
        path = Path(path)
        raw = path.read_bytes()
        lang = language_code(lang)
        grouped = {}
        transitions = Counter()
        previous_tag = "BOS"
        words = missing_lemma = foreign = 0
        for word in read_conllu(raw.decode("utf-8")):
            if word.token_id == 1 and previous_tag != "BOS":
                transitions[(previous_tag, "EOS")] += 1
                previous_tag = "BOS"
            words += 1
            if word.language and word.language != lang:
                if previous_tag != "BOS":
                    transitions[(previous_tag, "EOS")] += 1
                previous_tag = "BOS"
                foreign += 1
                continue
            transitions[(previous_tag, word.pos)] += 1
            previous_tag = word.pos
            if word.lemma is None:
                missing_lemma += 1
                continue
            key = (word.form, word.lemma, word.pos, word.features)
            if key not in grouped:
                grouped[key] = {
                    "form": word.form,
                    "lemma": word.lemma,
                    "pos": word.pos,
                    "features": dict(word.features),
                    "count": 0,
                    "examples": [],
                }
            entry = grouped[key]
            entry["count"] += 1
            # Bound provenance examples per candidate to avoid corpus-sized API output.
            if len(entry["examples"]) < 3:
                entry["examples"].append(
                    {"sentence_id": word.sentence_id, "token_id": word.token_id, "line": word.line}
                )
        source = {
            "name": path.name,
            "url": source_url,
            "declared_license": license,
            "sha256": sha256(raw).hexdigest(),
        }
        entries = sorted(
            grouped.values(),
            key=lambda e: (e["form"], e["lemma"], e["pos"], sorted(e["features"].items())),
        )
        summary = {
            "words": words,
            "missing_lemma": missing_lemma,
            "foreign_words": foreign,
            "forms": len({e["form"] for e in entries}),
            "analyses": len(entries),
        }
        if previous_tag != "BOS":
            transitions[(previous_tag, "EOS")] += 1
        return cls(
            lang,
            source,
            entries,
            summary,
            [[left, right, count] for (left, right), count in sorted(transitions.items())],
        )

    def pos_statistics(self) -> tuple[Counter, dict[str, Counter]]:
        """Return independent training counts for the statistical POS tagger."""
        transitions = Counter({(left, right): count for left, right, count in self.transitions})
        emissions = defaultdict(Counter)
        for form, entries in self._index.items():
            for entry in entries:
                emissions[form][entry["pos"]] += entry["count"]
        return transitions, emissions

    def lookup(self, form: str) -> list[dict]:
        # Return independent values; consumers cannot mutate the index.
        return [
            dict(
                lemma=e["lemma"],
                pos=e["pos"],
                features=dict(e["features"]),
                observations=e["count"],
                source=dict(self.source),
                examples=[dict(x) for x in e["examples"]],
            )
            for e in self._index.get(normalized_form(form), [])
        ]

    def save(self, path: str | Path) -> None:
        data = {
            "schema_version": self.schema_version,
            "language": self.lang,
            "source": self.source,
            "summary": self.summary,
            "transitions": self.transitions,
            "entries": [e for entries in self._index.values() for e in entries],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CorpusLexicon:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != cls.schema_version
        ):
            raise ValueError("Unsupported linguistic resource schema")
        source = data.get("source", {})
        if (
            not isinstance(source, dict)
            or not all(
                isinstance(source.get(k), str) and source[k].strip()
                for k in ("name", "url", "declared_license", "sha256")
            )
            or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
        ):
            raise ValueError("Resource needs source, declared license, and corpus checksum")
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Resource entries must be a list")
        seen = set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not all(
                    isinstance(entry.get(k), str) and entry[k] for k in ("form", "lemma", "pos")
                )
                or entry["pos"] not in UPOS
                or not isinstance(entry.get("features"), dict)
                or type(entry.get("count")) is not int
                or entry["count"] <= 0
            ):
                raise ValueError("Invalid lexical entry")
            features = parse_features(
                "|".join(f"{k}={v}" for k, v in entry["features"].items()) or "_"
            )
            entry["form"] = normalized_form(entry["form"])
            entry["lemma"] = normalized_form(entry["lemma"])
            key = (entry["form"], entry["lemma"], entry["pos"], features)
            if key in seen:
                raise ValueError("Duplicate lexical analysis")
            seen.add(key)
            examples = entry.get("examples")
            if not isinstance(examples, list) or not 1 <= len(examples) <= 3:
                raise ValueError("Each lexical entry requires 1–3 source examples")
            for example in examples:
                if (
                    not isinstance(example, dict)
                    or not isinstance(example.get("sentence_id"), str)
                    or not example["sentence_id"]
                    or any(
                        type(example.get(k)) is not int or example[k] <= 0
                        for k in ("token_id", "line")
                    )
                ):
                    raise ValueError("Invalid source example")
        summary = {
            "forms": len({e["form"] for e in entries}),
            "analyses": len(entries),
            "observations": sum(e["count"] for e in entries),
        }
        transitions = data.get("transitions", [])
        if not isinstance(transitions, list):
            raise ValueError("POS transitions must be a list")
        seen_transitions = set()
        for edge in transitions:
            if (
                not isinstance(edge, list)
                or len(edge) != 3
                or not isinstance(edge[0], str)
                or not isinstance(edge[1], str)
                or edge[0] not in UPOS | {"BOS"}
                or edge[1] not in UPOS | {"EOS"}
                or type(edge[2]) is not int
                or edge[2] <= 0
                or tuple(edge[:2]) in seen_transitions
            ):
                raise ValueError("Invalid POS transition")
            seen_transitions.add(tuple(edge[:2]))
        return cls(data.get("language", ""), source, entries, summary, transitions)
