"""Arc-standard transition dependency parser trained on Universal Dependencies."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from speech_engine.linguistic_resources import read_conllu, language_code
from speech_engine.sentence_boundaries import sentence_ranges


@dataclass
class DependencySentence:
    words: list
    heads: list
    labels: list


def read_dependencies(path):
    """Validate basic UD trees separately from the lexical resource reader."""
    text = Path(path).read_text(encoding="utf-8")
    for block in text.strip().split("\n\n"):
        words = list(read_conllu(block))
        if not words:
            continue
        rows = [
            line.split("\t")
            for line in block.splitlines()
            if not line.startswith("#") and line.split("\t")[0].isdigit()
        ]
        heads = [int(row[6]) for row in rows]
        labels = [row[7] for row in rows]
        n = len(words)
        if len(heads) != n or heads.count(0) != 1:
            raise ValueError("Basic dependency tree requires exactly one root")
        for i, head in enumerate(heads, 1):
            if head < 0 or head > n or head == i or (head == 0) != (labels[i - 1] == "root"):
                raise ValueError("Invalid dependency head/root label")
        for i, head in enumerate(heads, 1):
            seen = {i}
            current = head
            while current:
                if current in seen:
                    raise ValueError("Dependency cycle")
                seen.add(current)
                current = heads[current - 1]
        yield DependencySentence(words, heads, labels)


def projective(sentence):
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(sentence.heads, 1)]
    return not any(a < c < b < d or c < a < d < b for a, b in arcs for c, d in arcs)


class DependencyParser:
    version = "arc_standard_averaged_v2"

    def __init__(self, language, source=None):
        self.language = language_code(language)
        self.source = source or {}
        self.weights = {}
        self.labels = []
        self.forms = set()

    def features(self, tokens, stack, cursor, heads):
        positions = [
            stack[-1] if stack else -1,
            stack[-2] if len(stack) > 1 else -1,
            cursor if cursor < len(tokens) else -1,
        ]
        features = ["bias"]
        tags = []
        for name, position in zip(["s0", "s1", "b0"], positions):
            token = tokens[position] if position >= 0 else {}
            pos = token.get("pos", "NONE")
            tags.append(pos)
            features.append(name + "p=" + pos)
            form = token.get("surface", "")
            if form in self.forms:
                features.append(name + "w=" + form)
            for key, value in sorted(token.get("ud_features", {}).items()):
                features.append(name + "f=" + key + "=" + value)
            children = (
                [tokens[i]["pos"] for i, h in enumerate(heads) if h == position and i > 0]
                if position >= 0
                else []
            )
            features.append(name + "c=" + ",".join(children[:3]))
        features += [
            "pair=" + tags[1] + "|" + tags[0],
            "next=" + tags[0] + "|" + tags[2],
            "triple=" + "|".join(tags),
        ]
        if positions[0] > 0 and positions[1] > 0:
            features.append("distance=" + str(min(5, positions[0] - positions[1])))
        return features

    def legal(self, stack, cursor, size):
        actions = ["S"] if cursor < size else []
        if len(stack) > 1:
            if stack[-2] != 0:
                actions += ["L:" + label for label in self.labels if label != "root"]
                actions += ["R:" + label for label in self.labels if label != "root"]
            elif cursor == size and len(stack) == 2:
                actions.append("R:root")
        return actions

    @staticmethod
    def advance(action, stack, cursor, heads, labels):
        if action == "S":
            stack.append(cursor)
            return cursor + 1
        direction, label = action.split(":", 1)
        child, head = (stack[-2], stack[-1]) if direction == "L" else (stack[-1], stack[-2])
        heads[child] = head
        labels[child] = label
        stack.pop(-2 if direction == "L" else -1)
        return cursor

    def action_scores(self, features, actions):
        scores = dict.fromkeys(actions, 0.0)
        for feature in features:
            for action, value in self.weights.get(feature, {}).items():
                if action in scores:
                    scores[action] += value
        return scores

    def predict_action(self, features, actions):
        scores = self.action_scores(features, actions)
        return max(actions, key=lambda action: scores[action])

    def fit(self, sentences, epochs=3, token_sequences=None):
        sentences = list(sentences)
        if not sentences or type(epochs) is not int or epochs < 1:
            raise ValueError("Training requires sentences and positive epochs")
        if token_sequences is None:
            token_sequences = [
                [
                    {"surface": w.form, "pos": w.pos, "ud_features": dict(w.features)}
                    for w in sentence.words
                ]
                for sentence in sentences
            ]
        else:
            token_sequences = list(token_sequences)
            if len(token_sequences) != len(sentences) or any(
                len(tokens) != len(sentence.words)
                or any(
                    token.get("surface") != word.form
                    or not isinstance(token.get("pos"), str)
                    or not isinstance(token.get("ud_features", {}), dict)
                    for token, word in zip(tokens, sentence.words)
                )
                for tokens, sentence in zip(token_sequences, sentences)
            ):
                raise ValueError("Training evidence must align with dependency words")
        self.labels = sorted(
            {"dep", "root"} | {label for sentence in sentences for label in sentence.labels}
        )
        self.forms = {
            form
            for form, count in Counter(w.form for s in sentences for w in s.words).items()
            if count >= 5
        }
        self.weights = defaultdict(dict)
        totals = defaultdict(float)
        timestamps = defaultdict(int)
        step = 0
        usable = [(s, t) for s, t in zip(sentences, token_sequences) if projective(s)]
        if not usable:
            raise ValueError("No projective training trees available")
        for _ in range(epochs):
            for sentence, evidence in usable:
                tokens = [{"surface": "ROOT", "pos": "ROOT", "ud_features": {}}] + evidence
                stack = [0]
                cursor = 1
                heads = [-1] * len(tokens)
                labels = [""] * len(tokens)
                while cursor < len(tokens) or len(stack) > 1:
                    step += 1
                    action = "S"
                    if len(stack) > 1:
                        left, right = stack[-2:]

                        def complete(node):
                            return all(
                                heads[i] >= 0 for i, h in enumerate(sentence.heads, 1) if h == node
                            )

                        if left and sentence.heads[left - 1] == right and complete(left):
                            action = "L:" + sentence.labels[left - 1]
                        elif sentence.heads[right - 1] == left and complete(right):
                            action = "R:" + sentence.labels[right - 1]
                    features = self.features(tokens, stack, cursor, heads)
                    actions = self.legal(stack, cursor, len(tokens))
                    if action not in actions:
                        raise ValueError("Static oracle cannot reconstruct training tree")
                    guess = self.predict_action(features, actions)
                    if guess != action:
                        for feature in features:
                            for target, delta in [(action, 1), (guess, -1)]:
                                key = feature, target
                                old = self.weights[feature].get(target, 0)
                                totals[key] += (step - timestamps[key]) * old
                                timestamps[key] = step
                                self.weights[feature][target] = old + delta
                    cursor = self.advance(action, stack, cursor, heads, labels)
        for feature, values in self.weights.items():
            for action, value in list(values.items()):
                key = feature, action
                values[action] = (totals[key] + (step - timestamps[key]) * value) / max(1, step)
        return {
            "sentences": len(sentences),
            "projective_training_sentences": len(usable),
            "epochs": epochs,
            "steps": step,
        }

    def parse(self, tokens):
        if not tokens:
            return []
        tokens = [{"surface": "ROOT", "pos": "ROOT", "ud_features": {}}] + tokens
        stack = [0]
        cursor = 1
        heads = [-1] * len(tokens)
        labels = [""] * len(tokens)
        while cursor < len(tokens) or len(stack) > 1:
            actions = self.legal(stack, cursor, len(tokens))
            action = self.predict_action(self.features(tokens, stack, cursor, heads), actions)
            cursor = self.advance(action, stack, cursor, heads, labels)
        return self.arcs(heads, labels)

    @staticmethod
    def arcs(heads, labels):
        return [
            {
                "token_index": i - 1,
                "head_token_index": heads[i] - 1 if heads[i] else None,
                "relation": labels[i],
                "status": "predicted",
                "confidence": None,
            }
            for i in range(1, len(heads))
        ]

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "language": self.language,
                    "source": self.source,
                    "labels": self.labels,
                    "forms": sorted(self.forms),
                    "weights": dict(self.weights),
                }
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["version"] != cls.version:
            raise ValueError("Unsupported dependency model version")
        if (
            "root" not in data["labels"]
            or not any(label != "root" for label in data["labels"])
            or len(data["labels"]) != len(set(data["labels"]))
            or any(
                not isinstance(label, str) or not label or ":" == label for label in data["labels"]
            )
            or not isinstance(data["weights"], dict)
        ):
            raise ValueError("Invalid dependency model labels/weights")
        actions = (
            {"S"}
            | {"L:" + label for label in data["labels"]}
            | {"R:" + label for label in data["labels"]}
        )
        for feature, weights in data["weights"].items():
            if (
                not isinstance(feature, str)
                or not isinstance(weights, dict)
                or any(
                    action not in actions
                    or type(value) not in {int, float}
                    or not math.isfinite(value)
                    for action, value in weights.items()
                )
            ):
                raise ValueError("Invalid dependency model scores")
        model = cls(data["language"], data["source"])
        model.labels = data["labels"]
        model.forms = set(data["forms"])
        model.weights = data["weights"]
        return model


def load_dependency_models(engine, directory):
    directory = Path(directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError("Syntax resource path must be a directory")
    loaded = {}
    for path in sorted(directory.glob("*.json")):
        model = DependencyParser.load(path)
        lexicon = engine._lexicons.get(model.language)
        if (
            model.language in loaded
            or not lexicon
            or lexicon.source["sha256"] != model.source.get("training_sha256")
        ):
            raise ValueError("Dependency resource does not match lexical training source")
        if model.source.get("pos_model_sha256") is not None and model.source[
            "pos_model_sha256"
        ] != engine.tagger.model_fingerprint(model.language):
            raise ValueError("Dependency resource does not match POS model")
        model.source["model_sha256"] = sha256(path.read_bytes()).hexdigest()
        loaded[model.language] = model
    engine.dependency_parsers.update(loaded)
    return sorted(loaded)


def syntax_payload(analysis, models, maximum_tokens=128):
    model = models.get(language_code(analysis.lang))
    result = {
        "status": "experimental_prediction" if model else "unavailable",
        "method": DependencyParser.version if model else None,
        "confidence": None,
        "source": dict(model.source) if model else None,
        "arcs": [],
        "skipped_sentences": [],
        "note": "Experimental projective syntax, uncalibrated; POS/features and sentence boundaries may be wrong. Dependencies do not establish semantic roles or factual meaning.",
    }
    if not model:
        return result
    tokens = [a.to_dict() for a in analysis.analyses]
    ranges = analysis.sentence_ranges or sentence_ranges([t["surface"] for t in tokens])
    for sentence_index, (start, end) in enumerate(ranges):
        if end - start > maximum_tokens:
            result["skipped_sentences"].append(
                {
                    "sentence_index": sentence_index,
                    "token_range": [start, end],
                    "reason": "sentence_token_limit",
                }
            )
        else:
            for arc in model.parse(tokens[start:end]):
                arc["token_index"] += start
                if arc["head_token_index"] is not None:
                    arc["head_token_index"] += start
                arc["span"] = tokens[arc["token_index"]].get("span")
                arc["sentence_index"] = sentence_index
                result["arcs"].append(arc)
    if result["skipped_sentences"]:
        result["status"] = "partial_experimental_prediction"
    return result
