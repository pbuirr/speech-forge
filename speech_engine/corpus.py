"""Corpus and speaker identities for mixed speech training data.

Each corpus declares how to read a speaker ID, so a new corpus is never parsed
with LibriSpeech's filename convention by accident.
"""

import hashlib
import json
from pathlib import Path
import unicodedata
from collections import Counter, defaultdict
from speech_engine.tokenizer import normalize_text


def load_metadata(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for key in (
            "file",
            "text",
            "speaker",
            "corpus",
            "language",
            "license",
            "source",
            "prompt_id",
        ):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError("Missing speech metadata: " + key)
        identity = str(Path(record["file"]).resolve())
        if identity in records:
            raise ValueError("Duplicate audio metadata")
        records[identity] = record
    return records


def record_for(path, metadata):
    # Manifests and metadata already use resolved identities. Avoid repeating
    # filesystem work for those keys; aliases still go through resolution.
    key = str(path)
    if key in metadata:
        return metadata[key]
    try:
        return metadata[str(Path(path).resolve())]
    except KeyError:
        raise ValueError("Audio lacks explicit corpus/speaker metadata: " + str(path)) from None


def validate_partitions(partitions, metadata):
    audio = set()
    speakers = set()
    prompts = set()
    for rows in partitions.values():
        current_audio = set()
        current_speakers = set()
        current_prompts = set()
        for path, text in rows:
            record = record_for(path, metadata)
            if record["text"] != text:
                raise ValueError("Manifest transcript differs from corpus metadata")
            key = str(path)
            current_audio.add(key if key in metadata else str(Path(path).resolve()))
            current_speakers.add(record["speaker"])
            current_prompts.add(record["prompt_id"])
        if len(current_audio) != len(rows):
            raise ValueError("Duplicate audio within speech partition")
        if current_audio & audio or current_speakers & speakers or current_prompts & prompts:
            raise ValueError("Speech partitions overlap audio, speaker identity or prompt identity")
        audio.update(current_audio)
        speakers.update(current_speakers)
        prompts.update(current_prompts)


def lexical_normalization_loss(text, tokenizer):
    """Detect suppressed letters/digits rather than silently deleting meanings.

    Punctuation/spacing changes are allowed; lexical spelling changes are not.
    Digit-to-word rewriting is deliberately avoided without spoken-form evidence.
    """
    raw = unicodedata.normalize("NFC", text.lower())
    normalized = normalize_text(text, tokenizer.normalization)
    lexical = lambda t: "".join(c for c in t if unicodedata.category(c)[0] in "LMN")
    return (
        lexical(raw) != lexical(normalized)
        or tokenizer.decode(tokenizer.encode(text)) != normalized
    )


def metadata_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_weights(examples, metadata):
    """Sampling weights that split mass equally by corpus, then language, dialect and speaker.

    Recordings within a speaker get equal weight. This keeps small groups from
    disappearing in pooled data; it's an experimental choice, not a tuned one.
    """
    keys = []
    languages = defaultdict(set)
    dialects = defaultdict(set)
    speakers = defaultdict(set)
    for path, _ in examples:
        record = record_for(path, metadata)
        corpus, language, dialect, speaker = (
            record["corpus"],
            record["language"],
            record.get("dialect") or "unspecified",
            record["speaker"],
        )
        keys.append((corpus, language, dialect, speaker))
        languages[corpus].add(language)
        dialects[(corpus, language)].add(dialect)
        speakers[(corpus, language, dialect)].add(speaker)
    counts = Counter(keys)
    return [
        1
        / (
            len(languages)
            * len(languages[c])
            * len(dialects[(c, l)])
            * len(speakers[(c, l, d)])
            * counts[(c, l, d, s)]
        )
        for c, l, d, s in keys
    ]
