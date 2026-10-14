"""Join official training audio with existing dialect supervision; freeze evaluation."""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speech_engine.dataset import load_librispeech, load_manifest
from speech_engine.corpus import load_metadata, validate_partitions, lexical_normalization_loss
from speech_engine.tokenizer import CharTokenizer, normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100"],
        choices=("train-clean-100", "train-clean-360"),
        help="Official LibriSpeech training splits to include (speaker-disjoint from dev/test by design)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/speech_scale/mixed",
        help="New manifest directory; existing ones are never overwritten implicitly",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "data/speech_dialects/mixed"
    metadata = load_metadata(previous / "metadata.jsonl")
    tok = CharTokenizer.load(ROOT / "data/models/asr_en_diversity.tokenizer.json")
    parts = {k: load_manifest(previous / (k + ".tsv")) for k in ("train", "dev", "test")}
    excluded_speakers = {
        metadata[str(Path(p).resolve())]["speaker"] for k in ("dev", "test") for p, _ in parts[k]
    }
    fresh_metadata = ROOT / "data/speech_scale/fresh_test/metadata.jsonl"
    if fresh_metadata.is_file():
        excluded_speakers.update(
            record["speaker"] for record in load_metadata(fresh_metadata).values()
        )
    # Additional fixed external speaker challenges must also remain unseen.
    for manifest in (ROOT / "data/speech_challenge").glob("*.tsv"):
        for path, _ in load_manifest(manifest):
            excluded_speakers.add("librispeech:" + Path(path).stem.split("-")[0])
    exclusions = Counter()
    accepted = []
    rows = [
        (split, path, raw)
        for split in args.splits
        for path, raw in load_librispeech(ROOT / "data/speech_scale/LibriSpeech", split)
    ]
    for split, path, raw in rows:
        speaker = "librispeech:" + Path(path).stem.split("-")[0]
        if speaker in excluded_speakers:
            exclusions["reserved_speaker"] += 1
            continue
        if lexical_normalization_loss(raw, tok):
            exclusions["lexical_loss"] += 1
            continue
        duration = sf.info(path).duration
        if not 1 <= duration <= 18:
            exclusions["duration"] += 1
            continue
        ids = tok.encode(raw)
        required = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if (int(duration * 100) + 3) // 4 < required:
            exclusions["stride4_alignment"] += 1
            continue
        text = normalize_text(raw, tok.normalization)
        key = str(Path(path).resolve())
        metadata[key] = {
            "file": key,
            "text": text,
            "raw_text": raw,
            "speaker": speaker,
            "corpus": {"train-clean-100": "LibriSpeech100", "train-clean-360": "LibriSpeech360"}[
                split
            ],
            "language": "en",
            "dialect": None,
            "license": "CC BY 4.0",
            "source": "https://openslr.org/12/",
            "prompt_id": "librispeech:" + Path(path).stem,
        }
        accepted.append((key, text))
    # MiniLibriSpeech is a subset of train-clean-100: do not duplicate utterances.
    identities = {Path(p).stem for p, _ in accepted}
    extras = [r for r in parts["train"] if not Path(r[0]).stem in identities]
    parts["train"] = accepted + extras
    validate_partitions(parts, metadata)
    used = {str(Path(p).resolve()) for rows in parts.values() for p, _ in rows}
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as target:
        for key in sorted(used):
            target.write(json.dumps(metadata[key], ensure_ascii=False) + "\n")
    for k, rows in parts.items():
        (output / (k + ".tsv")).write_text(
            "".join(f"{p}\t{t}\n" for p, t in rows), encoding="utf-8"
        )
    report = {
        "counts": {k: len(v) for k, v in parts.items()},
        "hours": {k: sum(sf.info(p).duration for p, _ in v) / 3600 for k, v in parts.items()},
        "speakers": {
            k: len({metadata[str(Path(p).resolve())]["speaker"] for p, _ in v})
            for k, v in parts.items()
        },
        "librispeech_splits": args.splits,
        "exclusions": dict(exclusions),
        "deduplicated_mini_utterances": len(load_manifest(previous / "train.tsv")) - len(extras),
        "evaluation_policy": "Existing dev/test unchanged; no test selection. Reserved external challenge speakers excluded.",
        "metadata_sha256": hashlib.sha256((output / "metadata.jsonl").read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
