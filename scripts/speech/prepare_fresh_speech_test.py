"""Freeze a new speaker-balanced sample of LibriSpeech test-clean before evaluating candidates."""

from collections import defaultdict, Counter
import hashlib, json
from pathlib import Path
import sys
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speech_engine.dataset import load_librispeech
from speech_engine.tokenizer import CharTokenizer, normalize_text
from speech_engine.corpus import lexical_normalization_loss


def main():
    output = ROOT / "data/speech_scale/fresh_test"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = CharTokenizer.load(ROOT / "data/models/asr_en_diversity.tokenizer.json")
    grouped = defaultdict(list)
    excluded = Counter()
    for path, text in load_librispeech(ROOT / "data/speech_scale/LibriSpeech", "test-clean"):
        if lexical_normalization_loss(text, tokenizer):
            excluded["lexical_loss"] += 1
            continue
        if not 1 <= sf.info(path).duration <= 18:
            excluded["duration"] += 1
            continue
        grouped[Path(path).stem.split("-")[0]].append(
            (path, normalize_text(text, tokenizer.normalization))
        )
    selected = []
    metadata = []
    for speaker, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda r: hashlib.sha256(
                ("LinguaForge-fresh-142:" + Path(r[0]).stem).encode()
            ).hexdigest()
        )
        selected.extend(rows[:4])
        for path, text in rows[:4]:
            metadata.append(
                {
                    "file": str(Path(path).resolve()),
                    "text": text,
                    "speaker": "librispeech:" + speaker,
                    "language": "en",
                    "corpus": "LibriSpeech-test-clean-fresh",
                    "license": "CC BY 4.0",
                    "source": "https://openslr.org/12/",
                    "prompt_id": "librispeech:" + Path(path).stem,
                }
            )
    manifest = "".join(f"{p}\t{t}\n" for p, t in selected)
    destination = output / "test.tsv"
    if destination.exists() and destination.read_text(encoding="utf-8") != manifest:
        raise ValueError("Frozen test selection changed")
    destination.write_text(manifest, encoding="utf-8")
    (output / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in metadata), encoding="utf-8"
    )
    report = {
        "examples": len(selected),
        "speakers": len(grouped),
        "audio_hours": sum(sf.info(p).duration for p, _ in selected) / 3600,
        "selection": "Four deterministic hash-ranked clips per speaker; duration/lexical eligibility only, no candidate results.",
        "excluded": dict(excluded),
        "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
