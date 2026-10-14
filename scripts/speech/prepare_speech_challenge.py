"""Fixed evaluation-only sample of licensed LibriSpeech test-other speakers."""

import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import soundfile as sf
from speech_engine.dataset import load_librispeech, load_manifest
from scripts.speech.prepare_real_speech import speaker


def main():
    output = ROOT / "data/speech_challenge"
    output.mkdir(parents=True, exist_ok=True)
    base = "https://www.openslr.org/resources/12/"
    filename = "test-other.tar.gz"
    checks = urllib.request.urlopen(base + "md5sum.txt", timeout=30).read().decode()
    expected = {line.split()[-1].lstrip("*"): line.split()[0] for line in checks.splitlines()}
    archive = output / filename
    if not archive.exists():
        print("Downloading licensed test-other archive", flush=True)
        temporary = output / (filename + ".part")
        with (
            urllib.request.urlopen(base + filename, timeout=60) as response,
            temporary.open("wb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != expected[filename]:
        raise ValueError("Official archive checksum mismatch")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    marker = output / "extracted.sha256"
    if not marker.exists() or marker.read_text() != digest:
        with tarfile.open(archive) as package:
            package.extractall(output, filter="data")
        marker.write_text(digest)
    known = set()
    for split in ("train", "dev", "test"):
        known.update(
            speaker(path) for path, _ in load_manifest(ROOT / "data/speech_real" / (split + ".tsv"))
        )
    pool = load_librispeech(output / "LibriSpeech", "test-other")
    groups = {}
    for path, text in pool:
        if speaker(path) not in known and 1 <= sf.info(path).duration <= 29:
            groups.setdefault(speaker(path), []).append((path, text))
    # Equal speaker contributions, fixed before candidate evaluation; no transcript
    # or quality-based cherry-picking. Keep this challenge out of training/tuning.
    rows = []
    for key in sorted(groups):
        choices = sorted(groups[key])
        random.Random(127).shuffle(choices)
        rows.extend(choices[:4])
    if not rows:
        raise ValueError("No genuinely new speakers")
    manifest = output / "test.tsv"
    manifest.write_bytes(
        "".join(f"{Path(path).resolve()}\t{text}\n" for path, text in rows).encode("utf-8")
    )
    provenance = {
        "source": "https://openslr.org/12",
        "license": "CC BY 4.0",
        "attribution": "LibriSpeech by Panayotov et al., derived from LibriVox audiobooks",
        "archive_sha256": digest,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "speakers": sorted(groups),
        "excluded_prior_speakers": sorted(known),
        "examples": len(rows),
        "audio_seconds": sum(sf.info(path).duration for path, _ in rows),
        "selection": "seed127; four duration-eligible utterances per new speaker; evaluation only",
        "limits": "Different speakers and harder official other subset, still English audiobooks; not conversational or multilingual speech.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
