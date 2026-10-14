"""Acquire official CC BY 4.0 labeled audio, retaining publisher checksums."""

import hashlib
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/speech_scale"
BASE = "https://www.openslr.org/resources/12/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        choices=("train-clean-100", "train-clean-360", "test-clean", "dev-other"),
        default="train-clean-100",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = args.archive + ".tar.gz"
    checks = urllib.request.urlopen(BASE + "md5sum.txt", timeout=60).read()
    (OUTPUT / "publisher-md5sum.txt").write_bytes(checks)
    expected = {
        line.split()[-1].lstrip("*"): line.split()[0] for line in checks.decode().splitlines()
    }[name]
    archive = OUTPUT / name
    if not archive.exists():
        temporary = OUTPUT / (name + ".part")
        request = urllib.request.Request(BASE + name, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("Server must support bounded ranges")
            size = int(response.headers["Content-Range"].split("/")[-1])
        pieces = OUTPUT / (
            "download-pieces-32m"
            if args.archive == "train-clean-100"
            else args.archive + "-pieces-32m"
        )
        pieces.mkdir(exist_ok=True)
        chunk_size = 32 * 1024**2
        count = (size + chunk_size - 1) // chunk_size

        def acquire(index):
            start = chunk_size * index
            end = min(size, start + chunk_size) - 1
            piece = pieces / f"{index}.part"
            offset = piece.stat().st_size if piece.exists() else 0
            if offset > end - start + 1:
                raise ValueError("Unexpected cached range size")
            if offset < end - start + 1:
                begin = start + offset
                mirror = BASE if index % 2 == 0 else "https://openslr.elda.org/resources/12/"
                request = urllib.request.Request(
                    mirror + name, headers={"Range": f"bytes={begin}-{end}"}
                )
                with (
                    urllib.request.urlopen(request, timeout=60) as source,
                    piece.open("ab") as target,
                ):
                    if (
                        source.status != 206
                        or source.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}"
                    ):
                        raise ValueError("Unexpected range response")
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            if piece.stat().st_size != end - start + 1:
                raise ValueError("Incomplete downloaded range")
            print(f"Range {index + 1}/{count} acquired", flush=True)
            return piece

        with ThreadPoolExecutor(max_workers=12) as workers:
            parts = list(workers.map(acquire, range(count)))
        with temporary.open("wb") as target:
            for piece in parts:
                with piece.open("rb") as source:
                    shutil.copyfileobj(source, target)
        temporary.replace(archive)
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            md5.update(chunk)
            sha.update(chunk)
    if md5.hexdigest() != expected:
        raise ValueError("Publisher MD5 mismatch")
    print("Publisher checksum verified: " + args.archive, flush=True)
    marker = OUTPUT / (args.archive + ".extracted.sha256")
    if not marker.exists() or marker.read_text() != sha.hexdigest():
        with tarfile.open(archive) as package:
            members = package.getmembers()
            bound = (40 if args.archive == "train-clean-360" else 30) * 1024**3
            if sum(m.size for m in members) > bound:
                raise ValueError("Extraction size exceeds bound")
            for m in members:
                if not m.isfile() and not m.isdir():
                    raise ValueError("Unexpected non-file archive entry")
                if not (OUTPUT / m.name).resolve().is_relative_to(OUTPUT.resolve()):
                    raise ValueError("Unsafe archive path")
            print(f"Extracting {len(members)} validated entries", flush=True)
            package.extractall(OUTPUT, members=members, filter="data")
        marker.write_text(sha.hexdigest())
    (OUTPUT / (args.archive + ".provenance.json")).write_text(
        json.dumps(
            {
                "source": "https://openslr.org/12/",
                "license": "CC BY 4.0",
                "attribution": "Panayotov, Chen, Povey and Khudanpur (2015); LibriSpeech",
                "archive": name,
                "publisher_md5": expected,
                "sha256": sha.hexdigest(),
                "scope": args.archive
                + " English read-speech; not conversational/noisy/multilingual audio",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Verified and extracted " + args.archive, flush=True)


if __name__ == "__main__":
    main()
