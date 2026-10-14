"""Download a bounded subset of the SLR83 UK and Ireland English dialect corpus, with attribution."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/speech_dialects"
BASE = "https://www.openslr.org/resources/83/"
ARCHIVES = ("irish_english_male.zip", "northern_english_female.zip", "scottish_english_female.zip")


def download(name):
    path = OUTPUT / name
    if not path.exists():
        print("Downloading " + name, flush=True)
        temporary = OUTPUT / (name + ".part")
        with (
            urllib.request.urlopen(BASE + name, timeout=60) as response,
            temporary.open("wb") as target,
        ):
            total = 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                total += len(chunk)
                if total % (32 * 1024 * 1024) == 0:
                    print(f"{name}: {total // (1024 * 1024)} MiB", flush=True)
        temporary.replace(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if name.endswith(".zip"):
        destination = OUTPUT / path.stem
        destination.mkdir(exist_ok=True)
        marker = destination / "archive.sha256"
        if not marker.exists() or marker.read_text() != checksum:
            with zipfile.ZipFile(path) as package:
                if sum(m.file_size for m in package.infolist()) > 3 * 1024**3:
                    raise ValueError("Archive exceeds bounded extraction size")
                for member in package.infolist():
                    final = (destination / member.filename).resolve()
                    if not final.is_relative_to(destination.resolve()) or stat.S_ISLNK(
                        member.external_attr >> 16
                    ):
                        raise ValueError("Unsafe archive member")
                    if member.is_dir():
                        final.mkdir(parents=True, exist_ok=True)
                    else:
                        final.parent.mkdir(parents=True, exist_ok=True)
                        with package.open(member) as source, final.open("wb") as target:
                            shutil.copyfileobj(source, target)
            marker.write_text(checksum)
    print("Acquired " + name, flush=True)
    return name, checksum


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = ("LICENSE", "about.html", "dialect_info.txt", "line_index_all.csv") + ARCHIVES
    with ThreadPoolExecutor(max_workers=2) as workers:
        hashes = dict(workers.map(download, files))
    provenance = {
        "source": "https://openslr.org/83",
        "license": "CC BY-SA 4.0",
        "attribution": "Demirsahin, Kjartansson, Gutkin and Rivera (2020); Google Inc. (2018, 2019)",
        "sha256": hashes,
        "archives": list(ARCHIVES),
        "integrity": "HTTPS origin and ZIP CRC; locally recorded SHA256, no publisher checksum available at the inspected resource",
        "limits": "Bounded three-region subset of elicited English; not spontaneous/conversational or all dialects. Licenses and raw references retained.",
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
