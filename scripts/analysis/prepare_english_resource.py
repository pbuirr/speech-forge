"""Reproduce the pinned UD English EWT resource and held-out evaluation.

Downloads public data, trains original count-based models, and runs locally.
No pretrained weights or external inference services are used.
"""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from speech_engine.linguistic_resources import CorpusLexicon
from speech_engine.linguistic_evaluation import evaluate_lexicon, evaluate_context
import json

COMMIT = "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b"
SOURCE = f"https://github.com/UniversalDependencies/UD_English-EWT/tree/{COMMIT}"
RAW = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{COMMIT}/"
CHECKSUMS = {
    "en_ewt-ud-train.conllu": "d68e06122a702464c613076523d56740f047e5bbe89dd90ec32737e04d952143",
    "en_ewt-ud-test.conllu": "fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a",
    "en_ewt-ud-dev.conllu": "39239e0a60db3ae68f4b7036189f11b6692741d10ff8240dd91f74f2760d90f8",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", type=Path, default=Path("data/corpora/ud-ewt"))
    parser.add_argument("--output", type=Path, default=Path("data/linguistics/en.json"))
    parser.add_argument("--report", type=Path, default=Path("results/linguistic_eval_ewt.json"))
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Also fit local POS reliability on development data and evaluate on test",
    )
    args = parser.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "README.md",
        "LICENSE.txt",
        *(name for name in CHECKSUMS if args.calibrate or "-dev." not in name),
    ]:
        path = args.download_dir / name
        if not path.exists():
            with urlopen(RAW + name, timeout=60) as response:
                raw = response.read()
            if name in CHECKSUMS and sha256(raw).hexdigest() != CHECKSUMS[name]:
                raise ValueError(f"Checksum mismatch for {name}")
            path.write_bytes(raw)
        if name in CHECKSUMS and sha256(path.read_bytes()).hexdigest() != CHECKSUMS[name]:
            raise ValueError(f"Cached {name} does not match the pinned dataset")
    lexicon = CorpusLexicon.from_conllu(
        args.download_dir / "en_ewt-ud-train.conllu",
        "en",
        source_url=SOURCE,
        license="CC-BY-SA-4.0",
    )
    lexicon.save(args.output)
    for original, saved in [
        ("README.md", "UD_English-EWT.README.md"),
        ("LICENSE.txt", "UD_English-EWT.LICENSE.txt"),
    ]:
        (args.output.parent / saved).write_bytes((args.download_dir / original).read_bytes())
    report = evaluate_lexicon(lexicon, args.download_dir / "en_ewt-ud-test.conllu")
    report["contextual_comparison"] = evaluate_context(
        lexicon, args.download_dir / "en_ewt-ud-test.conllu"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "resource": str(args.output),
                "forms": lexicon.summary["forms"],
                "report": str(args.report),
                **report["contextual_comparison"],
            },
            indent=2,
        )
    )
    if args.calibrate:
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("calibrate_pos.py")),
                "--resource",
                str(args.output),
                "--training-corpus",
                str(args.download_dir / "en_ewt-ud-train.conllu"),
                "--development",
                str(args.download_dir / "en_ewt-ud-dev.conllu"),
                "--test",
                str(args.download_dir / "en_ewt-ud-test.conllu"),
                "--output",
                str(args.output.parent / "calibration" / "en.json"),
                "--report",
                str(args.report.parent / "pos_calibration_ewt.json"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
