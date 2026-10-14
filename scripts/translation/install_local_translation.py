"""Install an evaluated NMT checkpoint for the web app, keeping the previous one as a backup."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from speech_engine.nmt import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--allow-repeated-digits", action="store_true")
    args = parser.parse_args()
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model, sv, tv, metadata = load_checkpoint(args.checkpoint)
    evaluation = {}
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Evaluation report belongs to a different checkpoint: " + str(path))
        if (
            bool(report.get("decoding_options", {}).get("allow_repeated_digits", False))
            != args.allow_repeated_digits
        ):
            raise ValueError("Evaluation decoding policy does not match installation: " + str(path))
        options = report.get("decoding_options", {})
        if (
            report.get("decoder") != "beam_4"
            or not options.get("eos_boost", True)
            or not options.get("repeat_guard", True)
            or options.get("no_repeat_ngram", 3) != 3
            or options.get("allow_repeated_fragments", False)
        ):
            raise ValueError(
                "Report uses an experimental decoder not supported by installation: " + str(path)
            )
        evaluation[path.stem] = report["metrics"]
    corpus = Path(metadata["training_corpus"])
    expected = metadata["split_sha256"]["train"]
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != expected:
        raise ValueError(
            "Declared training partition changed; do not install mismatched provenance"
        )
    metadata.update(
        {
            "evaluation": evaluation,
            "evaluated_checkpoint_sha256": checkpoint_hash,
            "quality_status": "Experimental local model trained on everyday sentences and news; review translations. Names, numbers and longer passages remain unreliable.",
            "split_policy": "Saved article/source/target-disjoint news partitions with Tatoeba training-only replay",
            "learning_rate": metadata.get("learning_rate", 8e-4),
            "curriculum_epochs": metadata.get("curriculum_epochs", 1),
        }
    )
    if args.allow_repeated_digits:
        metadata["decoding_policy"] = "preserve_numeric_fragments"
    elif metadata.get("decoding_policy") == "preserve_numeric_fragments":
        metadata.pop("decoding_policy")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    if args.destination.exists():
        previous_hash = hashlib.sha256(args.destination.read_bytes()).hexdigest()
        archive = args.destination.parent / "archive"
        archive.mkdir(exist_ok=True)
        backup = archive / (args.destination.stem + "-" + previous_hash[:16] + ".pt")
        if not backup.exists():
            shutil.copy2(args.destination, backup)
    temp = args.destination.with_suffix(".install.tmp")
    try:
        save_checkpoint(model, sv, tv, temp, meta=metadata)
        load_checkpoint(temp)  # Check serialization before replacing the active file.
        os.replace(temp, args.destination)
    finally:
        if temp.exists():
            temp.unlink()
    print(
        json.dumps(
            {
                "installed": str(args.destination),
                "sha256": hashlib.sha256(args.destination.read_bytes()).hexdigest(),
                "evaluated_weights_sha256": checkpoint_hash,
                "quality_status": metadata["quality_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
