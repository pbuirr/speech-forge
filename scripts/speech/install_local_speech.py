"""Package our evaluated original CTC weights and training-only decoder for local use."""

import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--development-report", type=Path, required=True)
    p.add_argument("--test-report", type=Path, required=True)
    p.add_argument("--mode", choices=["greedy", "beam", "lm01", "lm02"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    path = a.directory / "best.pt"
    checksum = digest(path)
    reports = {
        n: json.loads(f.read_text(encoding="utf-8"))
        for n, f in [("development", a.development_report), ("test", a.test_report)]
    }
    for report in reports.values():
        if report["checkpoint_sha256"] != checksum or a.mode not in report["metrics"]:
            raise ValueError("Evaluation does not match checkpoint or decoder")
        if report["training_manifest_sha256"] != digest(a.directory / "train.tsv"):
            raise ValueError("Evaluation training data changed")
    chosen = min(
        reports["development"]["metrics"],
        key=lambda mode: reports["development"]["metrics"][mode]["wer"],
    )
    if a.mode != chosen:
        raise ValueError("Decoder must be selected by development WER")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    tokenizer = a.directory / "tokenizer.json"
    config.pop("asr_decoder", None)
    if config.get("tokenizer_sha256") != digest(tokenizer):
        raise ValueError("Tokenizer fingerprint differs")
    if a.mode != "greedy":
        metrics = reports["development"]["metrics"][a.mode]
        lm_manifest = Path(
            reports["development"].get("lm_training_manifest_path", a.directory / "train.tsv")
        )
        expected_lm = reports["development"].get(
            "lm_training_manifest_sha256", digest(a.directory / "train.tsv")
        )
        if (
            digest(lm_manifest) != expected_lm
            or reports["test"].get("lm_training_manifest_sha256", expected_lm) != expected_lm
        ):
            raise ValueError("Decoder evaluation LM data differs")
        lm_order = reports["development"].get("lm_order", 3)
        if lm_order not in (3, 4, 5) or reports["test"].get("lm_order", 3) != lm_order:
            raise ValueError("Decoder evaluation context order differs")
        config["asr_decoder"] = {
            "beam_width": 10,
            "lm_weight": metrics["lm_weight"],
            "token_reward": metrics["token_reward"],
            "training_manifest_path": str(lm_manifest.resolve()),
            "training_manifest_sha256": expected_lm,
            "lm_order": lm_order,
        }
    config["evaluation_summary"] = {
        n: {k: r["metrics"][a.mode][k] for k in ["wer", "cer"]} for n, r in reports.items()
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": checkpoint["model"], "config": config, "source_checkpoint_sha256": checksum},
        a.output,
    )
    a.output.with_suffix(".tokenizer.json").write_bytes(tokenizer.read_bytes())
    metadata = {
        "checkpoint_sha256": digest(a.output),
        "source_checkpoint_sha256": checksum,
        "mode": a.mode,
        "lm_order": config.get("asr_decoder", {}).get("lm_order", 3),
        "evaluation": config["evaluation_summary"],
        "limits": "Experimental English read speech. Aggregate error rates do not certify any transcript. No grammar correction.",
    }
    a.output.with_suffix(".evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
