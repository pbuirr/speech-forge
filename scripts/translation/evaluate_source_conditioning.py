"""Evaluate source sensitivity on explicit linguistic contrasts, without confidence claims."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from speech_engine.nmt import load_checkpoint
from scripts.translation.prepare_translation_quality import losses
from speech_engine.preprocessing import normalize_es_input, postprocess_en_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=ROOT / "tests/fixtures/source_contrasts.json")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    model, sv, tv, meta = load_checkpoint(a.checkpoint)
    model.eval()
    if meta.get("direction") != "es→en":
        raise ValueError("This supplied contrast set is ES–EN only")
    data = json.loads(a.cases.read_text(encoding="utf-8"))
    cases = data["splits"][a.split]
    paired = losses(
        model, [(r["source"], r["reference"]) for r in cases], sv, tv, torch.device("cpu")
    )
    wrong = losses(
        model, [(r["contrast_source"], r["reference"]) for r in cases], sv, tv, torch.device("cpu")
    )
    rows = []
    for r, x, y in zip(cases, paired, wrong):
        generated = []
        for text in [r["source"], r["contrast_source"]]:
            ids = sv.encode(normalize_es_input(text).normalized, add_bos=True, add_eos=True)
            result = model.translate_with_attention(torch.tensor([ids]))
            generated.append(postprocess_en_output(tv.decode(result["out_ids"])))
        rows.append(
            {
                **r,
                "paired_nll": x,
                "contrast_nll": y,
                "source_margin": y - x,
                "source_translation": generated[0],
                "contrast_translation": generated[1],
            }
        )
    report = {
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(a.cases.read_bytes()).hexdigest(),
        "split": a.split,
        "limits": data["limits"],
        "source_preference_count": sum(r["source_margin"] > 0 for r in rows),
        "examples": len(rows),
        "mean_source_margin": sum(r["source_margin"] for r in rows) / len(rows),
        "rows": rows,
    }
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
