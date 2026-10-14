"""Translate a fixed challenge set and check that names, numbers and key words survive."""

from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from speech_engine.nmt import load_checkpoint
from speech_engine.preprocessing import normalize_es_input, postprocess_en_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=ROOT / "tests/fixtures/translation_facts.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-repeated-digits", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    model, sv, tv, _ = load_checkpoint(a.checkpoint)
    model.eval()
    data = json.loads(a.cases.read_text(encoding="utf-8"))
    rows = []
    for case in data["cases"]:
        text = normalize_es_input(case["source"]).normalized
        result = model.translate_with_attention(
            torch.tensor([sv.encode(text, add_bos=True, add_eos=True)]),
            repeat_exempt_ids={
                i for i, t in tv._id_to_token.items() if t.removesuffix("</w>").isdigit()
            }
            if a.allow_repeated_digits
            else None,
        )
        output = postprocess_en_output(tv.decode(result["out_ids"]))
        # Expand common contracted negation; only a mention check, not scope or adequacy.
        tokens = set(re.findall(r"\w+", re.sub(r"n['’]t\b", " not", output.casefold())))
        missing = [x for x in case["required_lexical_anchors"] if x not in tokens]
        rows.append({**case, "translation": output, "missing_lexical_anchors": missing})
    report = {
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "allow_repeated_digits": a.allow_repeated_digits
        or bool(getattr(model, "_repeat_exempt_ids", set())),
        "cases_sha256": hashlib.sha256(a.cases.read_bytes()).hexdigest(),
        "limits": data["provenance"],
        "cases": len(rows),
        "all_required_anchors_present": sum(not r["missing_lexical_anchors"] for r in rows),
        "rows": rows,
    }
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
