"""Rescore stored predictions after display-only corrections, without new inference."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.translation.evaluate_local_translation import character_f2
from speech_engine.bleu import bleu_detail
from speech_engine.preprocessing import postprocess_en_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["rows"]
        for row in rows:
            row["application"] = postprocess_en_output(row["raw"])
        references = [row["reference"] for row in rows]
        hypotheses = [row["application"] for row in rows]
        data["metrics"]["application"] = {
            **bleu_detail(hypotheses, references),
            "character_f2_6_no_space": character_f2(hypotheses, references),
            "exact_match_lowercase": sum(
                h.lower() == r.lower() for h, r in zip(hypotheses, references)
            )
            / len(rows),
        }
        data["postprocessing_revision"] = (
            "preserve lexical content; source anchors remain separate warnings"
        )
        data["rescore_limits"] = (
            "Inference timings and review flags are historical; no new inference or flag calibration."
        )
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path.name, data["metrics"]["application"]["bleu"])


if __name__ == "__main__":
    main()
