"""Compare two models' saved translations on how well they preserve numbers."""

from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path


def numeric_mentions(rows):
    common = preserved = sentences = 0
    for row in rows:
        source = Counter(re.findall(r"\d+", row["source"]))
        reference = Counter(re.findall(r"\d+", row["reference"]))
        anchors = source & reference
        output = Counter(re.findall(r"\d+", row["application"]))
        common += sum(anchors.values())
        preserved += sum((anchors & output).values())
        sentences += bool(anchors)
    return {
        "anchor_occurrences": common,
        "preserved_occurrences": preserved,
        "sentences_with_common_anchors": sentences,
        "strict_mention_recall": preserved / common if common else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old = json.loads(a.previous.read_text(encoding="utf-8"))
    new = json.loads(a.selected.read_text(encoding="utf-8"))
    if [(r["source"], r["reference"]) for r in old["rows"]] != [
        (r["source"], r["reference"]) for r in new["rows"]
    ]:
        raise ValueError("Comparison requires the same ordered source/reference pairs")
    report = {
        "limits": "Literal source/reference digit-group intersection, occurrence-sensitive. Not numerical equivalence, hallucination precision, named entity accuracy, role/scope or factual correctness. Formatting/paraphrase can change counts.",
        "previous": {
            "report_sha256": hashlib.sha256(a.previous.read_bytes()).hexdigest(),
            **numeric_mentions(old["rows"]),
        },
        "selected": {
            "report_sha256": hashlib.sha256(a.selected.read_bytes()).hexdigest(),
            **numeric_mentions(new["rows"]),
        },
    }
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
