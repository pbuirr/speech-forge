"""Average the weights of several checkpoints of the same model.

Usage:
    python scripts/translation/avg_checkpoints.py \
        data/models/nmt_es_en_ep75.pt data/models/nmt_es_en_ep100.pt \
        data/models/nmt_es_en_ep125.pt --out data/models/nmt_es_en_avg.pt

    python scripts/translation/avg_checkpoints.py --glob "data/models/nmt_es_en_ep*.pt"
"""

from __future__ import annotations

import argparse
import glob as glob_module
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_engine.nmt import load_checkpoint


def average_checkpoints(paths: list[Path], out: Path) -> None:
    if len(paths) < 2:
        print("Need at least 2 checkpoints to average.")
        sys.exit(1)

    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  {p}")

    # Load all state dicts onto CPU
    raw = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

    # Average model weights
    avg_state: dict[str, torch.Tensor] = {}
    for key in raw[0]["model_state"]:
        tensors = [r["model_state"][key].float() for r in raw]
        avg_state[key] = torch.stack(tensors).mean(0).to(raw[0]["model_state"][key].dtype)

    # Use the config and vocabs from the latest checkpoint
    latest = raw[-1]
    latest["model_state"] = avg_state
    latest["meta"]["averaged_from"] = [str(p) for p in paths]
    latest["meta"]["n_averaged"] = len(paths)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latest, out)
    print(f"\nSaved averaged checkpoint → {out}")

    # Quick sanity: load and print param count
    model, src_vocab, tgt_vocab, meta = load_checkpoint(out)
    n = sum(p.numel() for p in model.parameters())
    print(f"  {n:,} parameters, averaged from ep {[r['meta'].get('epoch', '?') for r in raw]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="*", help="Checkpoint paths to average")
    p.add_argument("--glob", default=None, help="Glob pattern instead of explicit paths")
    p.add_argument("--out", required=True, help="Output path for averaged checkpoint")
    p.add_argument(
        "--last", type=int, default=None, help="When using --glob, keep only the last N checkpoints"
    )
    args = p.parse_args()

    if args.glob:
        paths = sorted(glob_module.glob(args.glob))
        if args.last:
            paths = paths[-args.last :]
        paths = [Path(p) for p in paths]
    else:
        paths = [Path(c) for c in args.checkpoints]

    if not paths:
        print("No checkpoints found.")
        sys.exit(1)

    average_checkpoints(paths, Path(args.out))


if __name__ == "__main__":
    main()
