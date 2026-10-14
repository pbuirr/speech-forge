"""Train the NMT model on real bitext with reproducible train/dev/test splits that share no source or target sentence."""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from speech_engine.bpe import BPEVocab
from speech_engine.nmt import Seq2SeqNMT, collate_pairs, save_checkpoint, load_checkpoint
from scripts.translation.train_nmt import load_corpus, train


def canonical(text):
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold() if c.isalnum())


def split_pairs(pairs, seed=731):
    """Keep exact/punctuation-normalized sources AND targets in the same partition."""
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for source, target in pairs:
        a, b = find(("s", canonical(source))), find(("t", canonical(target)))
        parents[a] = b
    groups = {}
    for pair in pairs:
        groups.setdefault(find(("s", canonical(pair[0]))), []).append(pair)
    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)
    splits = {"train": [], "dev": [], "test": []}
    goal = max(1, round(len(pairs) * 0.05))
    for group in grouped:
        key = (
            "test"
            if len(splits["test"]) < goal
            else "dev"
            if len(splits["dev"]) < goal
            else "train"
        )
        splits[key].extend(group)
    if not all(splits.values()):
        raise ValueError(
            "Corpus lacks enough independent source/target groups for three partitions"
        )
    return splits


@torch.inference_mode()
def validation_loss(model, encoded, device):
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(encoded), 64):
        src, _, tgt_in, tgt_out, _ = collate_pairs(encoded[start : start + 64], pad_id=0)
        logits = model(src.to(device), tgt_in.to(device))
        target = tgt_out.to(device)
        total += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=0, reduction="sum"
        ).item()
        tokens += target.ne(0).sum().item()
    return total / max(1, tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--src", default="es")
    parser.add_argument("--tgt", default="en")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--source-contrast-weight", type=float, default=0)
    parser.add_argument(
        "--curriculum-epochs",
        type=int,
        default=None,
        help="Default first 20%% of epochs; set 0 for controlled shuffled comparison",
    )
    parser.add_argument(
        "--dataset", default="User-provided parallel corpus; verify provenance and license"
    )
    parser.add_argument(
        "--load",
        type=Path,
        help="Warm-start existing weights and matching BPE vocabularies; new optimizer",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        help="Reuse saved train/dev/test partitions instead of repartitioning",
    )
    args = parser.parse_args()
    if args.accum_steps < 1 or args.accum_steps > args.batch:
        raise ValueError("Accumulation steps must be within 1..batch")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = list(
        dict.fromkeys(
            pair
            for pair in load_corpus(args.corpus)
            if all(
                2 <= len(side.split()) <= args.max_words and len(side) <= args.max_chars
                for side in pair
            )
        )
    )
    splits = (
        {name: load_corpus(args.splits_dir / f"{name}.tsv") for name in ["train", "dev", "test"]}
        if args.splits_dir
        else split_pairs(pairs, args.seed)
    )
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        for side in (0, 1):
            if {canonical(p[side]) for p in splits[left]} & {
                canonical(p[side]) for p in splits[right]
            }:
                raise ValueError("Saved partitions overlap in source or target text")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (args.output_dir / f"{name}.tsv").write_text(
            "".join(f"{s}\t{t}\n" for s, t in rows), encoding="utf-8"
        )
    metadata = {
        "direction": f"{args.src}→{args.tgt}",
        "seed": args.seed,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "split_policy": "saved partitions; validated source/target disjointness"
        if args.splits_dir
        else "source/target normalized connected groups; 90/5/5",
        "dataset": args.dataset,
        "quality_status": "experimental; development-loss selection; single-reference evaluation pending",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    metadata["learning_rate"] = args.lr
    metadata["source_contrast_weight"] = args.source_contrast_weight
    metadata["accum_steps"] = args.accum_steps
    curriculum_epochs = (
        args.curriculum_epochs if args.curriculum_epochs is not None else max(1, args.epochs // 5)
    )
    metadata["curriculum_epochs"] = curriculum_epochs
    metadata["split_sha256"] = {
        name: hashlib.sha256((args.output_dir / f"{name}.tsv").read_bytes()).hexdigest()
        for name in splits
    }
    metadata["warm_start"] = str(args.load) if args.load else None
    metadata["training_corpus"] = str((args.output_dir / "train.tsv").resolve())
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    start = time.perf_counter()
    if args.load:
        model, sv, tv, previous = load_checkpoint(args.load)
        if previous.get("decoding_policy"):
            metadata["decoding_policy"] = previous["decoding_policy"]
        if previous.get("direction") != metadata["direction"]:
            raise ValueError("Warm-start checkpoint direction does not match training")
    else:
        print("Learning BPE from training partition only…", flush=True)
        sv = BPEVocab().train([s for s, t in splits["train"]], args.vocab)
        tv = BPEVocab().train([t for s, t in splits["train"]], args.vocab)
        model = Seq2SeqNMT(
            sv.vocab_size,
            tv.vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            dropout=0.1,
            max_src_len=256,
            max_tgt_len=256,
        )

    def encode(rows):
        return [
            (sv.encode(s, add_bos=True, add_eos=True), tv.encode(t, add_bos=True, add_eos=True))
            for s, t in rows
        ]

    encoded = encode(splits["train"])
    dev = encode(splits["dev"])
    model = model.to(device)
    metadata["parameters"] = model.param_count()
    history = []
    best = float("inf")

    def checkpoint(epoch, model, loss, optimizer, scheduler, scaler):
        nonlocal best
        dev_loss = validation_loss(model, dev, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss_smoothed": loss,
                "dev_token_nll": dev_loss,
                "elapsed_s": time.perf_counter() - start,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if dev_loss < best:
            best = dev_loss
            save_checkpoint(
                model,
                sv,
                tv,
                args.output_dir / "best.pt",
                meta={**metadata, "epoch": epoch, "dev_token_nll": dev_loss},
            )
        save_checkpoint(
            model, sv, tv, args.output_dir / "latest.pt", meta={**metadata, "epoch": epoch}
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "random_state": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            args.output_dir / "training_state.pt",
        )
        (args.output_dir / "training.json").write_text(
            json.dumps({"metadata": metadata, "history": history}, indent=2), encoding="utf-8"
        )

    train(
        model,
        encoded,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        src_vocab=sv,
        tgt_vocab=tv,
        epoch_callback=checkpoint,
        curriculum_epochs=curriculum_epochs,
        source_contrast_weight=args.source_contrast_weight,
        accum_steps=args.accum_steps,
    )


if __name__ == "__main__":
    main()
