"""Training-only acoustic normalization audit; development chooses alternatives."""

from pathlib import Path
import sys, json, copy, time, hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import soundfile as sf
import numpy as np
from collections import Counter
from speech_engine.dataset import load_manifest, build_dataloader
from speech_engine.tokenizer import CharTokenizer, normalize_text
from scripts.speech.prepare_real_speech import validate_disjoint, speaker


def audit_data(folder, tok):
    partitions = {n: load_manifest(folder / (n + ".tsv")) for n in ["train", "dev", "test"]}
    validate_disjoint(partitions)
    trainwords = set(
        " ".join(normalize_text(t, tok.normalization) for _, t in partitions["train"]).split()
    )
    report = {"speaker_disjoint": True, "partitions": {}}
    for name, rows in partitions.items():
        durations = []
        margins = []
        unknown = 0
        changed = 0
        words = []
        rates = Counter()
        for path, text in rows:
            info = sf.info(path)
            rates[info.samplerate] += 1
            durations.append(info.duration)
            normalized = normalize_text(text, tok.normalization)
            ids = tok.encode(text)
            unknown += sum(c not in tok._char_to_id for c in normalized)
            changed += normalized != text.lower().strip()
            words += normalized.split()
            required = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
            frames = int(info.frames / info.samplerate * 100)
            margins.append((frames + 1) // 2 - required)
        pad = sum(
            max(durations[i : i + 8]) * len(durations[i : i + 8]) - sum(durations[i : i + 8])
            for i in range(0, len(durations), 8)
        )
        report["partitions"][name] = {
            "clips": len(rows),
            "speakers": len({speaker(p) for p, t in rows}),
            "hours": sum(durations) / 3600,
            "duration_quantiles": np.quantile(durations, [0, 0.5, 0.9, 1]).tolist(),
            "sample_rates": dict(rates),
            "unrepresented_characters": unknown,
            "normalization_changed_transcripts": changed,
            "minimum_ctc_alignment_slack_frames": min(margins),
            "unalignable": sum(v < 0 for v in margins),
            "training_word_oov_rate": sum(w not in trainwords for w in words) / len(words),
            "sequential_batch8_padding_fraction": pad / (sum(durations) + pad),
        }
    return report


from speech_engine.model_ctc import CTCModel
from speech_engine.evaluate import evaluate_dataset
from speech_engine.decode import greedy_decode


def main():
    torch.set_num_threads(4)
    folder = ROOT / "data/training/asr_expanded"
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    tok = CharTokenizer.load(folder / "tokenizer.json")
    (ROOT / "results/asr_data_audit.json").write_text(
        json.dumps(audit_data(folder, tok), indent=2), encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTCModel(
        tok.vocab_size,
        cnn_channels=tuple(cfg["cnn_channels"]),
        hidden_size=cfg["hidden_size"],
        num_lstm_layers=cfg["num_lstm_layers"],
        time_reduction=cfg.get("time_reduction", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    original = copy.deepcopy(model.state_dict())
    train_rows = load_manifest(folder / "train.tsv")
    dev_rows = load_manifest(folder / "dev.tsv")
    dev = build_dataloader(
        dev_rows,
        tok,
        batch_size=1,
        shuffle=False,
        max_frames=2100,
        cache_dir=folder / "features/dev",
    )
    report = {
        "checkpoint_sha256": hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest(),
        "alternatives": {},
    }
    for batch_size in [0, 1, 8]:
        model.load_state_dict(original)
        model.eval()
        start = time.perf_counter()
        if batch_size:
            # Recompute running moments from training audio only; disable dropout.
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.reset_running_stats()
                    layer.momentum = None
                    layer.train()
            loader = build_dataloader(
                train_rows,
                tok,
                batch_size=batch_size,
                shuffle=False,
                max_frames=2100,
                cache_dir=folder / "features/train",
            )
            with torch.inference_mode():
                for batch in loader:
                    model(batch["mel"].to(device), batch["mel_lengths"].to(device))
        model.eval()
        metrics = evaluate_dataset(model, dev, tok, greedy_decode, device)
        metrics["elapsed_s"] = time.perf_counter() - start
        report["alternatives"][str(batch_size)] = metrics
        print(json.dumps({"normalization_batch": batch_size, **metrics}), flush=True)
    model.load_state_dict(original)
    model.eval()
    rows = train_rows[:80]
    report["training_sample"] = evaluate_dataset(
        model,
        build_dataloader(
            rows,
            tok,
            batch_size=1,
            shuffle=False,
            max_frames=2100,
            cache_dir=folder / "features/train",
        ),
        tok,
        greedy_decode,
        device,
    )
    (ROOT / "results/asr_normalization_diagnosis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
