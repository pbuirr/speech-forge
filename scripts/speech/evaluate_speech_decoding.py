"""Compare greedy, beam and LM-fused CTC decoding on cached model outputs."""

from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import soundfile as sf
import numpy as np
from speech_engine.audio import preprocess
from speech_engine.features import log_mel_spectrogram
from speech_engine.dataset import load_manifest, build_dataloader
from speech_engine.tokenizer import CharTokenizer, normalize_text
from speech_engine.model_ctc import CTCModel
from speech_engine.train import load_checkpoint
from speech_engine.decode import greedy_decode, beam_decode
from speech_engine.evaluate import char_error_rate, word_error_rate, edit_counts
from speech_engine.lm import NgramLM
from scripts.speech.prepare_real_speech import speaker
from speech_engine.corpus import load_metadata, record_for, metadata_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--noise-snr",
        type=float,
        help="Evaluation-only synthetic white noise SNR in dB; not real recording diversity",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        action="append",
        help="Explicit corpus identities for external audio or LM data; repeat for multiple corpora",
    )
    p.add_argument(
        "--lm-training-manifest",
        type=Path,
        help="Training-only transcript ablation; acoustic weights unchanged",
    )
    p.add_argument(
        "--lm-order",
        type=int,
        choices=(3, 4, 5),
        default=3,
        help="Character backoff order; select on development only",
    )
    p.add_argument(
        "--mode",
        choices=["compare", "greedy", "beam", "lm01", "lm02", "greedy_lm02"],
        default="compare",
    )
    a = p.parse_args()
    if a.limit < 1:
        p.error("--limit must be positive")
    if a.noise_snr is not None and not -5 <= a.noise_snr <= 40:
        p.error("Noise SNR must be within -5..40 dB")
    torch.set_num_threads(4)
    tok = CharTokenizer.load(a.directory / "tokenizer.json")
    path = a.directory / "best.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = CTCModel(
        tok.vocab_size,
        cnn_channels=tuple(cfg["cnn_channels"]),
        hidden_size=cfg["hidden_size"],
        num_lstm_layers=cfg["num_lstm_layers"],
        time_reduction=cfg.get("time_reduction", 2),
    )
    load_checkpoint(path, model)
    model.eval()
    metadata = None
    if cfg.get("dataset_metadata_path"):
        if metadata_digest(cfg["dataset_metadata_path"]) != cfg["dataset_metadata_sha256"]:
            raise ValueError("Corpus metadata fingerprint changed")
        metadata = load_metadata(cfg["dataset_metadata_path"])
    metadata_sources = []
    for metadata_path in a.metadata or []:
        external = load_metadata(metadata_path)
        if metadata and any(k in metadata and metadata[k] != v for k, v in external.items()):
            raise ValueError("Conflicting corpus metadata")
        metadata = {**(metadata or {}), **external}
        metadata_sources.append(
            {"path": str(metadata_path.resolve()), "sha256": metadata_digest(metadata_path)}
        )
    training = load_manifest(a.directory / "train.tsv")
    lm = NgramLM(min_count=1, order=a.lm_order)
    lm_manifest = a.lm_training_manifest or a.directory / "train.tsv"
    lm_training = load_manifest(lm_manifest)
    lm_started = time.perf_counter()
    lm.train([list(normalize_text(text, tok.normalization)) for _, text in lm_training])
    lm.finalize()
    lm_build_elapsed = time.perf_counter() - lm_started
    manifest = a.manifest or a.directory / (a.split + ".tsv")
    rows = load_manifest(manifest)[: a.limit]
    if not rows:
        raise ValueError("No evaluation recordings")
    if metadata and any(
        str(Path(p).resolve()) in metadata and record_for(p, metadata)["text"] != t for p, t in rows
    ):
        raise ValueError("Evaluation transcript differs from corpus metadata")
    train_paths = {str(Path(path).resolve()) for path, text in training + lm_training}
    if any(str(Path(path).resolve()) in train_paths for path, text in rows):
        raise ValueError("Evaluation audio overlaps training")
    # This evaluator targets LibriSpeech manifests: IDs encode speaker identity.
    identity = lambda path: (
        record_for(path, metadata)["speaker"] if metadata else "librispeech:" + speaker(path)
    )

    # External LibriSpeech challenges need not appear in training metadata.
    def eval_identity(path):
        if metadata and str(Path(path).resolve()) in metadata:
            return identity(path)
        if not re.fullmatch(r"\d+-\d+-\d+", Path(path).stem):
            raise ValueError("External corpus requires explicit speaker metadata")
        return "librispeech:" + speaker(path)

    train_speakers = {identity(path) for path, _ in training + lm_training}
    if train_speakers & {eval_identity(path) for path, _ in rows}:
        raise ValueError("Evaluation speakers overlap training")
    loader = build_dataloader(
        rows,
        tok,
        batch_size=1,
        shuffle=False,
        max_frames=3000,
        cache_dir=a.directory / "decoding_features" / a.split,
    )
    cached = []
    refs = []
    acoustic_started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if a.noise_snr is not None:
                wave, _ = preprocess(rows[index][0], target_sr=16000)
                noise = (
                    np.random.default_rng(127 + index)
                    .standard_normal(wave.shape)
                    .astype(np.float32)
                )
                rms = float(np.sqrt(np.mean(wave**2)))
                noise *= rms / (10 ** (a.noise_snr / 20) * float(np.sqrt(np.mean(noise**2))))
                batch["mel"] = torch.from_numpy(log_mel_spectrogram(wave + noise))[None]
                batch["mel_lengths"] = torch.tensor([batch["mel"].shape[2]])
            cached.append(model(batch["mel"], batch["mel_lengths"]))
            refs.append(tok.decode(batch["tokens"][0, : int(batch["token_lengths"][0])].tolist()))
    acoustic_elapsed = time.perf_counter() - acoustic_started
    audio_seconds = sum(sf.info(path).duration for path, _ in rows)
    variants = {"greedy": (0, 0), "beam": (0, 0), "lm01": (0.1, 0.1), "lm02": (0.2, 0.2)}
    metrics = {}
    for mode, (weight, reward) in variants.items():
        if a.mode == "greedy_lm02" and mode not in ("greedy", "lm02"):
            continue
        if a.mode not in ("compare", "greedy_lm02") and mode != a.mode:
            continue
        started = time.perf_counter()
        hyps = []
        for logits in cached:
            hyps.extend(
                greedy_decode(logits, tok)
                if mode == "greedy"
                else beam_decode(
                    logits, tok, beam_width=10, lm=lm, lm_weight=weight, token_reward=reward
                )
            )
        metrics[mode] = {
            "wer": word_error_rate(refs, hyps),
            "cer": char_error_rate(refs, hyps),
            "decode_elapsed_s": time.perf_counter() - started,
            "lm_weight": weight,
            "token_reward": reward,
            "rows": [
                {
                    "file": path,
                    "reference": r,
                    "hypothesis": h,
                    "speaker": eval_identity(path),
                    "corpus": metadata[str(Path(path).resolve())]["corpus"]
                    if metadata and str(Path(path).resolve()) in metadata
                    else "LibriSpeech",
                    "dialect": metadata[str(Path(path).resolve())].get("dialect")
                    if metadata and str(Path(path).resolve()) in metadata
                    else None,
                }
                for (path, _), r, h in zip(rows, refs, hyps)
            ],
        }
        alignments = [edit_counts(r.split(), h.split()) for r, h in zip(refs, hyps)]
        metrics[mode]["word_edit_counts"] = {
            k: sum(a[k] for a in alignments) for k in alignments[0]
        }
        chars = [edit_counts(list(r), list(h)) for r, h in zip(refs, hyps)]
        metrics[mode]["character_edit_counts"] = {k: sum(c[k] for c in chars) for k in chars[0]}
        metrics[mode]["pipeline_real_time_factor"] = (
            acoustic_elapsed + metrics[mode]["decode_elapsed_s"]
        ) / audio_seconds
        for field in ("corpus", "dialect"):
            grouped = defaultdict(list)
            for row in metrics[mode]["rows"]:
                if row.get(field):
                    grouped[row[field]].append(row)
            metrics[mode]["by_" + field] = {
                key: {
                    "examples": len(group),
                    "wer": word_error_rate(
                        [r["reference"] for r in group], [r["hypothesis"] for r in group]
                    ),
                    "cer": char_error_rate(
                        [r["reference"] for r in group], [r["hypothesis"] for r in group]
                    ),
                }
                for key, group in grouped.items()
            }
        print(json.dumps({k: v for k, v in metrics[mode].items() if k != "rows"}), flush=True)
    report = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_manifest_sha256": hashlib.sha256(
            (a.directory / "train.tsv").read_bytes()
        ).hexdigest(),
        "lm_training_manifest_path": str(lm_manifest.resolve()),
        "lm_training_manifest_sha256": hashlib.sha256(lm_manifest.read_bytes()).hexdigest(),
        "examples": len(rows),
        "split": a.split,
        "metrics": metrics,
        "lm_order": a.lm_order,
        "lm_build_elapsed_s": lm_build_elapsed,
        "acoustic_and_feature_elapsed_s": acoustic_elapsed,
        "speaker_disjoint_from_training": True,
        "evaluation_metadata_sha256": metadata_sources[0]["sha256"]
        if len(metadata_sources) == 1
        else cfg.get("dataset_metadata_sha256"),
        "evaluation_metadata_sources": metadata_sources,
        "audio_seconds": audio_seconds,
        "device": "cpu",
        "threads": 4,
        "parameters": model.count_parameters(),
        "synthetic_noise": None
        if a.noise_snr is None
        else {
            "snr_db": a.noise_snr,
            "type": "Gaussian white noise",
            "seed": 127,
            "limits": "Controlled perturbation, not genuine environmental recordings",
        },
        "limits": "CTC probabilities + optional training-only character backoff scores. No dictionary correction or pretrained weights. Decoding time excludes cached acoustic inference. Select settings on development only; limited English read-speech.",
    }
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
