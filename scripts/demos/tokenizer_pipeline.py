"""Demo: character tokenizer, dataset and batching.

Builds a toy dataset (synthetic audio, no download) unless a corpus is given.

Usage:
    python scripts/demos/tokenizer_pipeline.py
    python scripts/demos/tokenizer_pipeline.py --librispeech data/LibriSpeech --split test-clean
    python scripts/demos/tokenizer_pipeline.py --manifest path/to/manifest.tsv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.audio import save_audio, TARGET_SR
from speech_engine.dataset import (
    ASRDataset,
    build_dataloader,
    load_librispeech,
    load_manifest,
)
from speech_engine.features import HOP_LENGTH, SAMPLE_RATE
from speech_engine.tokenizer import CharTokenizer, normalize_text, BLANK_ID

# ANSI colours
_G = "\033[92m"  # green
_B = "\033[94m"  # blue
_Y = "\033[93m"  # yellow
_C = "\033[96m"  # cyan
_D = "\033[2m"  # dim
_W = "\033[1m"  # bold
_R = "\033[0m"  # reset


def _sec(title: str) -> None:
    print(f"\n{_C}── {title} {'─' * max(0, 54 - len(title))}{_R}")


def _ok(label: str, value: str) -> None:
    print(f"  {_B}{label:<22}{_R} {value}")


# Toy dataset

TOY_SENTENCES = [
    "hello world",
    "the cat sat on the mat",
    "speech recognition is hard but fun",
    "we are building this from scratch",
    "each frame is ten milliseconds of audio",
    "the mel filterbank maps linear frequency to perceptual scale",
    "good morning how are you today",
    "language learning requires consistent daily practice",
    "i want to drink water",
    "the quick brown fox jumps over the lazy dog",
]


def make_toy_dataset(out_dir: Path, n: int = 10) -> list[tuple[str, str]]:
    """Generate synthetic audio files paired with transcripts.

    Each audio file is a unique combination of sine waves (different formant
    frequencies and durations) so the spectrograms look different from each
    other. The transcript is a realistic English sentence.

    No downloads needed.
    The audio won't match the transcript — this is synthetic — but every
    shape, padding, and collation operation will work identically to real data.

    Args:
        out_dir: directory to write audio files and the manifest TSV
        n:       number of examples to generate (max 10)

    Returns:
        list of (audio_path, transcript) pairs
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(n, len(TOY_SENTENCES))
    examples: list[tuple[str, str]] = []
    manifest_lines: list[str] = ["# LinguaForge toy dataset — auto-generated\n"]

    for i in range(n):
        sentence = TOY_SENTENCES[i]
        # Vary duration (1.5 – 4.5s) and formants so spectrograms differ
        duration = 1.5 + i * 0.35
        f0 = 120 + i * 20  # fundamental: 120–300 Hz
        f1 = 600 + i * 50  # first formant: 600–1050 Hz
        f2 = 1800 + i * 100  # second formant: 1800–2700 Hz
        sr = TARGET_SR
        t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)

        signal = (
            0.50 * np.sin(2 * np.pi * f0 * t)
            + 0.30 * np.sin(2 * np.pi * f1 * t)
            + 0.15 * np.sin(2 * np.pi * f2 * t)
        )
        # Syllable-rate modulation — different rate per sentence
        syllable_rate = 3.0 + i * 0.3
        signal *= (1.0 + np.sin(2 * np.pi * syllable_rate * t)) / 2.0
        signal = (signal / (np.abs(signal).max() + 1e-9) * 0.90).astype(np.float32)

        audio_path = out_dir / f"toy_{i:02d}.wav"
        save_audio(signal, audio_path, sample_rate=sr)

        examples.append((str(audio_path), sentence))
        manifest_lines.append(f"{audio_path}\t{sentence}\n")

    manifest_path = out_dir / "manifest.tsv"
    manifest_path.write_text("".join(manifest_lines), encoding="utf-8")
    return examples


# Section 1: Tokenizer


def demo_tokenizer(transcripts: list[str]) -> CharTokenizer:
    _sec("Tokenizer")

    tok = CharTokenizer()
    _ok("Base vocab size:", str(tok.vocab_size))
    _ok("Representation:", repr(tok))

    # Show the full character mapping
    char_ids = {
        k: v
        for k, v in sorted(tok._char_to_id.items(), key=lambda x: x[1])
        if not k.startswith("<")
    }
    print(f"\n  Character → ID mapping (first 15 shown):")
    items = list(char_ids.items())
    for i in range(0, min(15, len(items)), 5):
        row = "  " + "   ".join(f"'{k}'→{v}" for k, v in items[i : i + 5])
        print(f"  {_D}{row}{_R}")
    if len(items) > 15:
        print(f"  {_D}  ... ({len(items)} chars total){_R}")

    # Fit on the transcript corpus (adds any missing chars)
    print(f"\n  Fitting on {len(transcripts)} transcripts...")
    tok.fit(transcripts)
    _ok("Vocab after fit:", str(tok.vocab_size))

    # Encode / decode round-trip
    sample = "hello world"
    ids = tok.encode(sample)
    reconstructed = tok.decode(ids)
    print(f"\n  Encode/decode round-trip:")
    _ok("  Input text:", repr(sample))
    _ok("  Encoded IDs:", str(ids))
    _ok("  Decoded text:", repr(reconstructed))
    _ok("  Round-trip ok:", str(reconstructed == sample))

    # BOS/EOS for the seq2seq decoder
    ids_with_eos = tok.encode_with_eos(sample)
    _ok("  With BOS/EOS:", str(ids_with_eos))

    # CTC collapse (greedy decoding simulation)
    print(f"\n  CTC collapse simulation:")
    print(f"  {'─' * 50}")
    # Simulate what CTC output looks like before collapse
    raw_ctc = [
        tok._char_to_id.get("h", 0),
        tok._char_to_id.get("h", 0),  # h h
        BLANK_ID,
        tok._char_to_id.get("e", 0),  # e
        tok._char_to_id.get("l", 0),
        tok._char_to_id.get("l", 0),  # l l
        BLANK_ID,
        tok._char_to_id.get("l", 0),  # l
        tok._char_to_id.get("o", 0),
        tok._char_to_id.get("o", 0),  # o o
    ]
    collapsed = tok.ctc_collapse(raw_ctc)
    raw_chars = [tok._id_to_char.get(i, f"<{i}>") for i in raw_ctc]
    print(f"  Raw CTC output:  {raw_chars}")
    print(f"  After collapse:  {repr(collapsed)}")

    # Save and reload
    tok_path = PROJECT_ROOT / "data" / "tokenizer.json"
    tok.save(tok_path)
    reloaded = CharTokenizer.load(tok_path)
    _ok("\n  Save/load ok:", str(reloaded.vocab_size == tok.vocab_size))
    _ok("  Saved to:", str(tok_path.relative_to(PROJECT_ROOT)))

    # Normalize text demo
    print(f"\n  normalize_text() examples:")
    norm_tests = [
        "Hello, World! It's 2024.",
        "MISTER QUILTER IS THE APOSTLE",
        "  extra   spaces   here  ",
        "Señor García (Mexican Spanish)",
    ]
    for t in norm_tests:
        print(f"  {_D}  {repr(t)}{_R}")
        print(f"    → {repr(normalize_text(t))}")

    return tok


# Section 2: Dataset


def demo_dataset(examples: list[tuple[str, str]], tok: CharTokenizer) -> ASRDataset:
    _sec("ASRDataset — single example inspection")

    dataset = ASRDataset(examples=examples, tokenizer=tok)
    _ok("Dataset repr:", repr(dataset))
    _ok("Num examples:", str(len(dataset)))

    # Inspect 3 individual examples
    print(f"\n  Per-example shapes (first 3):")
    print(f"  {'idx':>4}  {'mel shape':>14}  {'tokens':>8}  {'transcript'}")
    print(f"  {'─' * 60}")

    for i in range(min(3, len(dataset))):
        t0 = time.perf_counter()
        mel, tokens = dataset[i]
        elapsed = (time.perf_counter() - t0) * 1000
        transcript = examples[i][1]
        print(
            f"  {i:>4}  {str(tuple(mel.shape)):>14}  "
            f"{str(tuple(tokens.shape)):>8}  "
            f"{repr(transcript[:40])}"
            f"  {_D}({elapsed:.0f}ms){_R}"
        )

    # Show a token encoding in detail for the first example
    _, transcript0 = examples[0]
    ids0 = tok.encode(transcript0)
    print(f"\n  Token detail for example 0:")
    _ok("  Transcript:", repr(transcript0))
    _ok("  Normalized:", repr(normalize_text(transcript0)))
    chars_and_ids = list(zip(normalize_text(transcript0), ids0))
    print(f"  Char→ID:   {chars_and_ids}")

    # Show mel value statistics
    mel0, _ = dataset[0]
    print(f"\n  Mel spectrogram statistics (example 0):")
    _ok("  Shape:", str(tuple(mel0.shape)))
    _ok("  dtype:", str(mel0.dtype))
    _ok("  min:", f"{mel0.min().item():.4f}")
    _ok("  max:", f"{mel0.max().item():.4f}")
    _ok("  mean:", f"{mel0.mean().item():.4f}")
    _ok("  std:", f"{mel0.std().item():.4f}")

    frames = mel0.shape[1]
    real_frames_before_pad = len([x for x in mel0[0].numpy() if x != 0.0])
    duration_s = frames * HOP_LENGTH / SAMPLE_RATE
    print(f"\n  Padding info (example 0):")
    _ok("  Padded to:", f"{frames} frames ({duration_s:.1f}s)")
    _ok("  Real audio:", f"~{examples[0][0].split('/')[-1]} duration varies")

    return dataset


# Section 3: DataLoader and batch


def demo_dataloader(examples: list[tuple[str, str]], tok: CharTokenizer) -> None:
    _sec("DataLoader — batch collation")

    BATCH_SIZE = 4
    loader = build_dataloader(
        examples=examples,
        tokenizer=tok,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    _ok("Batch size:", str(BATCH_SIZE))
    _ok("Num batches:", str(len(loader)))
    _ok("Num examples:", str(len(examples)))

    # Fetch and inspect the first batch
    print(f"\n  Fetching first batch...")
    t0 = time.perf_counter()
    batch = next(iter(loader))
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Batch tensor shapes (this is what the model receives):")
    print(f"  {'─' * 56}")
    for key, tensor in batch.items():
        val_range = f"  [{tensor.float().min():.3f}, {tensor.float().max():.3f}]"
        print(
            f"  {_B}{key:<20}{_R}  shape={str(tuple(tensor.shape)):<20}"
            f"  dtype={str(tensor.dtype):<12}"
            f"  {_D}{val_range}{_R}"
        )

    print(f"\n  Actual mel lengths (real frames before padding):")
    for i, l in enumerate(batch["mel_lengths"].tolist()):
        dur_s = l * HOP_LENGTH / SAMPLE_RATE
        print(f"    example {i}: {l} frames  ({dur_s:.2f}s real audio)")

    print(f"\n  Actual token lengths (transcript length before padding):")
    for i, l in enumerate(batch["token_lengths"].tolist()):
        toks = batch["tokens"][i, :l].tolist()
        text = tok.decode(toks)
        print(f"    example {i}: {l:>3} tokens  → {repr(text[:50])}")

    print(f"\n  {_G}Batch collated in {elapsed:.0f}ms{_R}")

    # Show what CTCLoss will see
    _sec("What nn.CTCLoss will receive")
    mel = batch["mel"]  # (B, 80, T_max)
    mel_lengths = batch["mel_lengths"]  # (B,)
    tokens = batch["tokens"]  # (B, L_max)
    token_lengths = batch["token_lengths"]  # (B,)

    print(f"""
  In train.py, the CTC training step will look like:

      # Forward pass through encoder
      log_probs = model(batch["mel"])         # (B, 80, T_max) → (T, B, C)

      # CTC loss expects time-first log probabilities
      loss = ctc_loss(
          log_probs,                           # (T, B, num_classes)
          batch["tokens"],                     # (B, L_max)
          input_lengths  = encoder_lengths,    # (B,) — frames after CNN downsampling
          target_lengths = batch["token_lengths"],  # (B,)
      )

  mel shape:           {tuple(mel.shape)}   (batch, n_mels, time)
  mel_lengths:         {mel_lengths.tolist()}  (actual frames per example)
  tokens shape:        {tuple(tokens.shape)}   (batch, max_transcript_length)
  token_lengths:       {token_lengths.tolist()}  (actual tokens per example)
    """)


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="LinguaForge Tokenizer and Dataset Demo")
    parser.add_argument(
        "--librispeech",
        default=None,
        help="Path to LibriSpeech root (e.g. data/LibriSpeech). "
        "If omitted, a synthetic toy dataset is generated.",
    )
    parser.add_argument(
        "--split",
        default="test-clean",
        help="LibriSpeech split to load (default: test-clean)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a TSV manifest file. Overrides --librispeech.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of toy examples to generate (max 10, default 10)",
    )
    args = parser.parse_args()

    # Header
    print(f"\n{_W}{'═' * 62}{_R}")
    print(f"{_W}  LinguaForge — Tokenizer and Dataset Pipeline{_R}")
    print(f"{_W}{'═' * 62}{_R}")

    # Load or generate examples
    _sec("Dataset Source")

    if args.manifest:
        print(f"  Loading from manifest: {args.manifest}")
        examples = load_manifest(args.manifest)
        print(f"  Loaded {len(examples)} examples")
    elif args.librispeech:
        print(f"  Loading LibriSpeech {args.split} from {args.librispeech}...")
        examples = load_librispeech(args.librispeech, args.split)
        print(f"  Loaded {len(examples)} examples")
        # Cap at 20 for demo speed
        if len(examples) > 20:
            examples = examples[:20]
            print(f"  (capped at 20 for demo)")
    else:
        toy_dir = PROJECT_ROOT / "data" / "toy_dataset"
        print(f"  No source provided — generating toy dataset in {toy_dir}/")
        print(
            f"  {_D}(real LibriSpeech: python scripts/demos/tokenizer_pipeline.py --librispeech data/LibriSpeech){_R}"
        )
        examples = make_toy_dataset(toy_dir, n=args.n)
        print(f"  Generated {len(examples)} synthetic audio files + manifest.tsv")

    transcripts = [t for _, t in examples]

    # Run demo sections
    tok = demo_tokenizer(transcripts)
    demo_dataset(examples, tok)
    demo_dataloader(examples, tok)

    print("\nDone.")


if __name__ == "__main__":
    main()
