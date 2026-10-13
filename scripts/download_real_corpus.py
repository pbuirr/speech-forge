"""
scripts/download_real_corpus.py — Language-agnostic parallel corpus downloader.

Downloads from OPUS (https://opus.nlpl.eu/), the largest open collection of
translated texts, covering 500+ language pairs across multiple domains:

  - OpenSubtitles  — movie & TV dialogue  (colloquial, spoken registers)
  - Tatoeba        — community sentences   (broad everyday coverage)
  - Europarl       — EU Parliament         (formal, high-quality)
  - GlobalVoices   — multilingual news     (journalism register)

Language-pair agnostic: the same script downloads ES↔EN, FR↔EN, DE↔EN,
ZH↔EN, or any other pair available on OPUS. All quality filters and format
handling are language-independent.

Usage:
    # Spanish → English (default)
    python3 scripts/download_real_corpus.py

    # French → English
    python3 scripts/download_real_corpus.py --src fr --tgt en

    # German → English, only Tatoeba
    python3 scripts/download_real_corpus.py --src de --tgt en --corpora tatoeba

    # List available corpus sources
    python3 scripts/download_real_corpus.py --list
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPORA_DIR  = PROJECT_ROOT / "data" / "corpora"
CACHE_DIR    = CORPORA_DIR / ".cache"

OPUS_API = "https://opus.nlpl.eu/opusapi/"


# ── Corpus source registry ────────────────────────────────────────────────────
# Each entry describes a corpus source available on OPUS.
# The actual language pair is injected at runtime — these are domain descriptors,
# not language-specific configurations.

SOURCES: dict[str, dict] = {
    "tatoeba": {
        "id":        "Tatoeba",
        "version":   "v2023-04-12",
        "desc":      "Community-contributed sentences — broad everyday coverage",
        "size_hint": "~300k pairs per language pair",
        "sample":    None,  # take all — Tatoeba is high quality
    },
    "europarl": {
        "id":        "Europarl",
        "version":   "v8",
        "desc":      "EU Parliament proceedings — formal, high-quality translations",
        "size_hint": "~2M pairs (we sample 150k)",
        "sample":    150_000,
    },
    "opensubs": {
        "id":        "OpenSubtitles",
        "version":   "v2018",
        "desc":      "Movie & TV subtitles — natural spoken-language register",
        "size_hint": "62M pairs (we sample 200k)",
        "sample":    200_000,
    },
    "globalvoices": {
        "id":        "GlobalVoices",
        "version":   "v2018q4",
        "desc":      "Multilingual news articles — journalism register",
        "size_hint": "~400k pairs (we sample 100k)",
        "sample":    100_000,
    },
    "wikimedia": {
        "id":        "wikimedia",
        "version":   "v20230407",
        "desc":      "Wikipedia + other Wikimedia translations",
        "size_hint": "varies by language",
        "sample":    100_000,
    },
}


# ── Quality filters (language-agnostic) ───────────────────────────────────────

def _is_good_pair(src: str, tgt: str) -> bool:
    """
    Language-agnostic sentence pair quality filter.

    Rejects:
    - Very short or very long sentences
    - Extreme length-ratio mismatches (sign of misalignment)
    - Lines with low alphabetic content (numbers, codes, punctuation dumps)
    - Excessive uppercase (headers, acronym lists)
    - HTML / URL artefacts
    """
    sw = src.split()
    tw = tgt.split()

    if not (3 <= len(sw) <= 60) or not (3 <= len(tw) <= 60):
        return False

    ratio = len(tw) / max(len(sw), 1)
    if not (0.3 <= ratio <= 3.0):
        return False

    alpha_frac = sum(1 for c in src if c.isalpha()) / max(len(src), 1)
    if alpha_frac < 0.50:
        return False

    upper_frac = (
        sum(1 for c in src if c.isupper())
        / max(sum(1 for c in src if c.isalpha()), 1)
    )
    if upper_frac > 0.60:
        return False

    if any(tag in src for tag in ("<", "&amp;", "http", "www.")):
        return False

    return True


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


# ── OPUS API helpers ──────────────────────────────────────────────────────────

def _resolve_url(corpus_id: str, version: str, src_lang: str, tgt_lang: str) -> str:
    """
    Query the OPUS API to find the Moses-format zip URL for a language pair.
    Falls back to a canonical constructed URL if the API is unreachable.

    OPUS normalizes language pair direction: always lower-alpha-first. E.g.
    the en-es corpus is stored as "en-es", not "es-en".
    """
    l1, l2 = sorted([src_lang, tgt_lang])  # OPUS canonical order
    pair = f"{l1}-{l2}"

    params = f"?source={l1}&target={l2}&corpus={corpus_id}&version={version}&preprocessing=moses"
    try:
        with urllib.request.urlopen(OPUS_API + params, timeout=15) as resp:
            data = json.loads(resp.read())
        for entry in data.get("corpora", []):
            u = entry.get("url", "")
            if u.endswith(".txt.zip") and pair in u.lower():
                return u
    except Exception as e:
        print(f"  [warn] OPUS API unreachable ({e}); using canonical URL")

    return (
        f"https://object.pouta.csc.fi"
        f"/OPUS-{corpus_id}/{version}/moses/{pair}.txt.zip"
    )


def _download_zip(url: str, label: str) -> bytes | None:
    cache_key = re.sub(r"[^a-zA-Z0-9]", "_", url[-70:]) + ".zip"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        print(f"  Using cache: {cache_path.name}")
        return cache_path.read_bytes()

    print(f"  Downloading {label} …\n  {url}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "LinguaForge/1.0 corpus-downloader"}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            chunks: list[bytes] = []
            downloaded = 0
            while True:
                chunk = resp.read(1 << 17)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {downloaded/1e6:.1f} MB / {total/1e6:.1f} MB"
                        f"  ({downloaded/total*100:.0f}%)",
                        end="", flush=True,
                    )
            print()
            data = b"".join(chunks)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return data
    except Exception as e:
        print(f"  [error] Download failed: {e}")
        return None


def _extract_pairs(
    zip_bytes: bytes,
    src_lang: str,
    tgt_lang: str,
    sample: int | None,
) -> list[tuple[str, str]]:
    """
    Extract sentence pairs from an OPUS Moses-format zip.

    OPUS zips contain two aligned plain-text files, one per language.
    File extensions match the ISO 639-1 language code (e.g. .en, .es, .fr).
    We detect them dynamically, so this works for any language pair.
    """
    pairs: list[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            src_file = next((n for n in names if n.endswith(f".{src_lang}")), None)
            tgt_file = next((n for n in names if n.endswith(f".{tgt_lang}")), None)

            if not src_file or not tgt_file:
                # Some zips use the canonical order (lower alpha first)
                l1, l2 = sorted([src_lang, tgt_lang])
                src_file = next((n for n in names if n.endswith(f".{src_lang}")), None)
                tgt_file = next((n for n in names if n.endswith(f".{tgt_lang}")), None)

            if not src_file or not tgt_file:
                print(f"  [warn] No {src_lang}/{tgt_lang} files found. Contents: {names[:8]}")
                return []

            print(f"  Extracting {src_file} / {tgt_file} …")
            with zf.open(src_file) as sf, zf.open(tgt_file) as tf:
                src_lines = sf.read().decode("utf-8", errors="ignore").splitlines()
                tgt_lines = tf.read().decode("utf-8", errors="ignore").splitlines()

            total = min(len(src_lines), len(tgt_lines))
            print(f"  {total:,} raw line pairs")

            indices = (
                sorted(random.sample(range(total), min(sample * 4, total)))
                if sample and total > sample * 4
                else range(total)
            )

            seen: set[str] = set()
            for i in indices:
                s = _normalize(src_lines[i])
                t = _normalize(tgt_lines[i])
                if s not in seen and _is_good_pair(s, t):
                    seen.add(s)
                    pairs.append((s, t))
                if sample and len(pairs) >= sample:
                    break

    except Exception as e:
        print(f"  [error] Extraction failed: {e}")
    return pairs


# ── Main downloader ───────────────────────────────────────────────────────────

def download(
    src_lang: str,
    tgt_lang: str,
    source_keys: list[str],
    max_per_source: int | None = None,
    output_path: Path | None = None,
) -> None:
    """Download and merge parallel corpora for the given language pair."""
    out_path = output_path or CORPORA_DIR / f"{src_lang}_{tgt_lang}_real.tsv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[tuple[str, str]] = []
    seen_src: set[str] = set()

    for key in source_keys:
        cfg = SOURCES[key]
        sample = max_per_source or cfg["sample"]

        print(f"\n{'─'*60}")
        print(f"  {key.upper()}  |  {cfg['desc']}")
        print(f"  Language pair: {src_lang}→{tgt_lang}  |  {cfg['size_hint']}")

        url = _resolve_url(cfg["id"], cfg["version"], src_lang, tgt_lang)
        zdata = _download_zip(url, f"{key} ({src_lang}-{tgt_lang})")
        if not zdata:
            continue

        pairs = _extract_pairs(zdata, src_lang, tgt_lang, sample)
        added = 0
        for s, t in pairs:
            if s not in seen_src:
                seen_src.add(s)
                all_pairs.append((s, t))
                added += 1
        print(f"  +{added:,} unique pairs  (total so far: {len(all_pairs):,})")

    if not all_pairs:
        print("\n[error] No pairs collected — check network connection and language codes")
        sys.exit(1)

    random.shuffle(all_pairs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Parallel corpus — OPUS (opus.nlpl.eu)\n")
        f.write(f"# Language pair: {src_lang}→{tgt_lang}\n")
        f.write(f"# Sources: {', '.join(source_keys)}\n")
        f.write(f"# Total pairs: {len(all_pairs):,}\n")
        for s, t in all_pairs:
            f.write(f"{s}\t{t}\n")

    print(f"\n{'='*60}")
    print(f"Saved {len(all_pairs):,} pairs → {out_path}")
    print(f"Next: python3 scripts/phase7_train.py  (will auto-load this file)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OPUS parallel corpora for LinguaForge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--src", default="es", help="Source language ISO code (default: es)")
    parser.add_argument("--tgt", default="en", help="Target language ISO code (default: en)")
    parser.add_argument(
        "--corpora", nargs="+",
        default=["tatoeba", "europarl", "opensubs"],
        choices=list(SOURCES.keys()),
        help="Which corpus sources to download",
    )
    parser.add_argument("--max", type=int, default=None, help="Max pairs per source")
    parser.add_argument('--output', type=Path, help='Separate output for independent evaluation corpora')
    parser.add_argument("--list", action="store_true", help="List available sources and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable OPUS corpus sources (language-agnostic):\n")
        for k, v in SOURCES.items():
            print(f"  {k:<14} {v['desc']}")
            print(f"               {v['size_hint']}\n")
        return

    print(f"\nLinguaForge — Corpus Downloader")
    print(f"Language pair : {args.src} → {args.tgt}")
    print(f"Sources       : {', '.join(args.corpora)}")
    print(f"Output        : {args.output or CORPORA_DIR / f'{args.src}_{args.tgt}_real.tsv'}\n")

    download(args.src, args.tgt, args.corpora, max_per_source=args.max, output_path=args.output)


if __name__ == "__main__":
    main()
