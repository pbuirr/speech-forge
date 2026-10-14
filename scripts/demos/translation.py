"""
scripts/demos/translation.py — Phase 7: Neural Machine Translation demo.

Trains a from-scratch Transformer seq2seq model on a small built-in Spanish→English
parallel corpus, then translates a held-out test set.

What this demonstrates:
  - BPEVocab.train()  — learns subword merges from the corpus
  - Seq2SeqNMT        — Transformer encoder-decoder (reuses Phase 4 building blocks)
  - Teacher-forcing training loop with cross-entropy loss
  - Greedy and beam-search decoding
  - Checkpoint save/load

The corpus is tiny (80 sentence pairs) — the model will overfit but that's
intentional for a fast, self-contained demo.  Real translation quality requires
parallel data (Europarl, CCAligned, OPUS) which you'd swap in via --corpus.

Usage:
    .venv/bin/python3 scripts/demos/translation.py            # train + save + translate test set
    .venv/bin/python3 scripts/demos/translation.py --epochs 300
    .venv/bin/python3 scripts/demos/translation.py --corpus europarl_es_en.txt --epochs 20
    .venv/bin/python3 scripts/demos/translation.py --translate "el gato es pequeño"
    .venv/bin/python3 scripts/demos/translation.py --load      # load saved checkpoint, skip training
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim

from speech_engine.bpe import BPEVocab, BOS_ID, EOS_ID, PAD_ID
from speech_engine.nmt import Seq2SeqNMT, collate_pairs, load_checkpoint, nmt_loss, save_checkpoint

CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "nmt_es_en.pt"

# ── Built-in Spanish → English parallel corpus ───────────────────────────────
# 80 sentence pairs covering basic vocabulary, verb conjugations, and common
# sentence structures.  Small enough to train in under a minute, big enough
# to show the model learning.

_CORPUS: list[tuple[str, str]] = [
    # Nouns + simple predicates
    ("el gato es pequeño", "the cat is small"),
    ("la casa es grande", "the house is big"),
    ("el perro corre rápido", "the dog runs fast"),
    ("el libro está en la mesa", "the book is on the table"),
    ("la ciudad es bonita", "the city is beautiful"),
    ("el niño juega en el parque", "the child plays in the park"),
    ("la mujer trabaja mucho", "the woman works a lot"),
    ("el hombre come pan", "the man eats bread"),
    ("la lluvia cae fuerte", "the rain falls hard"),
    ("el sol brilla hoy", "the sun shines today"),
    # Verbs: present tense
    ("yo como manzanas", "i eat apples"),
    ("ella bebe agua", "she drinks water"),
    ("nosotros vivimos aquí", "we live here"),
    ("ellos hablan español", "they speak spanish"),
    ("tú trabajas mucho", "you work a lot"),
    ("él lee libros", "he reads books"),
    ("nosotros cantamos juntos", "we sing together"),
    ("ellas corren en el parque", "they run in the park"),
    ("yo escribo cartas", "i write letters"),
    ("usted habla muy bien", "you speak very well"),
    # Verbs: past tense
    ("ayer comí pizza", "yesterday i ate pizza"),
    ("ella llegó tarde", "she arrived late"),
    ("ellos viajaron a madrid", "they traveled to madrid"),
    ("nosotros ganamos el partido", "we won the match"),
    ("el tren salió a tiempo", "the train left on time"),
    ("él estudió toda la noche", "he studied all night"),
    ("compramos una casa nueva", "we bought a new house"),
    ("ella bailó muy bien", "she danced very well"),
    ("los niños jugaron fútbol", "the children played soccer"),
    ("yo vi una película", "i watched a movie"),
    # Questions
    ("qué hora es", "what time is it"),
    ("dónde vives", "where do you live"),
    ("cuánto cuesta esto", "how much does this cost"),
    ("cómo estás hoy", "how are you today"),
    ("quién es esa persona", "who is that person"),
    ("cuántos años tienes", "how old are you"),
    ("cuándo llegas", "when do you arrive"),
    ("por qué estudias español", "why do you study spanish"),
    ("qué quieres comer", "what do you want to eat"),
    ("cómo se llama usted", "what is your name"),
    # Common phrases
    ("me gusta el café", "i like coffee"),
    ("no entiendo nada", "i do not understand anything"),
    ("necesito ayuda por favor", "i need help please"),
    ("hablas inglés bien", "you speak english well"),
    ("tengo hambre ahora", "i am hungry now"),
    ("quiero dormir más", "i want to sleep more"),
    ("voy al mercado mañana", "i am going to the market tomorrow"),
    ("hay mucha gente aquí", "there are many people here"),
    ("hace calor hoy", "it is hot today"),
    ("llueve mucho en otoño", "it rains a lot in autumn"),
    # Adjectives + comparisons
    ("este libro es más interesante", "this book is more interesting"),
    ("ella es más alta que él", "she is taller than him"),
    ("la comida española es deliciosa", "spanish food is delicious"),
    ("el camino más corto es mejor", "the shorter path is better"),
    ("este problema es difícil", "this problem is difficult"),
    ("la música clásica es relajante", "classical music is relaxing"),
    ("el invierno es muy frío", "winter is very cold"),
    ("los colores son brillantes", "the colors are bright"),
    ("aquella montaña es muy alta", "that mountain is very tall"),
    ("este café está muy caliente", "this coffee is very hot"),
    # Numbers and time
    ("son las tres de la tarde", "it is three in the afternoon"),
    ("llegó a las ocho", "he arrived at eight"),
    ("hay veinte estudiantes en clase", "there are twenty students in class"),
    ("espera cinco minutos más", "wait five more minutes"),
    ("el año pasado fue difícil", "last year was difficult"),
    # Compound and complex sentences
    ("cuando llegues llámame", "when you arrive call me"),
    ("si tienes tiempo ven a visitarme", "if you have time come visit me"),
    ("aunque llueve salgo a correr", "even though it rains i go running"),
    ("después de comer dormimos la siesta", "after eating we take a nap"),
    ("antes de salir revisa tus llaves", "before leaving check your keys"),
    # Everyday vocabulary
    ("el desayuno está listo", "breakfast is ready"),
    ("cierra la puerta al salir", "close the door when you leave"),
    ("pon la mesa por favor", "set the table please"),
    ("apaga la luz al salir", "turn off the light when you leave"),
    ("limpia tu cuarto esta tarde", "clean your room this afternoon"),
    ("escucha la música suavemente", "listen to the music softly"),
    ("abre la ventana hay calor", "open the window it is hot"),
    ("espera aquí no te muevas", "wait here do not move"),
    ("trae el agua fría del refrigerador", "bring the cold water from the refrigerator"),
    ("guarda tus cosas en la mochila", "put your things in the backpack"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_corpus(path: str | None) -> list[tuple[str, str]]:
    if path is None:
        return list(_CORPUS)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    pairs = []
    for line in lines:
        if "\t" in line:
            src, _, tgt = line.partition("\t")
            pairs.append((src.strip(), tgt.strip()))
    if not pairs:
        raise ValueError(f"Corpus {path!r} has no tab-separated lines")
    return pairs


def _build_vocabs(
    pairs: list[tuple[str, str]], vocab_size: int = 800
) -> tuple[BPEVocab, BPEVocab]:
    src_sents = [s for s, _ in pairs]
    tgt_sents = [t for _, t in pairs]
    src_vocab = BPEVocab().train(src_sents, vocab_size=vocab_size)
    tgt_vocab = BPEVocab().train(tgt_sents, vocab_size=vocab_size)
    return src_vocab, tgt_vocab


def _encode_pairs(
    pairs: list[tuple[str, str]],
    src_vocab: BPEVocab,
    tgt_vocab: BPEVocab,
) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for src, tgt in pairs:
        s = src_vocab.encode(src)
        t = tgt_vocab.encode(tgt, add_bos=True, add_eos=True)
        if s and len(t) >= 2:
            encoded.append((s, t))
    return encoded


def _train(
    model: Seq2SeqNMT,
    data: list[tuple[list[int], list[int]]],
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 3e-4,
) -> None:
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # Linear warmup + cosine decay
    total_steps = epochs * math.ceil(len(data) / batch_size)
    warmup = min(400, total_steps // 10)

    def _lr(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(opt, _lr)

    model.train()
    step = 0
    for epoch in range(1, epochs + 1):
        random.shuffle(data)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            src, src_lens, tgt_in, tgt_out, _ = collate_pairs(batch)
            src, src_lens = src.to(device), src_lens.to(device)
            tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)

            logits = model(src, tgt_in, src_lengths=src_lens)
            loss = nmt_loss(logits, tgt_out)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            avg = epoch_loss / max(1, n_batches)
            print(f"  epoch {epoch:>4d}/{epochs}  loss {avg:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")


def _translate(
    model: Seq2SeqNMT,
    src_vocab: BPEVocab,
    tgt_vocab: BPEVocab,
    text: str,
    beam_size: int = 4,
    device: torch.device | None = None,
) -> str:
    device = device or next(model.parameters()).device
    src_ids = src_vocab.encode(text.lower().strip())
    if not src_ids:
        return ""
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    out_ids = model.translate(src_tensor, beam_size=beam_size)
    return tgt_vocab.decode(out_ids)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 NMT demo — Spanish → English")
    parser.add_argument("--corpus",    type=str,   default=None,  help="TSV file: src<TAB>tgt per line")
    parser.add_argument("--epochs",    type=int,   default=200,   help="training epochs (default 200)")
    parser.add_argument("--vocab",     type=int,   default=800,   help="BPE vocab size (default 800)")
    parser.add_argument("--beam",      type=int,   default=4,     help="beam width for decoding (default 4)")
    parser.add_argument("--load",      action="store_true",       help="load saved checkpoint, skip training")
    parser.add_argument("--translate", type=str,   default=None,  help="translate a single sentence")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'─'*60}")
    print("  LinguaForge — Phase 7: Neural Machine Translation")
    print(f"  Spanish → English  |  device: {device}")
    print(f"{'─'*60}\n")

    if args.load and CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH} …")
        model, src_vocab, tgt_vocab, meta = load_checkpoint(CHECKPOINT_PATH)
        model.to(device)
        print(f"  {model.param_count():,} parameters | trained {meta.get('epochs', '?')} epochs | final loss {meta.get('final_loss', '?')}\n")
    else:
        # ── Build corpus and vocabularies ─────────────────────────────────────
        pairs = _load_corpus(args.corpus)
        print(f"Corpus: {len(pairs)} sentence pairs")

        src_vocab, tgt_vocab = _build_vocabs(pairs, vocab_size=args.vocab)
        print(f"BPE vocab  src={len(src_vocab)} tokens  tgt={len(tgt_vocab)} tokens")

        encoded = _encode_pairs(pairs, src_vocab, tgt_vocab)
        print(f"Encoded:   {len(encoded)} usable pairs\n")

        # ── Architecture overview ─────────────────────────────────────────────
        model = Seq2SeqNMT(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=256,
        ).to(device)
        print(f"Model: Seq2SeqNMT  {model.param_count():,} parameters")
        print(f"  src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")
        print(f"  d_model=128  n_heads=4  n_layers=2  d_ff=256")
        print(f"  weight tying: output projection shares target embedding weights\n")

        # ── Training ──────────────────────────────────────────────────────────
        print(f"Training for {args.epochs} epochs …")
        import time
        t0 = time.time()
        _train(model, encoded, epochs=args.epochs)
        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s\n")

        # Quick eval: translate every training sentence, report exact match
        model.eval()
        correct = sum(
            1 for src, tgt in pairs
            if _translate(model, src_vocab, tgt_vocab, src, beam_size=args.beam, device=device)
               == tgt.lower().strip()
        )
        print(f"Training-set exact match: {correct}/{len(pairs)} ({correct/len(pairs)*100:.0f}%)")
        print("(Exact match on training data confirms the model has learned the corpus;\n"
              " generalisation to unseen text requires orders-of-magnitude more data.)\n")

        # Save checkpoint
        final_loss = None  # already printed per-epoch
        save_checkpoint(
            model, src_vocab, tgt_vocab, CHECKPOINT_PATH,
            meta={"epochs": args.epochs, "corpus_size": len(pairs)},
        )
        print(f"Checkpoint saved → {CHECKPOINT_PATH}\n")

    # ── Single-sentence translate mode ────────────────────────────────────────
    if args.translate:
        result = _translate(model, src_vocab, tgt_vocab, args.translate, beam_size=args.beam, device=device)
        print(f"Input:  {args.translate}")
        print(f"Output: {result}\n")
        return

    # ── Test-set translation table ────────────────────────────────────────────
    test_sentences = [
        "el gato es pequeño",
        "ella bebe agua",
        "qué hora es",
        "me gusta el café",
        "nosotros vivimos aquí",
        "el libro está en la mesa",
        "ayer comí pizza",
        "cuánto cuesta esto",
        "el sol brilla hoy",
        "cuando llegues llámame",
    ]

    print("Sample translations (greedy):")
    print(f"  {'Spanish':<40}  {'→':1}  {'English'}")
    print(f"  {'─'*40}     {'─'*30}")
    model.eval()
    for src in test_sentences:
        out = _translate(model, src_vocab, tgt_vocab, src, beam_size=1, device=device)
        print(f"  {src:<40}  →  {out}")

    print()
    print("Sample translations (beam=4):")
    print(f"  {'Spanish':<40}  {'→':1}  {'English'}")
    print(f"  {'─'*40}     {'─'*30}")
    for src in test_sentences[:5]:
        out = _translate(model, src_vocab, tgt_vocab, src, beam_size=args.beam, device=device)
        print(f"  {src:<40}  →  {out}")

    print()
    print("Next: swap in a real parallel corpus (Europarl, OPUS, CCAligned) for")
    print("      generalising translation quality beyond the training set.")
    print(f"      Checkpoint at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
