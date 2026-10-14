"""
scripts/demos/linguistic_analysis.py — LinguaForge Phase 5: Linguistic Engine

Demonstrates:
  1. Language detection — 170+ languages, script identification
  2. Dialect detection — 60+ dialect variants
  3. Mixed-script text handling
  4. Morphological analysis — BPE segmentation, POS tagging
  5. Grammar checking — 25+ English rules, Spanish/French rules
  6. Morphological typology awareness

Usage:
    python scripts/demos/linguistic_analysis.py
    python scripts/demos/linguistic_analysis.py --text "Your own text here"
    python scripts/demos/linguistic_analysis.py --file mytext.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from speech_engine.grammar import GrammarChecker, Severity
from speech_engine.language_detector import LANG_DB, LanguageDetector
from speech_engine.morphology import BPESegmenter, MorphologyEngine, morphological_typology

# ── ANSI colours ──────────────────────────────────────────────────────────────
_G = "\033[92m"
_B = "\033[94m"
_C = "\033[96m"
_Y = "\033[93m"
_D = "\033[2m"
_W = "\033[1m"
_R = "\033[0m"
_RE = "\033[91m"  # red


def _sec(title: str) -> None:
    print(f"\n{_C}── {title} {'─' * max(0, 56 - len(title))}{_R}")


def _ok(label: str, value: str) -> None:
    print(f"  {_B}{label:<26}{_R} {value}")


def _print_detection(result) -> None:
    print(f"\n  Language: {result.language}  ({result.name})")
    print(f"  Dialect:  {result.dialect or '—'}")
    print(f"  Script:   {result.script}")
    print(f"  Conf:     {result.confidence:.3f}")
    if result.confidence < 0.3:
        print(
            f"  {_Y}⚠ Low confidence — text has too few language-distinguishing "
            f"words/characters for a reliable result. Try a longer sample "
            f"(≥ 1 full sentence with common function words).{_R}"
        )
    print(f"  Top candidates:")
    for m in result.top:
        print(f"    {m.code:<12}  {m.name:<35}  conf={m.confidence:.3f}")


# ── Section 1: Language detection across scripts ───────────────────────────────

# (text, expected_language, expected_script, description)
DETECTION_TESTS: list[tuple[str, str, str, str]] = [
    # Latin-script languages
    ("The quick brown fox jumps over the lazy dog.", "en", "Latin", "English"),
    ("La vida es sueño y los sueños sueños son.", "es", "Latin", "Spanish (Calderón de la Barca)"),
    (
        "Être ou ne pas être, telle est la question.",
        "fr",
        "Latin",
        "French (Shakespeare translated)",
    ),
    ("Das Schloss liegt am Rande des kleinen Dörfchens.", "de", "Latin", "German"),
    ("Il dolce far niente è una filosofia di vita italiana.", "it", "Latin", "Italian"),
    ("O amor é cego, mas o casamento é um abridor de olhos.", "pt", "Latin", "Portuguese"),
    (
        "De wereld is een toneel en alle mensen zijn slechts spelers.",
        "nl",
        "Latin",
        "Dutch (Shakespeare translated)",
    ),
    ("Livet er ikke det, vi lever, men det, vi mindes.", "da", "Latin", "Danish"),
    ("Den som inte riskerar, han inte dricker champagne.", "sv", "Latin", "Swedish"),
    ("Jeg er ikke hjemme, men vi kommer snart tilbake.", "nb", "Latin", "Norwegian Bokmål"),
    ("Eg er ikkje heime, men dei kjem snart tilbake.", "nn", "Latin", "Norwegian Nynorsk"),
    ("Elämä on kuin tietokone, jonka käyttöjärjestelmä on rakkaus.", "fi", "Latin", "Finnish"),
    ("Życie jest sztuką rysowania bez gumki.", "pl", "Latin", "Polish"),
    ("Život je příliš krátký na to, abychom pili špatné víno.", "cs", "Latin", "Czech"),
    ("Az élet egy ajándék, amelyet minden reggel újra kell nyitni.", "hu", "Latin", "Hungarian"),
    ("Viajar é descobrir que todo o mundo tem razão.", "pt", "Latin", "Portuguese (alt)"),
    ("Bize her şeyi gösterir, fakat hiçbir şeyi açıklamaz.", "tr", "Latin", "Turkish"),
    (
        "Ang pag-ibig ay bulag, ngunit ang kasal ay nagbubukas ng mga mata.",
        "tl",
        "Latin",
        "Filipino/Tagalog",
    ),
    ("Tình yêu là mù quáng, nhưng hôn nhân khôi phục thị lực.", "vi", "Latin", "Vietnamese"),
    ("Bahasa adalah cermin peradaban.", "id", "Latin", "Indonesian"),
    ("Usiku wa manane mwizi huingia.", "sw", "Latin", "Swahili"),
    ("Bụt chùa nhà không thiêng.", "vi", "Latin", "Vietnamese (proverb)"),
    # Cyrillic
    (
        "Жизнь — это то, что с тобой происходит, пока ты строишь другие планы.",
        "ru",
        "Cyrillic",
        "Russian",
    ),
    ("Люби та знай міру; не люби — не знай.", "uk", "Cyrillic", "Ukrainian"),
    ("Животот е кратка прошетка низ убаво градинче.", "mk", "Cyrillic", "Macedonian"),
    ("Животът е красив, но трябва да знаете къде да гледате.", "bg", "Cyrillic", "Bulgarian"),
    # Arabic script
    ("إن الحياة سفر والسفر حياة.", "ar", "Arabic", "Arabic"),
    ("زندگی کوتاه است، پس آن را دوست بدار.", "fa", "Arabic", "Persian/Farsi"),
    ("زندگی کی قدر کرو، کیونکہ وقت واپس نہیں آتا۔", "ur", "Arabic", "Urdu"),
    # Indic scripts
    ("जीवन में सफलता के लिए मेहनत और लगन जरूरी है।", "hi", "Devanagari", "Hindi"),
    ("আমাদের জীবন একটি সুন্দর উপহার।", "bn", "Bengali", "Bengali"),
    ("வாழ்க்கை என்பது ஒரு அழகான பயணம்.", "ta", "Tamil", "Tamil"),
    ("జీవితం ఒక అద్భుతమైన ప్రయాణం.", "te", "Telugu", "Telugu"),
    ("ಜೀವನ ಒಂದು ಸುಂದರ ಪ್ರಯಾಣ.", "kn", "Kannada", "Kannada"),
    # East Asian
    ("生命中最重要的不是我们活了多少年，而是在这些年里我们活得有多精彩。", "zh", "CJK", "Chinese"),
    (
        "人生とは、あなたが何者であるかではなく、何をするかで決まるものです。",
        "ja",
        "Hiragana",
        "Japanese",
    ),
    ("인생은 짧고 예술은 길다.", "ko", "Hangul", "Korean"),
    # Southeast Asian
    ("ชีวิตเป็นเรื่องสั้น ดังนั้นจงทำให้มันคุ้มค่า.", "th", "Thai", "Thai"),
    ("ជីវិតជាការធ្វើដំណើរដ៏ស្រស់ស្អាត.", "km", "Khmer", "Khmer/Cambodian"),
    # Script families
    ("ცხოვრება მოკლეა, ხელოვნება კი მარადიული.", "ka", "Georgian", "Georgian"),
    ("Կյանքն արվեստ է, սովոր ճշտի:", "hy", "Armenian", "Armenian"),
    ("Η ζωή είναι σύντομη· η τέχνη αιώνια.", "el", "Greek", "Greek"),
    ("חיים קצרים, אמנות ארוכה.", "he", "Hebrew", "Hebrew"),
    ("ህይወት በጣም ቆንጆ ነው እና በፍቅር የተሞላ ነው.", "am", "Ethiopic", "Amharic/Ethiopic"),
]

# Dialect detection tests
DIALECT_TESTS: list[tuple[str, str, str]] = [
    # English dialects
    ("I love soccer, and getting a math test is no fun. Gotten used to it though.", "en", "en-US"),
    (
        "She loves football, maths is brilliant, and it's been ages since I had a proper cuppa.",
        "en",
        "en-GB",
    ),
    ("Arvo drinks at the barbie, heaps of people reckon it's the best.", "en", "en-AU"),
    ("Eh, we went to the washroom by the hockey rink, eh. The toque was warm.", "en", "en-CA"),
    (
        "Please do the needful and revert back as soon as possible. Kindly arrange a lakh rupees.",
        "en",
        "en-IN",
    ),
    # Spanish dialects
    ("¿Vosotros queréis venir? ¡Coged el ordenador y venid!", "es", "es-ES"),
    ("¡Che, boludo! ¿Querés laburar con nosotros? ¡Qué copado!", "es", "es-AR"),
    ("Órale, qué chido ese chamaco. Mande, señor.", "es", "es-MX"),
    # Portuguese dialects
    ("Você está indo de ônibus? Que legal! O celular tá sem bateria.", "pt", "pt-BR"),
    ("Vais de autocarro? Que fixe! O telemóvel está sem bateria.", "pt", "pt-PT"),
    # French dialects
    ("Ostie, j'ai magasiné toute la fin de semaine avec mon chum!", "fr", "fr-CA"),
    (
        "En Belgique on dit septante et nonante, pas soixante-dix et quatre-vingt-dix.",
        "fr",
        "fr-BE",
    ),
    # Arabic dialects
    ("إيه ده؟ ازيك يا صاحبي؟ كده مش كويس.", "ar", "ar-EG"),
    ("واش كاين شي جديد؟ هاد الشي بزاف.", "ar", "ar-MA"),
    ("شو بدك؟ هلق رح نروح عالبيت.", "ar", "ar-LB"),
    # Chinese
    ("嘅係咁㗎，唔係喺嗰度㗎，你睇佢做咩！", "zh", "yue"),  # Cantonese
    ("个时来说为问样国达", "zh", "zh-Hans"),  # Simplified
]


def demo_language_detection(detector: LanguageDetector) -> None:
    _sec(f"Language Detection — {len(DETECTION_TESTS)} languages")

    print(
        f"\n  {'Text (truncated)':<42}  {'Expected':<12}  {'Got':<12}  {'Script':<10}  {'Conf':<6}  {'✓'}"
    )
    print(f"  {'─' * 42}  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 6}  {'─' * 3}")

    correct = 0
    for text, expected_lang, expected_script, desc in DETECTION_TESTS:
        result = detector.detect(text, top_n=3)
        got = result.language
        hit = got == expected_lang
        if hit:
            correct += 1
        mark = f"{_G}✓{_R}" if hit else f"{_RE}✗{_R}"
        got_str = f"{got}" + (f" ({result.dialect})" if result.dialect else "")
        print(
            f"  {repr(text[:40]):<42}  {expected_lang:<12}  {got_str:<30}  {result.script:<10}  "
            f"{result.confidence:<6.2f}  {mark}"
        )

    pct = 100 * correct / len(DETECTION_TESTS)
    print(f"\n  Accuracy: {correct}/{len(DETECTION_TESTS)} = {pct:.1f}%")
    print(f"  {_D}(Note: short texts score lower — accuracy improves with ≥50 chars){_R}")


def demo_dialect_detection(detector: LanguageDetector) -> None:
    _sec(f"Dialect Detection — {len(DIALECT_TESTS)} dialect variants")

    print(f"\n  {'Text (truncated)':<46}  {'Expected':<12}  {'Got dialect':<16}  {'✓'}")
    print(f"  {'─' * 46}  {'─' * 12}  {'─' * 16}  {'─' * 3}")

    correct = 0
    for text, lang, expected_dialect in DIALECT_TESTS:
        result = detector.detect(text)
        got_d = result.dialect or "—"
        hit = got_d == expected_dialect
        if hit:
            correct += 1
        mark = f"{_G}✓{_R}" if hit else f"{_Y}~{_R}"  # yellow for partial
        print(f"  {repr(text[:44]):<46}  {expected_dialect:<12}  {got_d:<16}  {mark}")

    pct = 100 * correct / len(DIALECT_TESTS)
    print(f"\n  Dialect accuracy: {correct}/{len(DIALECT_TESTS)} = {pct:.1f}%")


def demo_script_analysis(detector: LanguageDetector) -> None:
    _sec("Script Analysis — Mixed-Language Texts")

    mixed_samples = [
        ("I love 東京 and also 漢字!", "English + Japanese CJK"),
        ("السلام عليكم Hello مرحبا", "Arabic + English"),
        ("Привет world! Как дела?", "Russian + English"),
        ("한국어와 English를 같이 쓰는 예시입니다.", "Korean + English"),
        ("Namasté! 你好! مرحبا! Hello! Привет!", "5 scripts"),
    ]

    for text, desc in mixed_samples:
        info = detector.detect_script(text)
        scripts = ", ".join(
            f"{s}={v:.0%}" for s, v in sorted(info.distribution.items(), key=lambda x: -x[1])
        )
        result = detector.detect(text)
        print(f"\n  {desc}:")
        print(f"    Text:    {repr(text[:60])}")
        print(f"    Scripts: {scripts}")
        print(
            f"    Best:    {result.language} ({result.name})  conf={result.confidence:.2f}  mixed={info.is_mixed}"
        )


def demo_morphology(engine: MorphologyEngine) -> None:
    _sec("Morphological Analysis — English")

    words_en = [
        "unbelievably",
        "internationalization",
        "running",
        "realization",
        "prefabricated",
        "misunderstood",
        "overcrowded",
        "disagreement",
        "cats",
        "faster",
        "beautifully",
        "antidisestablishmentarianism",
        "postmodern",
        "hyperventilation",
    ]

    print(f"\n  {'Word':<30}  {'Root':<18}  {'Morphemes':<35}  {'POS':<6}  {'Features'}")
    print(f"  {'─' * 30}  {'─' * 18}  {'─' * 35}  {'─' * 6}  {'─' * 20}")
    for word in words_en:
        a = engine.analyze_word(word, lang="en")
        morphs = " | ".join(a.morphemes)
        feats = ", ".join(f"{k}={v}" for k, v in a.features.to_dict().items())
        print(f"  {word:<30}  {a.root:<18}  {morphs:<35}  {a.pos:<6}  {feats}")

    # POS tagging demo
    _sec("POS Tagging — Multiple Languages")

    sentences = [
        ("en", "The quick brown fox jumps over the lazy dog."),
        ("es", "El rápido zorro marrón salta sobre el perro perezoso."),
        ("fr", "Le renard brun rapide saute par-dessus le chien paresseux."),
        ("de", "Der schnelle braune Fuchs springt über den faulen Hund."),
        ("tr", "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar."),
        ("ja", "速い茶色のキツネが怠け者の犬を飛び越えた。"),
    ]

    for lang, sent in sentences:
        result = engine.analyze(sent, lang=lang)
        tagged = [(a.surface, a.pos) for a in result.analyses]
        tok_str = "  ".join(f"{w}/{p}" for w, p in tagged[:10])
        defn = LANG_DB.get(lang)
        typology = morphological_typology(lang)
        print(f"\n  [{lang} — {typology}] {defn.name if defn else lang}")
        print(f"    {tok_str}")


def demo_bpe(engine: MorphologyEngine) -> None:
    _sec("BPE Morpheme Segmentation — Training from Scratch")

    # Small Turkish corpus to show agglutination
    turkish_corpus = [
        "gidiyorum gidiyorsun gidiyor gidiyoruz gidiyorsunuz gidiyorlar",
        "koşmak koşuyor koştu koşacak koşuyordu koşmaktadır",
        "evde evden eve evlerde evlerden evlere",
        "geliyorum geliyorsun geliyor geliyoruz geliyorsunuz geliyorlar",
        "kitap kitabı kitabın kitaba kitaplar kitapları kitaplarda",
        "sevmek seviyor sevdi sevecek seviyordu sevmektedir",
        "okumak okuyor okudu okuyacak okuyordu okumaktadır",
        "yazmak yazıyor yazdı yazacak yazıyordu yazmaktadır",
        "çalışmak çalışıyor çalıştı çalışacak çalışıyordu çalışmaktadır",
        "anlamak anlıyor anladı anlayacak anlıyordu anlamaktadır",
    ]

    seg = BPESegmenter(vocab_size=200)
    seg.fit(turkish_corpus, min_freq=1)

    test_words = [
        "gidiyorum",  # I am going
        "evlerden",  # from the houses
        "kitaplarımda",  # in my books
        "sevmektedir",  # he/she loves (formal)
        "çalışmaktadır",  # he/she is working (formal)
        "okumaktadır",  # he/she is reading (formal)
    ]

    print(f"\n  Turkish agglutination — BPE segments (trained on {len(turkish_corpus)} sentences):")
    print(f"\n  {'Word':<22}  Segments")
    print(f"  {'─' * 22}  {'─' * 40}")
    for word in test_words:
        pieces = seg.segment(word)
        print(f"  {word:<22}  {'  |  '.join(pieces)}")

    print(f"\n  {_D}Learned {len(seg.merges)} merge rules.  With a larger corpus (10k+ sentences),")
    print(f"  BPE discovers real Turkish morpheme boundaries.{_R}")

    # Also demo Finnish (another agglutinative language)
    finnish_corpus = [
        "talossa talosta taloon taloissa taloista taloihin",
        "juoksen juokset juoksee juoksemme juoksette juoksevat",
        "kirjassa kirjasta kirjaan kirjoissa kirjoista kirjoihin",
        "katson katsot katsoo katsomme katsotte katsovat",
        "pidän pidät pitää pidämme pidätte pitävät",
    ]

    seg_fi = BPESegmenter(vocab_size=150)
    seg_fi.fit(finnish_corpus, min_freq=1)

    fi_words = ["talossa", "kirjoihin", "juoksemme", "katsovat"]
    print(f"\n  Finnish — BPE segments:")
    print(f"  {'─' * 22}  {'─' * 40}")
    for word in fi_words:
        pieces = seg_fi.segment(word)
        print(f"  {word:<22}  {'  |  '.join(pieces)}")


def demo_typology() -> None:
    _sec("Morphological Typology — Language Families")

    typology_examples = {
        "agglutinative": [
            ("tr", "sevmektedir", "love-PROG-3SG (formal)"),
            ("fi", "talossani", "house-IN-my"),
            ("hu", "házaimban", "house-PL-my-in"),
            ("ko", "갑니다", "go-FORMAL-DECL"),
            ("sw", "nitakusaidia", "1SG-FUT-2SG-help"),
            ("eu", "etxean", "house-in"),
        ],
        "fusional": [
            ("ru", "хожу", "I walk (person+tense fused)"),
            ("ar", "يكتبون", "they write (subject in verb)"),
            ("de", "gehe", "I go (person+mood fused)"),
            ("es", "hablo", "I speak (person+tense fused)"),
        ],
        "isolating": [
            ("zh", "我 去 北京", "I go Beijing (each word=one morpheme)"),
            ("vi", "tôi đi", "I go (no inflection)"),
            ("th", "ฉัน ไป", "I go (particles for grammar)"),
        ],
        "analytic": [
            ("en", "I will have gone", "auxiliaries encode tense/aspect"),
            ("af", "ek sal gaan", "I will go"),
        ],
    }

    for typology, examples in typology_examples.items():
        print(f"\n  {_Y}{typology.upper()}{_R}  (morphemes stack to create meaning):")
        for lang, example, gloss in examples:
            defn = LANG_DB.get(lang)
            name = defn.name if defn else lang
            print(f"    [{lang}] {name:<22}  {example:<25}  — {gloss}")


def demo_grammar(checker: GrammarChecker) -> None:
    _sec("Grammar Checking — English Rules")

    test_cases = [
        (
            "He don't have nothing to eat, and her sister aren't happy neither.",
            "en",
            "double negative + SVA",
        ),
        ("I seen a elephant yesterday. She go to store every morning.", "en", "SVA + a/an"),
        ("Their going to there house to get there stuff.", "en", "their/there/they're confusion"),
        (
            "You should of called me, irregardless of what happened.",
            "en",
            "should of + irregardless",
        ),
        ("The past history of end results is filled with added bonuses.", "en", "redundancy"),
        ("She was born in 1990.She is now 34 years old.She lives in Paris.", "en", "punctuation"),
        (
            "This sentence is perfectly correct and there are no issues whatsoever with any of the grammar or spelling.",
            "en",
            "should be clean (errors = 0)",
        ),
        # Spanish
        ("Como estas? Adonde vas?", "es", "missing accents in interrogatives"),
        # French
        ("Je pas comprends pourquoi tu fais ça.", "fr", "missing 'ne' in negation"),
    ]

    for text, lang, desc in test_cases:
        result = checker.check(text, lang=lang)
        n_err = len([e for e in result.errors if e.severity == Severity.ERROR])
        n_warn = len([e for e in result.errors if e.severity == Severity.WARNING])
        n_sty = len([e for e in result.errors if e.severity == Severity.STYLE])

        print(f"\n  [{lang}] {desc}:")
        print(f"    Text:  {repr(text[:70])}")
        print(f"    Score: {result.score:.2f}  |  errors={n_err}  warnings={n_warn}  style={n_sty}")

        for e in result.errors[:4]:
            sug = f" → '{e.suggestion}'" if e.suggestion and len(e.suggestion) < 40 else ""
            sev_col = (
                _RE
                if e.severity == Severity.ERROR
                else _Y
                if e.severity == Severity.WARNING
                else _D
            )
            print(f"    {sev_col}[{e.severity.upper():<7}]{_R} {e.message[:70]}{sug}")


def demo_coverage(detector: LanguageDetector) -> None:
    _sec("Coverage Summary")

    langs = detector.available_languages()
    dialects = detector.available_dialects()

    # Count by script
    script_counts: dict[str, int] = {}
    for code, defn in LANG_DB.items():
        if defn.dialect_of is None:
            script_counts[defn.script] = script_counts.get(defn.script, 0) + 1

    print(f"\n  Total languages:  {len(langs)}")
    print(f"  Total dialects:   {len(dialects)}")
    print(f"  Scripts covered:  {len(script_counts)}")
    print(f"\n  Languages per script:")
    for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
        print(f"    {script:<20}  {count} languages")

    print(f"\n  Dialect variants per language:")
    from collections import Counter

    parent_counts: Counter = Counter(
        defn.dialect_of for code, defn in LANG_DB.items() if defn.dialect_of
    )
    for parent, count in parent_counts.most_common(10):
        defn = LANG_DB.get(parent)
        name = defn.name if defn else parent
        print(f"    {name:<25}  {count} variants")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="LinguaForge Phase 5: Linguistic Engine")
    parser.add_argument("--text", default=None, help="Detect language of a custom text")
    parser.add_argument("--file", default=None, help="Detect language of a text file")
    parser.add_argument("--grammar", default=None, help="Grammar-check a custom English text")
    args = parser.parse_args()

    print(f"\n{_W}{'═' * 66}{_R}")
    print(f"{_W}  LinguaForge — Phase 5: Linguistic Engine{_R}")
    print(f"{_W}{'═' * 66}{_R}")

    detector = LanguageDetector()
    engine = MorphologyEngine()
    checker = GrammarChecker()

    if args.text:
        _print_detection(detector.detect(args.text))
        return

    if args.file:
        text = Path(args.file).read_text()
        _print_detection(detector.detect(text))
        return

    if args.grammar:
        r = checker.check(args.grammar, lang="en")
        print(f"\n{r.summary()}")
        return

    # Full demo
    demo_language_detection(detector)
    demo_dialect_detection(detector)
    demo_script_analysis(detector)
    demo_morphology(engine)
    demo_bpe(engine)
    demo_typology()
    demo_grammar(checker)
    demo_coverage(detector)

    print(f"\n{_W}{'─' * 66}{_R}")
    print(f"{_G}Phase 5 complete.{_R}  Linguistic engine built from scratch.\n")
    print("What you built:")
    print("  • Language detector — 170+ languages, all major scripts    (language_detector.py)")
    print("  • Dialect detector  — 60+ dialect variants                 (language_detector.py)")
    print("  • Unicode script classifier — 35+ script families          (language_detector.py)")
    print("  • BPE segmenter — subword morpheme splitting               (morphology.py)")
    print("  • POS tagger — rule-based + statistical, 10 languages      (morphology.py)")
    print("  • Morphological analysis — typology-aware                  (morphology.py)")
    print("  • Grammar checker — 25+ English rules + ES/FR rules        (grammar.py)")
    print("""
Next: Phase 6 — Learning Engine
  speech_engine/flashcard.py     — SRS flashcard system (SM-2 algorithm)
  speech_engine/exercise.py      — fill-in, multiple choice, pronunciation
  speech_engine/srs.py           — spaced repetition scheduler
  scripts/demos/flashcards.py
""")


if __name__ == "__main__":
    main()
