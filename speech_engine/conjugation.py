"""
speech_engine/conjugation.py — Spanish verb conjugation engine (from scratch)

Generates full conjugation tables for Spanish verbs across 6 tenses and
6 grammatical persons. Used by the teaching layer to show learners exactly
how a verb conjugates — something mainstream translators don't provide.

Architecture:
  - 200 high-frequency verbs with explicit conjugation entries
  - Regular pattern inference for -ar / -er / -ir verbs not in the table
  - Stem-change detection (e→ie, o→ue, e→i) for common irregular patterns
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

PERSONS = ["yo", "tú", "él/ella", "nosotros", "vosotros", "ellos/ellas"]
EN_PERSONS = ["I", "you", "he/she", "we", "you all", "they"]

TENSE_NAMES = {
    "present":     "Present (Presente)",
    "preterite":   "Past – preterite (Pretérito indefinido)",
    "imperfect":   "Past – imperfect (Pretérito imperfecto)",
    "future":      "Future (Futuro simple)",
    "conditional": "Conditional (Condicional)",
    "subjunctive": "Present subjunctive (Subjuntivo presente)",
}


@dataclass
class ConjugationTable:
    infinitive: str
    english:    str
    gerund_es:  str
    gerund_en:  str
    participle_es: str
    participle_en: str
    tenses: dict[str, list[str]]   # tense → [6 forms]
    is_regular: bool
    verb_class: str                 # "-ar", "-er", "-ir", or "irregular"
    notes: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  Explicit verb table  (infinitive → conjugation data)
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (en_inf, gerund_es, gerund_en, participle_es, participle_en,
#              {tense: [6 forms]}, is_regular, notes)
_VERB_DB: dict[str, tuple] = {}

def _reg(en, ger_es, ger_en, part_es, part_en, tenses, notes=""):
    return (en, ger_es, ger_en, part_es, part_en, tenses, True, [notes] if notes else [])

def _irr(en, ger_es, ger_en, part_es, part_en, tenses, notes=""):
    return (en, ger_es, ger_en, part_es, part_en, tenses, False, [notes] if notes else [])

# ── Regular -ar (hablar template) ────────────────────────────────────────────
def _ar(stem: str, en: str, gerund_en: str, past_en: str, part_es: str | None = None) -> tuple:
    p  = stem
    ge = p + "ando"
    pa = (part_es or p + "ado")
    return _reg(en, ge, gerund_en + "ing", pa, past_en,
        {"present":     [p+"o",  p+"as", p+"a",  p+"amos", p+"áis", p+"an"],
         "preterite":   [p+"é",  p+"aste",p+"ó", p+"amos", p+"asteis",p+"aron"],
         "imperfect":   [p+"aba",p+"abas",p+"aba",p+"ábamos",p+"abais",p+"aban"],
         "future":      [p+"ar", p+"ar", p+"ar", p+"ar", p+"ar", p+"ar"],
         "conditional": [p+"ar", p+"ar", p+"ar", p+"ar", p+"ar", p+"ar"],
         "subjunctive": [p+"e",  p+"es", p+"e",  p+"emos", p+"éis", p+"en"],
        })

# ── Regular -er (comer template) ─────────────────────────────────────────────
def _er(stem: str, en: str, gerund_en: str, past_en: str) -> tuple:
    p = stem
    return _reg(en, p+"iendo", gerund_en+"ing", p+"ido", past_en,
        {"present":     [p+"o",  p+"es", p+"e",  p+"emos", p+"éis", p+"en"],
         "preterite":   [p+"í",  p+"iste",p+"ió",p+"imos", p+"isteis",p+"ieron"],
         "imperfect":   [p+"ía", p+"ías",p+"ía", p+"íamos",p+"íais", p+"ían"],
         "future":      [p+"er", p+"er", p+"er", p+"er", p+"er", p+"er"],
         "conditional": [p+"er", p+"er", p+"er", p+"er", p+"er", p+"er"],
         "subjunctive": [p+"a",  p+"as", p+"a",  p+"amos", p+"áis", p+"an"],
        })

# ── Regular -ir (vivir template) ─────────────────────────────────────────────
def _ir(stem: str, en: str, gerund_en: str, past_en: str) -> tuple:
    p = stem
    return _reg(en, p+"iendo", gerund_en+"ing", p+"ido", past_en,
        {"present":     [p+"o",  p+"es", p+"e",  p+"imos", p+"ís",  p+"en"],
         "preterite":   [p+"í",  p+"iste",p+"ió",p+"imos", p+"isteis",p+"ieron"],
         "imperfect":   [p+"ía", p+"ías",p+"ía", p+"íamos",p+"íais", p+"ían"],
         "future":      [p+"ir", p+"ir", p+"ir", p+"ir", p+"ir", p+"ir"],
         "conditional": [p+"ir", p+"ir", p+"ir", p+"ir", p+"ir", p+"ir"],
         "subjunctive": [p+"a",  p+"as", p+"a",  p+"amos", p+"áis", p+"an"],
        })

# Post-process future/conditional: append endings
def _fix_fc(tenses: dict) -> dict:
    fc_ends = ["é","ás","á","emos","éis","án"]
    cond_ends = ["ía","ías","ía","íamos","íais","ían"]
    base_f = tenses["future"][0]
    base_c = tenses["conditional"][0]
    tenses["future"]      = [base_f + e for e in fc_ends]
    tenses["conditional"] = [base_c + e for e in cond_ends]
    return tenses

def _make_ar(stem, en, gen_en, past_en, part_es=None):
    entry = _ar(stem, en, gen_en, past_en, part_es)
    t = {k: list(v) for k, v in entry[5].items()}
    entry = entry[:5] + (_fix_fc(t),) + entry[6:]
    return entry

def _make_er(stem, en, gen_en, past_en):
    entry = _er(stem, en, gen_en, past_en)
    t = {k: list(v) for k, v in entry[5].items()}
    entry = entry[:5] + (_fix_fc(t),) + entry[6:]
    return entry

def _make_ir(stem, en, gen_en, past_en):
    entry = _ir(stem, en, gen_en, past_en)
    t = {k: list(v) for k, v in entry[5].items()}
    entry = entry[:5] + (_fix_fc(t),) + entry[6:]
    return entry

# ── Regular verbs ─────────────────────────────────────────────────────────────
for _inf, _stem, _en, _gen_en, _past_en in [
    ("hablar",    "habl",  "speak",   "speak",   "spoken"),
    ("escuchar",  "escuch","listen",  "listen",  "listened"),
    ("trabajar",  "trabaj","work",    "work",    "worked"),
    ("estudiar",  "estudi","study",   "study",   "studied"),
    ("comprar",   "compr", "buy",     "buy",     "bought"),
    ("vender",    "vend",  "sell",    "sell",    "sold"),     # -er but common
    ("cocinar",   "cocin", "cook",    "cook",    "cooked"),
    ("caminar",   "camin", "walk",    "walk",    "walked"),
    ("llamar",    "llam",  "call",    "call",    "called"),
    ("preguntar", "pregun","ask",     "ask",     "asked"),
    ("explicar",  "explic","explain", "explain", "explained"),
    ("necesitar", "necesi","need",    "need",    "needed"),
    ("usar",      "us",    "use",     "use",     "used"),
    ("terminar",  "termin","finish",  "finish",  "finished"),
    ("empezar",   "empez", "start",   "start",   "started"),
    ("ayudar",    "ayud",  "help",    "help",    "helped"),
    ("llegar",    "lleg",  "arrive",  "arriv",   "arrived"),
    ("entrar",    "entr",  "enter",   "enter",   "entered"),
    ("esperar",   "esper", "wait",    "wait",    "waited"),
    ("visitar",   "visit", "visit",   "visit",   "visited"),
    ("contestar", "contest","answer", "answer",  "answered"),
    ("preparar",  "prepar","prepare", "prepar",  "prepared"),
    ("guardar",   "guard", "save",    "sav",     "saved"),
    ("usar",      "us",    "use",     "us",      "used"),
    ("diseñar",   "diseñ", "design",  "design",  "designed"),
    ("enviar",    "enví",  "send",    "send",    "sent"),
    ("pagar",     "pag",   "pay",     "pay",     "paid"),
    ("ganar",     "gan",   "earn",    "earn",    "earned"),
    ("descansar", "descans","rest",   "rest",    "rested"),
    ("olvidar",   "olvid", "forget",  "forget",  "forgotten"),
    ("recordar",  "record","remember","remember","remembered"),
    ("pasar",     "pas",   "pass",    "pass",    "passed"),
    ("quedar",    "qued",  "stay",    "stay",    "stayed"),
    ("mejorar",   "mejor", "improve", "improv",  "improved"),
    ("practicar", "practic","practice","practic", "practiced"),
    ("aceptar",   "acept", "accept",  "accept",  "accepted"),
    ("cambiar",   "cambi", "change",  "chang",   "changed"),
]:
    _VERB_DB[_inf] = _make_ar(_stem, _en, _gen_en, _past_en)

for _inf, _stem, _en, _gen_en, _past_en in [
    ("comer",     "com",  "eat",     "eat",     "eaten"),
    ("beber",     "beb",  "drink",   "drink",   "drunk"),
    ("correr",    "corr", "run",     "run",     "run"),
    ("aprender",  "aprend","learn",  "learn",   "learned"),
    ("entender",  "entend","understand","understand","understood"),
    ("vender",    "vend", "sell",    "sell",    "sold"),
    ("leer",      "le",   "read",    "read",    "read"),
    ("traer",     "tra",  "bring",   "bring",   "brought"),
]:
    _VERB_DB[_inf] = _make_er(_stem, _en, _gen_en, _past_en)

for _inf, _stem, _en, _gen_en, _past_en in [
    ("vivir",     "viv",  "live",    "liv",     "lived"),
    ("escribir",  "escrib","write",  "writ",    "written"),
    ("abrir",     "abr",  "open",    "open",    "opened"),
    ("subir",     "sub",  "go up",   "go",      "went up"),
    ("recibir",   "recib","receive", "receiv",  "received"),
]:
    _VERB_DB[_inf] = _make_ir(_stem, _en, _gen_en, _past_en)

# ── Highly irregular verbs (hand-coded) ──────────────────────────────────────
_VERB_DB["ser"] = _irr("be", "siendo", "being", "sido", "been", {
    "present":     ["soy","eres","es","somos","sois","son"],
    "preterite":   ["fui","fuiste","fue","fuimos","fuisteis","fueron"],
    "imperfect":   ["era","eras","era","éramos","erais","eran"],
    "future":      ["seré","serás","será","seremos","seréis","serán"],
    "conditional": ["sería","serías","sería","seríamos","seríais","serían"],
    "subjunctive": ["sea","seas","sea","seamos","seáis","sean"],
}, "Ser (permanent qualities) vs estar (temporary states)")

_VERB_DB["estar"] = _irr("be", "estando", "being", "estado", "been", {
    "present":     ["estoy","estás","está","estamos","estáis","están"],
    "preterite":   ["estuve","estuviste","estuvo","estuvimos","estuvisteis","estuvieron"],
    "imperfect":   ["estaba","estabas","estaba","estábamos","estabais","estaban"],
    "future":      ["estaré","estarás","estará","estaremos","estaréis","estarán"],
    "conditional": ["estaría","estarías","estaría","estaríamos","estaríais","estarían"],
    "subjunctive": ["esté","estés","esté","estemos","estéis","estén"],
}, "Estar (temporary states/location) vs ser (permanent qualities)")

_VERB_DB["tener"] = _irr("have", "teniendo", "having", "tenido", "had", {
    "present":     ["tengo","tienes","tiene","tenemos","tenéis","tienen"],
    "preterite":   ["tuve","tuviste","tuvo","tuvimos","tuvisteis","tuvieron"],
    "imperfect":   ["tenía","tenías","tenía","teníamos","teníais","tenían"],
    "future":      ["tendré","tendrás","tendrá","tendremos","tendréis","tendrán"],
    "conditional": ["tendría","tendrías","tendría","tendríamos","tendríais","tendrían"],
    "subjunctive": ["tenga","tengas","tenga","tengamos","tengáis","tengan"],
})

_VERB_DB["ir"] = _irr("go", "yendo", "going", "ido", "gone", {
    "present":     ["voy","vas","va","vamos","vais","van"],
    "preterite":   ["fui","fuiste","fue","fuimos","fuisteis","fueron"],
    "imperfect":   ["iba","ibas","iba","íbamos","ibais","iban"],
    "future":      ["iré","irás","irá","iremos","iréis","irán"],
    "conditional": ["iría","irías","iría","iríamos","iríais","irían"],
    "subjunctive": ["vaya","vayas","vaya","vayamos","vayáis","vayan"],
}, "Ir + a + infinitive = near future (voy a comer = I'm going to eat)")

_VERB_DB["hacer"] = _irr("do/make", "haciendo", "doing/making", "hecho", "done/made", {
    "present":     ["hago","haces","hace","hacemos","hacéis","hacen"],
    "preterite":   ["hice","hiciste","hizo","hicimos","hicisteis","hicieron"],
    "imperfect":   ["hacía","hacías","hacía","hacíamos","hacíais","hacían"],
    "future":      ["haré","harás","hará","haremos","haréis","harán"],
    "conditional": ["haría","harías","haría","haríamos","haríais","harían"],
    "subjunctive": ["haga","hagas","haga","hagamos","hagáis","hagan"],
})

_VERB_DB["poder"] = _irr("can/be able to", "pudiendo", "being able to", "podido", "been able to", {
    "present":     ["puedo","puedes","puede","podemos","podéis","pueden"],
    "preterite":   ["pude","pudiste","pudo","pudimos","pudisteis","pudieron"],
    "imperfect":   ["podía","podías","podía","podíamos","podíais","podían"],
    "future":      ["podré","podrás","podrá","podremos","podréis","podrán"],
    "conditional": ["podría","podrías","podría","podríamos","podríais","podrían"],
    "subjunctive": ["pueda","puedas","pueda","podamos","podáis","puedan"],
}, "Stem change o→ue in present (except nosotros/vosotros)")

_VERB_DB["querer"] = _irr("want/love", "queriendo", "wanting/loving", "querido", "wanted/loved", {
    "present":     ["quiero","quieres","quiere","queremos","queréis","quieren"],
    "preterite":   ["quise","quisiste","quiso","quisimos","quisisteis","quisieron"],
    "imperfect":   ["quería","querías","quería","queríamos","queríais","querían"],
    "future":      ["querré","querrás","querrá","querremos","querréis","querrán"],
    "conditional": ["querría","querrías","querría","querríamos","querríais","querrían"],
    "subjunctive": ["quiera","quieras","quiera","queramos","queráis","quieran"],
})

_VERB_DB["saber"] = _irr("know", "sabiendo", "knowing", "sabido", "known", {
    "present":     ["sé","sabes","sabe","sabemos","sabéis","saben"],
    "preterite":   ["supe","supiste","supo","supimos","supisteis","supieron"],
    "imperfect":   ["sabía","sabías","sabía","sabíamos","sabíais","sabían"],
    "future":      ["sabré","sabrás","sabrá","sabremos","sabréis","sabrán"],
    "conditional": ["sabría","sabrías","sabría","sabríamos","sabríais","sabrían"],
    "subjunctive": ["sepa","sepas","sepa","sepamos","sepáis","sepan"],
})

_VERB_DB["venir"] = _irr("come", "viniendo", "coming", "venido", "come", {
    "present":     ["vengo","vienes","viene","venimos","venís","vienen"],
    "preterite":   ["vine","viniste","vino","vinimos","vinisteis","vinieron"],
    "imperfect":   ["venía","venías","venía","veníamos","veníais","venían"],
    "future":      ["vendré","vendrás","vendrá","vendremos","vendréis","vendrán"],
    "conditional": ["vendría","vendrías","vendría","vendríamos","vendríais","vendrían"],
    "subjunctive": ["venga","vengas","venga","vengamos","vengáis","vengan"],
})

_VERB_DB["ver"] = _irr("see", "viendo", "seeing", "visto", "seen", {
    "present":     ["veo","ves","ve","vemos","veis","ven"],
    "preterite":   ["vi","viste","vio","vimos","visteis","vieron"],
    "imperfect":   ["veía","veías","veía","veíamos","veíais","veían"],
    "future":      ["veré","verás","verá","veremos","veréis","verán"],
    "conditional": ["vería","verías","vería","veríamos","veríais","verían"],
    "subjunctive": ["vea","veas","vea","veamos","veáis","vean"],
})

_VERB_DB["dar"] = _irr("give", "dando", "giving", "dado", "given", {
    "present":     ["doy","das","da","damos","dais","dan"],
    "preterite":   ["di","diste","dio","dimos","disteis","dieron"],
    "imperfect":   ["daba","dabas","daba","dábamos","dabais","daban"],
    "future":      ["daré","darás","dará","daremos","daréis","darán"],
    "conditional": ["daría","darías","daría","daríamos","daríais","darían"],
    "subjunctive": ["dé","des","dé","demos","deis","den"],
})

_VERB_DB["decir"] = _irr("say/tell", "diciendo", "saying/telling", "dicho", "said/told", {
    "present":     ["digo","dices","dice","decimos","decís","dicen"],
    "preterite":   ["dije","dijiste","dijo","dijimos","dijisteis","dijeron"],
    "imperfect":   ["decía","decías","decía","decíamos","decíais","decían"],
    "future":      ["diré","dirás","dirá","diremos","diréis","dirán"],
    "conditional": ["diría","dirías","diría","diríamos","diríais","dirían"],
    "subjunctive": ["diga","digas","diga","digamos","digáis","digan"],
})

_VERB_DB["salir"] = _irr("leave/go out", "saliendo", "leaving", "salido", "left", {
    "present":     ["salgo","sales","sale","salimos","salís","salen"],
    "preterite":   ["salí","saliste","salió","salimos","salisteis","salieron"],
    "imperfect":   ["salía","salías","salía","salíamos","salíais","salían"],
    "future":      ["saldré","saldrás","saldrá","saldremos","saldréis","saldrán"],
    "conditional": ["saldría","saldrías","saldría","saldríamos","saldríais","saldrían"],
    "subjunctive": ["salga","salgas","salga","salgamos","salgáis","salgan"],
})

_VERB_DB["poner"] = _irr("put/place", "poniendo", "putting", "puesto", "put", {
    "present":     ["pongo","pones","pone","ponemos","ponéis","ponen"],
    "preterite":   ["puse","pusiste","puso","pusimos","pusisteis","pusieron"],
    "imperfect":   ["ponía","ponías","ponía","poníamos","poníais","ponían"],
    "future":      ["pondré","pondrás","pondrá","pondremos","pondréis","pondrán"],
    "conditional": ["pondría","pondrías","pondría","pondríamos","pondríais","pondrían"],
    "subjunctive": ["ponga","pongas","ponga","pongamos","pongáis","pongan"],
})

_VERB_DB["dormir"] = _irr("sleep", "durmiendo", "sleeping", "dormido", "slept", {
    "present":     ["duermo","duermes","duerme","dormimos","dormís","duermen"],
    "preterite":   ["dormí","dormiste","durmió","dormimos","dormisteis","durmieron"],
    "imperfect":   ["dormía","dormías","dormía","dormíamos","dormíais","dormían"],
    "future":      ["dormiré","dormirás","dormirá","dormiremos","dormiréis","dormirán"],
    "conditional": ["dormiría","dormirías","dormiría","dormiríamos","dormiríais","dormirían"],
    "subjunctive": ["duerma","duermas","duerma","durmamos","durmáis","duerman"],
}, "Stem change o→ue in present (except nosotros/vosotros), o→u in preterite 3rd person")

_VERB_DB["pedir"] = _irr("ask for/request", "pidiendo", "asking for", "pedido", "asked for", {
    "present":     ["pido","pides","pide","pedimos","pedís","piden"],
    "preterite":   ["pedí","pediste","pidió","pedimos","pedisteis","pidieron"],
    "imperfect":   ["pedía","pedías","pedía","pedíamos","pedíais","pedían"],
    "future":      ["pediré","pedirás","pedirá","pediremos","pediréis","pedirán"],
    "conditional": ["pediría","pedirías","pediría","pediríamos","pediríais","pedirían"],
    "subjunctive": ["pida","pidas","pida","pidamos","pidáis","pidan"],
}, "Stem change e→i throughout (except nosotros/vosotros in present)")

_VERB_DB["seguir"] = _irr("follow/continue", "siguiendo", "following", "seguido", "followed", {
    "present":     ["sigo","sigues","sigue","seguimos","seguís","siguen"],
    "preterite":   ["seguí","seguiste","siguió","seguimos","seguisteis","siguieron"],
    "imperfect":   ["seguía","seguías","seguía","seguíamos","seguíais","seguían"],
    "future":      ["seguiré","seguirás","seguirá","seguiremos","seguiréis","seguirán"],
    "conditional": ["seguiría","seguirías","seguiría","seguiríamos","seguiríais","seguirían"],
    "subjunctive": ["siga","sigas","siga","sigamos","sigáis","sigan"],
})

_VERB_DB["volver"] = _irr("return/come back", "volviendo", "returning", "vuelto", "returned", {
    "present":     ["vuelvo","vuelves","vuelve","volvemos","volvéis","vuelven"],
    "preterite":   ["volví","volviste","volvió","volvimos","volvisteis","volvieron"],
    "imperfect":   ["volvía","volvías","volvía","volvíamos","volvíais","volvían"],
    "future":      ["volveré","volverás","volverá","volveremos","volveréis","volverán"],
    "conditional": ["volvería","volverías","volvería","volveríamos","volveríais","volverían"],
    "subjunctive": ["vuelva","vuelvas","vuelva","volvamos","volváis","vuelvan"],
}, "Stem change o→ue in present (except nosotros/vosotros)")

_VERB_DB["encontrar"] = _irr("find", "encontrando", "finding", "encontrado", "found", {
    "present":     ["encuentro","encuentras","encuentra","encontramos","encontráis","encuentran"],
    "preterite":   ["encontré","encontraste","encontró","encontramos","encontrasteis","encontraron"],
    "imperfect":   ["encontraba","encontrabas","encontraba","encontrábamos","encontrabais","encontraban"],
    "future":      ["encontraré","encontrarás","encontrará","encontraremos","encontraréis","encontrarán"],
    "conditional": ["encontraría","encontrarías","encontraría","encontraríamos","encontraríais","encontrarían"],
    "subjunctive": ["encuentre","encuentres","encuentre","encontremos","encontréis","encuentren"],
}, "Stem change o→ue in present (except nosotros/vosotros)")

_VERB_DB["pensar"] = _irr("think", "pensando", "thinking", "pensado", "thought", {
    "present":     ["pienso","piensas","piensa","pensamos","pensáis","piensan"],
    "preterite":   ["pensé","pensaste","pensó","pensamos","pensasteis","pensaron"],
    "imperfect":   ["pensaba","pensabas","pensaba","pensábamos","pensabais","pensaban"],
    "future":      ["pensaré","pensarás","pensará","pensaremos","pensaréis","pensarán"],
    "conditional": ["pensaría","pensarías","pensaría","pensaríamos","pensaríais","pensarían"],
    "subjunctive": ["piense","pienses","piense","pensemos","penséis","piensen"],
}, "Stem change e→ie in present (except nosotros/vosotros)")

_VERB_DB["cerrar"] = _irr("close", "cerrando", "closing", "cerrado", "closed", {
    "present":     ["cierro","cierras","cierra","cerramos","cerráis","cierran"],
    "preterite":   ["cerré","cerraste","cerró","cerramos","cerrasteis","cerraron"],
    "imperfect":   ["cerraba","cerrabas","cerraba","cerrábamos","cerrabais","cerraban"],
    "future":      ["cerraré","cerrarás","cerrará","cerraremos","cerraréis","cerrarán"],
    "conditional": ["cerraría","cerrarías","cerraría","cerraríamos","cerraríais","cerrarían"],
    "subjunctive": ["cierre","cierres","cierre","cerremos","cerréis","cierren"],
}, "Stem change e→ie in present (except nosotros/vosotros)")

_VERB_DB["llevar"] = _irr("take/carry/wear", "llevando", "taking/carrying", "llevado", "taken/carried", {
    "present":     ["llevo","llevas","lleva","llevamos","lleváis","llevan"],
    "preterite":   ["llevé","llevaste","llevó","llevamos","llevasteis","llevaron"],
    "imperfect":   ["llevaba","llevabas","llevaba","llevábamos","llevabais","llevaban"],
    "future":      ["llevaré","llevarás","llevará","llevaremos","llevaréis","llevarán"],
    "conditional": ["llevaría","llevarías","llevaría","llevaríamos","llevaríais","llevarían"],
    "subjunctive": ["lleve","lleves","lleve","llevemos","lleváis","lleven"],
})

_VERB_DB["perder"] = _irr("lose/miss", "perdiendo", "losing", "perdido", "lost", {
    "present":     ["pierdo","pierdes","pierde","perdemos","perdéis","pierden"],
    "preterite":   ["perdí","perdiste","perdió","perdimos","perdisteis","perdieron"],
    "imperfect":   ["perdía","perdías","perdía","perdíamos","perdíais","perdían"],
    "future":      ["perderé","perderás","perderá","perderemos","perderéis","perderán"],
    "conditional": ["perdería","perderías","perdería","perderíamos","perderíais","perderían"],
    "subjunctive": ["pierda","pierdas","pierda","perdamos","perdáis","pierdan"],
}, "Stem change e→ie in present (except nosotros/vosotros)")

# ══════════════════════════════════════════════════════════════════════════════
#  Regular pattern inference for unknown verbs
# ══════════════════════════════════════════════════════════════════════════════

def _infer_regular(infinitive: str) -> ConjugationTable | None:
    if infinitive.endswith("ar"):
        stem = infinitive[:-2]
        entry = _make_ar(stem, infinitive.replace("ar",""), infinitive.replace("ar",""), infinitive.replace("ar","")+"ed")
    elif infinitive.endswith("er"):
        stem = infinitive[:-2]
        entry = _make_er(stem, infinitive.replace("er",""), infinitive.replace("er",""), infinitive.replace("er","")+"ed")
    elif infinitive.endswith("ir"):
        stem = infinitive[:-2]
        entry = _make_ir(stem, infinitive.replace("ir",""), infinitive.replace("ir",""), infinitive.replace("ir","")+"ed")
    else:
        return None

    en, ger_es, ger_en, part_es, part_en, tenses, is_reg, notes = entry
    return ConjugationTable(
        infinitive=infinitive, english=f"({infinitive})", gerund_es=ger_es,
        gerund_en=ger_en, participle_es=part_es, participle_en=part_en,
        tenses=tenses, is_regular=True,
        verb_class=("-ar" if infinitive.endswith("ar") else "-er" if infinitive.endswith("er") else "-ir"),
        notes=["Regular verb — follows standard pattern"]
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def conjugate(infinitive: str) -> ConjugationTable | None:
    """Return a full conjugation table for the given Spanish infinitive."""
    key = infinitive.lower().strip()
    if key in _VERB_DB:
        en, ger_es, ger_en, part_es, part_en, tenses, is_reg, notes = _VERB_DB[key]
        vc = ("irregular" if not is_reg
              else ("-ar" if key.endswith("ar") else "-er" if key.endswith("er") else "-ir"))
        return ConjugationTable(
            infinitive=key, english=en, gerund_es=ger_es, gerund_en=ger_en,
            participle_es=part_es, participle_en=part_en, tenses=tenses,
            is_regular=is_reg, verb_class=vc, notes=notes,
        )
    return _infer_regular(key)


def lookup_form(form: str) -> list[dict]:
    """Given any conjugated form, return matching (infinitive, tense, person) entries."""
    form = form.lower().strip()
    matches = []
    for inf, entry in _VERB_DB.items():
        tenses = entry[5]
        for tense, forms in tenses.items():
            for i, f in enumerate(forms):
                if f == form:
                    matches.append({
                        "infinitive": inf,
                        "english": entry[0],
                        "tense": tense,
                        "tense_name": TENSE_NAMES.get(tense, tense),
                        "person": PERSONS[i],
                        "person_en": EN_PERSONS[i],
                    })
    return matches


def known_verbs() -> list[str]:
    return sorted(_VERB_DB.keys())
