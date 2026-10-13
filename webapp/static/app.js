// LinguaForge app.js — vanilla JS, no framework, no build step

// ── Utilities ─────────────────────────────────────────────────────────────────
let _speechSource = null;
let _translationEvidence = null;
function speechEvidenceNote() {
  if (_speechSource?.method === 'local_translation') return 'Unverified model-generated text. Further analysis does not verify its meaning or fidelity to the original source. Automatic learning cards are unavailable; add manually checked material in Flashcards.';
  return _speechSource ? 'Unverified speech-derived text. Analysis and translation describe the supplied transcript, not verified speech content. Check names, numbers, negation and word boundaries against the recording. Automatic learning cards are unavailable for this input; add manually checked material in Flashcards.' : '';
}
function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

async function api(url, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(url, opts);
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || `${url} failed`);
  return d;
}
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls)  n.className = cls;
  if (html) n.innerHTML = html;
  return n;
}

// ── State ─────────────────────────────────────────────────────────────────────
let _allLangs   = [];
let _srcCode    = "";        // "" = auto-detect
let _srcName    = "Auto-detect";
let _tgtCode    = "en";
let _tgtName    = "English";
let _reviewQ    = [];
let _reviewIdx  = 0;

const CONTENT_POS   = new Set(["NOUN","PROPN","VERB","ADJ","ADV"]);
const POS_PLAIN     = { NOUN:"noun", PROPN:"proper noun", VERB:"verb", AUX:"aux. verb", ADJ:"adjective", ADV:"adverb", DET:"article", PREP:"preposition", ADP:"adposition", CONJ:"conjunction", CCONJ:"conjunction", SCONJ:"conjunction", PRON:"pronoun" };
const POS_CSS       = { NOUN:"noun", PROPN:"noun", VERB:"verb", AUX:"verb", ADJ:"adj", ADV:"adv" };
const SPEECH_TAGS   = { en:"en-US", es:"es-ES", fr:"fr-FR", de:"de-DE", ja:"ja-JP", ar:"ar-SA", ru:"ru-RU", zh:"zh-CN", ko:"ko-KR", pt:"pt-BR", it:"it-IT", nl:"nl-NL", pl:"pl-PL", tr:"tr-TR", sv:"sv-SE", nb:"nb-NO", nn:"nn-NO" };

// ── Language picker ───────────────────────────────────────────────────────────
// Creates a reusable picker component. allowAuto adds the "Auto-detect" row.
function LangPicker({ btnId, dropdownId, nameId, allowAuto, onChange }) {
  const btn  = document.getElementById(btnId);
  const dd   = document.getElementById(dropdownId);
  const name = document.getElementById(nameId);
  let isOpen = false;

  function setLabel(code, n) {
    name.textContent = n;
    if (nameId === "src-name") document.getElementById("src-icon").textContent = code ? "🌐" : "🔍";
    onChange(code, n);
  }

  function fillList(list, query) {
    list.innerHTML = "";
    const q = query.toLowerCase();
    const shown = _allLangs.filter(l => !q || l.code.includes(q) || l.name.toLowerCase().includes(q) || l.script.toLowerCase().includes(q));
    if (!shown.length) {
      list.appendChild(el("div", null, `<span style="padding:0.75rem 0.9rem;display:block;color:var(--text-3);font-size:0.83rem">No match</span>`));
      return;
    }
    if (q) {
      for (const l of shown) list.appendChild(makeRow(l));
    } else {
      const groups = {};
      for (const l of shown) (groups[l.script] = groups[l.script] || []).push(l);
      for (const [script, langs] of Object.entries(groups)) {
        const lbl = el("div", "lp-group-label"); lbl.textContent = script; list.appendChild(lbl);
        for (const l of langs) list.appendChild(makeRow(l));
      }
    }
  }

  function makeRow(l) {
    const b = el("button", "lp-item");
    b.type = "button";
    b.innerHTML = `<span class="lp-item-code">${l.code}</span><span class="lp-item-name">${l.name}</span>${l.caps?.morphology ? '<span class="lp-item-deep">⚙</span>' : ""}`;
    b.addEventListener("click", () => { close(); setLabel(l.code, l.name); });
    return b;
  }

  function open() {
    isOpen = true;
    dd.innerHTML = "";
    btn.classList.add("open");

    if (allowAuto) {
      const a = el("button", "lp-auto"); a.type = "button";
      a.innerHTML = `<span class="lp-auto-code">auto</span><span class="lp-auto-name">Auto-detect</span><span class="lp-auto-tag">recommended</span>`;
      a.addEventListener("click", () => { close(); setLabel("", "Auto-detect"); });
      dd.appendChild(a);
    }

    const sw = el("div", "lp-search-wrap");
    const si = el("input"); si.type = "text"; si.placeholder = `Search ${_allLangs.length || 148} languages…`; si.autocomplete = "off";
    sw.appendChild(si); dd.appendChild(sw);

    const list = el("div", "lp-list"); dd.appendChild(list);
    dd.appendChild(el("div", "lp-footer", "⚙ marks available morphology evidence. Coverage varies; it does not indicate translation support."));

    fillList(list, "");
    si.addEventListener("input", () => fillList(list, si.value));

    dd.classList.remove("hidden");
    requestAnimationFrame(() => si.focus());
  }

  function close() {
    isOpen = false;
    dd.classList.add("hidden");
    btn.classList.remove("open");
  }

  btn.addEventListener("click", (e) => { e.stopPropagation(); isOpen ? close() : open(); });
  document.addEventListener("click", (e) => { if (isOpen && !btn.contains(e.target) && !dd.contains(e.target)) close(); });

  return { setLabel };
}

// Wire up the two pickers
const srcPicker = LangPicker({ btnId:"src-btn", dropdownId:"src-dropdown", nameId:"src-name", allowAuto:true, onChange:(c,n)=>{ _srcCode=c; _srcName=n; document.getElementById("results").classList.add("hidden"); } });
const tgtPicker = LangPicker({ btnId:"tgt-btn", dropdownId:"tgt-dropdown", nameId:"tgt-name", allowAuto:false, onChange:(c,n)=>{ _tgtCode=c; _tgtName=n; document.getElementById("results").classList.add("hidden"); } });

// Swap source ↔ target languages
document.getElementById("swap-btn").addEventListener("click", () => {
  // Can't swap when source is auto-detect (nothing concrete to put in target)
  if (!_srcCode) return;
  const hadVisibleResult = !document.getElementById('results').classList.contains('hidden');
  const transText = document.querySelector(".trans-text");
  const [prevSrcCode, prevSrcName, prevTgtCode, prevTgtName] =
        [_srcCode, _srcName, _tgtCode, _tgtName];
  srcPicker.setLabel(prevTgtCode, prevTgtName);
  tgtPicker.setLabel(prevSrcCode, prevSrcName);
  // Also swap the textarea content with the last translation if present
  if (transText && transText.textContent.trim() && hadVisibleResult) {
    document.getElementById("main-input").value = transText.textContent.trim();
    if (_translationEvidence) _speechSource = {..._translationEvidence};
    document.getElementById("results").classList.add("hidden");
  }
});

// Load language list from server
(async () => {
  try {
    const data = await api("/api/languages");
    _allLangs = data.languages;
    const cardLanguages = document.getElementById("add-lang");
    cardLanguages.innerHTML = "";
    for (const language of _allLangs) {
      const option = document.createElement("option");
      option.value = language.code; option.textContent = `${language.name} (${language.code})`;
      cardLanguages.appendChild(option);
    }
    cardLanguages.value = "en";
  } catch {
    document.getElementById("src-name").textContent = "Language list unavailable — reload";
  }
})();

// ── Main pipeline ─────────────────────────────────────────────────────────────
document.getElementById("go-btn").addEventListener("click", runTranslate);
document.getElementById('main-input').addEventListener('input', () => {
  // Editing a predicted transcript cannot establish fidelity to the recording.
  // Clearing the input begins a new, independent typed-text workflow.
  if (!document.getElementById('main-input').value.trim() && _speechSource) {
    _speechSource = null;
    document.getElementById('local-speech-status').textContent = 'Generated input cleared. New typed text starts an independent analysis.';
  } else if (_speechSource) {
    document.getElementById('local-speech-status').textContent = _speechSource.method==='local_translation'
      ? 'Edited model-generated text remains unverified; check it against the source.'
      : 'Edited speech-derived text remains unverified; check it against the recording.';
  }
  document.getElementById('results').classList.add('hidden');
});
document.getElementById("main-input").addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runTranslate();
});

async function runTranslate() {
  const sourceCode = _srcCode, targetCode = _tgtCode;
  const text = document.getElementById("main-input").value.trim();
  const sourceEvidence = _speechSource ? {..._speechSource} : null;
  if (!text) return;

  const goBtn     = document.getElementById("go-btn");
  if (goBtn.disabled) return;
  const goLabel   = document.getElementById("go-label");
  const goSpinner = document.getElementById("go-spinner");
  goBtn.disabled = true;
  goLabel.textContent = "Analyzing…";
  goSpinner.classList.remove("hidden");

  try {
    const analysis = await api("/api/analyze", { text, lang: sourceCode || "auto",input_evidence:sourceEvidence });
    const detectResult = analysis.detection;
    const effectiveSrc = analysis.lang;
    const morph = analysis.morphology;
    const gram = analysis.grammar;
    let trans;
    try {
      if (effectiveSrc.toLowerCase() === targetCode.toLowerCase()) {
        trans = {translation:text,raw_translation:text,method:"identity",attention:null,
          model_status:"Source and target selections match; showing your original text. No translation or correction was performed."};
      } else {
        trans = await api("/api/translate", { text, direction: `${effectiveSrc}→${targetCode}`, beam: 4,input_evidence:sourceEvidence });
      }
    } catch (error) {
      trans = { error: error.message || "Unavailable" };
    }

    if (_srcCode !== sourceCode || _tgtCode !== targetCode || document.getElementById("main-input").value.trim() !== text) return;
    if (JSON.stringify(_speechSource)!==JSON.stringify(sourceEvidence)) return;
    renderDetect(detectResult, effectiveSrc, sourceEvidence);
    const speechNote = speechEvidenceNote();
    if (speechNote) document.getElementById('detect-card').innerHTML += `<p class="low-conf-warn">${escHtml(speechNote)}</p>`;
    renderTranslation(trans, effectiveSrc);
    renderWords(morph, effectiveSrc);
    renderGrammar(gram, effectiveSrc);
    renderVocab(morph, effectiveSrc, analysis);
    renderAdvanced(morph, effectiveSrc);
    renderAttention(trans?.attention || null);

    // Badge for Words
    if (morph) {
      const words = (morph.tokens||[]).filter(t => t.pos !== "PUNCT" && t.surface.trim());
      setBadge("badge-words", `${words.length} ${morph.capabilities?.tokenization === "characters" ? "character tokens" : "tokens"}`);
    }

    const results = document.getElementById("results");
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (e) {
    for (const id of ["trans-card","body-words","body-grammar","body-vocab","body-advanced","body-attention"]) {
      document.getElementById(id).innerHTML = "";
    }
    for (const id of ["badge-words","badge-grammar","badge-vocab"]) setBadge(id, "");
    document.getElementById("detect-card").innerHTML = `<div class="low-conf-warn">⚠ ${escHtml(e.message)}</div>`;
    document.getElementById("results").classList.remove("hidden");
  } finally {
    goBtn.disabled = false;
    goLabel.innerHTML = "Translate &amp; Explain";
    goSpinner.classList.add("hidden");
  }
}

function setBadge(id, text, color) {
  const b = document.getElementById(id);
  if (!b) return;
  b.textContent = text;
  if (color) { b.style.color = color; b.style.background = `${color}18`; }
}

// ── Render: language detection ────────────────────────────────────────────────
function renderDetect(r, analysisLang, sourceEvidence) {
  const generated = !!sourceEvidence || r.evidence_scope === 'generated_text';
  const pct = Math.round(r.confidence * 100);
  const cls = r.confidence < 0.3 ? "low" : r.confidence < 0.65 ? "med" : "";
  const disagrees = analysisLang && analysisLang.split('-')[0] !== r.language.split('-')[0];

  document.getElementById("detect-card").innerHTML = `
    <div class="det-row">
      <div class="det-lang-badge">
        <span class="det-code">${escHtml(r.language)}</span>
        <span class="det-name">${generated ? 'Transcript/text hypothesis' : 'Detected'}: ${escHtml(r.name)}</span>
      </div>
      <div class="det-meta">
        <span>Script: <b>${escHtml(r.script)}</b></span>
        ${r.dialect ? `<span class="det-dialect-tag">🗺 ${escHtml(r.dialect)}</span>` : ""}
        ${r.is_mixed ? `<span style="color:var(--yellow)">⚠ Mixed script</span>` : ""}
      </div>
    </div>
    ${generated ? '<p class="muted-sm">Language detection examines generated text, not the original audio. Its ranking cannot verify the spoken language.</p>' : `<div class="conf-row">
      <span class="conf-label">Detection score (uncalibrated)</span>
      <div class="conf-track"><div class="conf-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>`}
    <p class="muted-sm">Analysis language: ${escHtml(analysisLang || r.language)}. Detection scores do not measure linguistic correctness.</p>
    ${disagrees ? '<div class="low-conf-warn">Automatic detection disagrees with your selected source language. Analysis uses your selection; check that it matches the text.</div>' : ''}
    ${r.confidence < 0.3 ? `<div class="low-conf-warn" style="margin-top:0.65rem">⚠ Weak detection evidence — check the source language.</div>` : ""}`;
}

// ── Render: translation ───────────────────────────────────────────────────────
function renderTranslation(r, srcLang) {
  _translationEvidence = r && !r.error && r.method!=='identity'
    ? (r.output_evidence || {method:'local_translation',language:_tgtCode}) : null;
  const card = document.getElementById("trans-card");

  if (r.error) {
    const noModel = r.error.includes("No trained model") || r.error.includes("no such endpoint");
    card.innerHTML = `
      <div class="trans-header"><span class="trans-dir-label">Translation → ${_tgtName}</span></div>
      <div class="trans-unavail">
        ${noModel
          ? `Translation <b>${srcLang} → ${_tgtCode}</b> isn't available yet for this pair.<br>
             <span style="color:var(--text-3);font-size:0.8rem">${escHtml(r.error)}</span>`
          : escHtml(r.error)
        }
      </div>`;
    return;
  }

  const normalizedNote = r.normalized_input
    ? `<div class="trans-normalized">Normalized: <em>${escHtml(r.normalized_input)}</em></div>`
    : "";

  const literalBlock = r.literal_fallback
    ? `<div class="trans-literal">
         <span class="trans-literal-label">Unverified word-by-word alternative (not a correction):</span>
         ${escHtml(r.literal_fallback)}
       </div>`
    : "";

  const confidenceFlag = r.low_confidence
    ? `<span class="trans-low-conf" title="Heuristic review signal; not a calibrated probability of correctness">⚠ review recommended</span>`
    : "";

  const coverageNote = r.coverage_warnings?.length
    ? `<div class="low-conf-warn">Review source numbers or amounts not matched in the output: ${r.coverage_warnings.map(escHtml).join(', ')}. This check is heuristic; values may be written differently.</div>`
    : "";

  card.innerHTML = `
    <div class="trans-header">
      <span class="trans-dir-label">${r.method === "identity" ? `Original text · ${escHtml(srcLang)}` : `Translation · ${escHtml(srcLang)} → ${escHtml(_tgtCode)}`}</span>
      ${confidenceFlag}
      <button class="tts-btn" id="tts-btn" type="button">🔊 Play</button>
    </div>
    ${normalizedNote}
    <div class="trans-text">${r.translation ? escHtml(r.translation) : "<em style='color:var(--text-3)'>No output</em>"}</div>
    ${coverageNote}
    ${literalBlock}
    <div class="trans-model-note">${escHtml(r.model_status || 'Local Transformer; quality unverified.')} · <a href="#" id="open-fc-note">Open flashcards ↗</a></div>`;

  document.getElementById("open-fc-note")?.addEventListener("click", e => { e.preventDefault(); openFlashcards(); });

  const ttsBtn = document.getElementById("tts-btn");
  if (!window.speechSynthesis || typeof SpeechSynthesisUtterance === "undefined") {
    ttsBtn.disabled = true; ttsBtn.textContent = "Browser audio unavailable"; return;
  }
  ttsBtn.title = "Uses browser speech synthesis, not a LinguaForge voice model";
  ttsBtn.addEventListener("click", () => {
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
      ttsBtn.classList.remove("playing"); ttsBtn.innerHTML = "🔊 Play";
      return;
    }
    const u = new SpeechSynthesisUtterance(r.translation || "");
    u.lang = SPEECH_TAGS[_tgtCode] || _tgtCode;
    u.onstart = () => { ttsBtn.classList.add("playing"); ttsBtn.innerHTML = "⏹ Stop"; };
    u.onend   = () => { ttsBtn.classList.remove("playing"); ttsBtn.innerHTML = "🔊 Play"; };
    u.onerror = () => { ttsBtn.classList.remove("playing"); ttsBtn.textContent = "Audio unavailable — retry"; };
    window.speechSynthesis.speak(u);
  });
}

// ── Render: word-by-word breakdown ────────────────────────────────────────────
function feats(f) {
  const p = [];
  if ((f.Tense || f.tense) === "Past") p.push("past");
  if ((f.Tense || f.tense) === "Pres") p.push("present");
  if ((f.Number || f.number) === "Plur") p.push("plural");
  if ((f.Gender || f.gender) === "Fem") p.push("feminine");
  if ((f.Gender || f.gender) === "Masc") p.push("masculine");
  return p.join(" · ");
}

function analysisLabel(t) {
  return t.method === "corpus_context" ? "Context prediction" :
    t.method === "corpus_lookup" ? "Corpus-attested" :
    t.method === "corpus_ambiguous" ? "Ambiguous corpus matches" :
    t.method === "english_affix_rules" ? "Heuristic affix rules" :
    t.method === "structural" ? "Punctuation / number" : "Morphology unavailable";
}

function renderPosUncertainty(t) {
  const evidence = t.pos_uncertainty;
  if (!evidence || !evidence.applies_to_returned_pos) return "";
  const calibration = evidence.calibration;
  const estimate = calibration && !calibration.outside_observed_bins
    ? `POS reliability estimate: ${Math.round(calibration.accuracy_estimate * 100)}% on development data · ${calibration.development_observations} supporting observations.`
    : "POS model score is uncalibrated for this input.";
  return `<div class="limited-note">${escHtml(estimate)} Assumes correct token and sentence boundaries; it does not verify the lemma or meaning.</div>`;
}

function renderEvidence(t) {
  if (!t.candidates?.length) return "";
  const entries = t.candidates.map((c, index) => {
    const featureText = Object.entries(c.features || {}).map(([k,v]) => `${k}=${v}`).join(", ");
    const examples = (c.examples || []).map(e => `${e.sentence_id}, token ${e.token_id}, line ${e.line}`).join("; ");
    const source = c.source || {};
    let sourceLabel = escHtml(source.name || "Source");
    try {
      const url = new URL(source.url);
      if (["http:", "https:"].includes(url.protocol)) sourceLabel = `<a href="${escHtml(url.href)}" target="_blank" rel="noopener noreferrer">${sourceLabel}</a>`;
    } catch (_) { /* Retain source text when it has no usable public URL. */ }
    const selected = t.method === "corpus_context" && t.selected_candidate_indices?.includes(index) ? " · selected by context model" : "";
    return `<li><b>${escHtml(c.lemma)}</b> · ${escHtml(c.pos)} ${escHtml(featureText)}${selected}<br>
      ${c.observations} corpus observation${c.observations === 1 ? "" : "s"} · ${sourceLabel}<br>
      ${escHtml(examples)} · ${escHtml(source.declared_license || "")}</li>`;
  }).join("");
  return `<details><summary>${t.candidates.length} attested interpretation${t.candidates.length === 1 ? "" : "s"}</summary><ul>${entries}</ul></details>`;
}

function renderWords(r, lang) {
  const body = document.getElementById("body-words");
  if (!r) { body.innerHTML = `<div class="unavail-note">Word breakdown unavailable.</div>`; return; }

  const tokens = (r.tokens || []).filter(t => t.pos !== "PUNCT" && t.surface.trim());

  let html = `<div class="word-grid">`;
  for (const t of tokens) {
    const isContent = CONTENT_POS.has(t.pos);
    const posLabel  = POS_PLAIN[t.pos];
    const posCls    = POS_CSS[t.pos] || "other";
    const chipCls   = r.limited ? "word-chip is-limited" : isContent ? "word-chip is-content" : "word-chip is-function";
    const ft        = feats(t.features || {});
    const showRoot  = t.root && ["heuristic", "corpus_attested", "predicted"].includes(t.root_status) && t.root !== t.surface;
    const isVerb    = (t.pos === "VERB" || t.pos === "AUX") && lang.split("-")[0] === "es";
    const verbForm  = isVerb ? (t.root || t.surface) : null;

    html += `<div class="${chipCls}${isVerb ? ' is-verb-clickable' : ''}" ${isVerb ? `data-verb="${escHtml(verbForm)}" title="Click to see full conjugation table"` : ""}>
      <span class="wc-surface">${escHtml(t.surface)}</span>
      ${showRoot  ? `<span class="wc-root">${t.root_status === "corpus_attested" ? "Corpus lemma" : t.root_status === "predicted" ? "Predicted lemma" : "Heuristic stem"}: ${escHtml(t.root)}</span>` : ""}
      ${posLabel ? `<span class="wc-pos ${posCls}">${t.pos_status === "corpus_attested" ? "Attested" : "Predicted"} ${posLabel}</span>` : ""}
      ${t.method === "corpus_ambiguous" ? `<span class="wc-pos">${t.candidates.length} possible analyses</span>` : ""}
      ${ft && !r.limited       ? `<span class="wc-feats">${ft}</span>` : ""}
      ${t.grammar_issue_indices?.length ? `<span class="wc-feats">Rule finding · see grammar notes</span>` : ""}
      ${t.pos_review_recommended ? `<span class="wc-feats">POS needs extra review</span>` : ""}
      ${isVerb ? `<span class="wc-conj-hint">conjugation ↓</span>` : ""}
    </div>`;
  }
  html += `</div>`;

  if (r.limited) {
    html += `<div class="limited-note">
      ${escHtml(r.limited_note || "Linguistic morphology is unavailable for this language.")}
    </div>`;
  }

  html += `<div class="limited-note">${escHtml(r.analysis_note || "Analyses are heuristic and need review.")}</div>`;
  html += `<div id="conj-panel" class="conj-panel" style="display:none"></div>`;
  body.innerHTML = html;

  // Wire up verb chip clicks to show conjugation table
  body.querySelectorAll(".is-verb-clickable").forEach(chip => {
    chip.addEventListener("click", async () => {
      const verb = chip.dataset.verb;
      const panel = document.getElementById("conj-panel");
      if (panel.dataset.verb === verb && panel.style.display !== "none") {
        panel.style.display = "none";
        panel.dataset.verb = "";
        return;
      }
      panel.dataset.verb = verb;
      panel.style.display = "block";
      panel.innerHTML = `<div class="conj-loading">Loading conjugation for <b>${verb}</b>…</div>`;
      try {
        const d = await api("/api/conjugate", { verb });
        panel.innerHTML = renderConjugationTable(d);
      } catch (e) {
        panel.innerHTML = `<div class="conj-error">${e.message}</div>`;
      }
    });
  });
}

function renderConjugationTable(d) {
  const TENSE_ORDER = ["present","preterite","imperfect","future","conditional","subjunctive"];
  const badge = d.is_regular
    ? `<span class="conj-badge regular">regular ${d.verb_class}</span>`
    : `<span class="conj-badge irregular">irregular</span>`;

  let html = `<div class="conj-header">
    <span class="conj-inf">${d.infinitive}</span>
    <span class="conj-en">— ${d.english}</span>
    ${badge}
  </div>
  <div class="conj-forms-meta">
    <span>Gerund: <b>${d.gerund_es}</b> (${d.gerund_en})</span>
    &nbsp;·&nbsp;
    <span>Past participle: <b>${d.participle_es}</b> (${d.participle_en})</span>
  </div>`;

  if (d.notes && d.notes.length) {
    html += `<div class="conj-notes">${d.notes.map(n => `ℹ ${n}`).join("<br>")}</div>`;
  }

  html += `<div class="conj-tense-grid">`;
  for (const tk of TENSE_ORDER) {
    const t = d.tenses[tk];
    if (!t) continue;
    html += `<div class="conj-tense">
      <div class="conj-tense-name">${t.name}</div>
      <table class="conj-table">`;
    for (const f of t.forms) {
      html += `<tr>
        <td class="cp-es">${f.person_es}</td>
        <td class="cp-en">(${f.person_en})</td>
        <td class="cp-form">${f.form}</td>
      </tr>`;
    }
    html += `</table></div>`;
  }
  html += `</div>`;
  return html;
}

// ── Render: grammar notes ─────────────────────────────────────────────────────
function renderGrammar(r, lang) {
  const body = document.getElementById("body-grammar");

  if (!r) {
    body.innerHTML = `<div class="unavail-note" style="padding-top:0.85rem">Grammar checking is available for <b>English, Spanish, and French</b>. Detected: <b>${lang.toUpperCase()}</b>.</div>`;
    setBadge("badge-grammar", "");
    return;
  }
  if (!r.errors || !r.errors.length) {
    body.innerHTML = `<p class="grammar-ok" style="padding-top:0.85rem">${r.limited ? escHtml(r.limited_note || "Limited grammar coverage.") : "No issues found by the available rules."}</p><p class="unavail-note">${escHtml(r.analysis_note || "Rule findings are heuristics.")}</p>`;
    setBadge("badge-grammar", r.limited ? "Limited coverage" : "No rule findings");
    return;
  }

  setBadge("badge-grammar", `${r.errors.length} issue${r.errors.length > 1 ? "s" : ""}`, "var(--red)");
  body.innerHTML = r.errors.map(e => `
    <div class="grammar-issue ${["error", "warning", "style"].includes(e.severity) ? e.severity : "warning"}" style="margin-top:0.5rem">
      <span class="gi-sev">${escHtml(e.severity)}</span> · ${escHtml(e.category)}<br>
      ${escHtml(e.message)}
      ${e.suggestion ? `<br><span class="gi-fix">→ ${escHtml(e.suggestion)}</span>` : ""}
    </div>`).join("");
  body.innerHTML += `<p class="unavail-note">${escHtml(r.analysis_note || "Rule findings are heuristics.")}</p>`;
  if (r.limited_note) body.innerHTML += `<p class="unavail-note">${escHtml(r.limited_note)}</p>`;
  body.style.paddingTop = "0.5rem";
}

// ── Render: key vocabulary ────────────────────────────────────────────────────
function vocabularyEvidence(token, morphology, analysis) {
  if (!analysis) return token;
  const index = morphology.tokens.indexOf(token);
  const syntax = morphology.syntax;
  return {...token, occurrences:[{token_index:index,surface:token.surface,span:token.span,analysis:{...token}}],
    context:{language_selection:analysis.language_selection,
      input_evidence:analysis.input_evidence || null,
      grammar_status:analysis.grammar?.status,
      grammar_findings:(token.grammar_issue_indices || []).map(i=>analysis.grammar.errors[i]),
      failed_rules:analysis.grammar?.failed_rules || [],
      sentences:(morphology.sentences || []).filter(s=>index>=s.token_range[0] && index<s.token_range[1]),
      form_relationships:(morphology.relationships || []).filter(link=>link.token_indices.includes(index)),
      syntax:syntax ? {...syntax,arcs:syntax.arcs.filter(arc=>arc.token_index===index || arc.head_token_index===index)
        .map(arc=>({...arc,surface:morphology.tokens[arc.token_index].surface,
          head_surface:arc.head_token_index == null ? null : morphology.tokens[arc.head_token_index].surface}))} : null}};
}

function renderVocab(r, lang, analysis) {
  const body = document.getElementById("body-vocab");

  if (speechEvidenceNote() || analysis?.input_evidence) {
    setBadge('badge-vocab',(analysis?.input_evidence || _speechSource)?.method==='local_translation' ? 'Review source first' : 'Review transcript first');
    body.innerHTML = `<p class="unavail-note">Automatic vocabulary cards are unavailable for model-generated input. Add manually checked material in Flashcards.</p>`;
    return;
  }

  if (!r || r.capabilities?.pos === "default_heuristic") {
    body.innerHTML = `<div class="unavail-note" style="padding-top:0.85rem">
      Suggested vocabulary needs language-specific POS rules. Add words manually for this language.<br>
      <a href="#" id="open-fc-link">Open flashcards to add words manually →</a>
    </div>`;
    document.getElementById("open-fc-link")?.addEventListener("click", e => { e.preventDefault(); openFlashcards(); });
    return;
  }

  const words = (r.tokens || []).filter(t => CONTENT_POS.has(t.pos) && t.surface.trim().length > 1);
  if (!words.length) { body.innerHTML = `<div class="unavail-note" style="padding-top:0.85rem">No content words found.</div>`; return; }

  setBadge("badge-vocab", `${words.length} words`);
  let html = `<div class="vocab-grid">`;
  for (const t of words) {
    const info = [POS_PLAIN[t.pos] ? `${t.pos_status === "corpus_attested" ? "Corpus-attested" : "Predicted"} ${POS_PLAIN[t.pos]}` : "", feats(t.features || {})].filter(Boolean).join(" · ");
    const root = ["heuristic", "corpus_attested"].includes(t.root_status) && t.root !== t.surface ? ` · ${t.root_status === "corpus_attested" ? "lemma" : "heuristic stem"}: <i>${escHtml(t.root)}</i>` : "";
    const rootInfo = t.root !== t.surface && ["heuristic", "corpus_attested", "predicted"].includes(t.root_status)
      ? ` · ${t.root_status === "corpus_attested" ? "lemma" : t.root_status === "predicted" ? "predicted lemma" : "heuristic stem"}: ${t.root}` : "";
    html += `<div class="vocab-row">
      <span class="vr-word">${escHtml(t.surface)}</span>
      <span class="vr-info">${escHtml(info)}${root}</span>
      <button class="vr-save" type="button" data-surface="${escHtml(t.surface)}" data-info="${escHtml(info + rootInfo)}" data-lang="${escHtml(lang)}" data-analysis="${escHtml(JSON.stringify(vocabularyEvidence(t,r,analysis)))}">+ Save</button>
    </div>`;
  }
  html += `</div><p style="margin-top:0.6rem;font-size:0.75rem;color:var(--text-3)">Saves analysis to your flashcard deck, not a verified translation. Open Flashcards to practice or review; add manual cards for meanings.</p>`;
  body.innerHTML = html;

  body.querySelectorAll(".vr-save").forEach(b => {
    b.addEventListener("click", async () => {
      if (b.classList.contains("saved")) return;
      b.disabled = true;
      try {
        await api("/api/add_card", { front: b.dataset.surface, back: b.dataset.info, lang: b.dataset.lang,
          analysis: JSON.parse(b.dataset.analysis), example: r.text });
        b.textContent = "✓ Saved"; b.classList.add("saved");
      } catch (error) {
        b.textContent = "Save failed — retry"; b.title = error.message;
      } finally { b.disabled = false; }
    });
  });
}

// ── Render: advanced / morphology ─────────────────────────────────────────────
function renderFormRelationships(r) {
  const links = r.relationships || [];
  if (!links.length) return "";
  return `<details><summary>Related forms in context (${links.length})</summary><ul>${links.map(link => {
    const surfaces = link.token_indices.map(i => r.tokens[i]?.surface || "").join(" ↔ ");
    const contrasts = link.contrasts.map(c => `${c.feature}: ${c.left} → ${c.right}. ${c.interpretation}`).join(" ");
    return `<li>${escHtml(surfaces)} · ${escHtml(link.lemma)} · ${escHtml(link.status)}<br>${escHtml(contrasts || "No differing shared feature annotations.")}<br>${escHtml(link.note)}</li>`;
  }).join("")}</ul></details>`;
}

function renderSyntax(r) {
  const syntax = r.syntax;
  if (!syntax || syntax.status === "unavailable") return `<p class="adv-limited">Syntax unavailable: no local dependency model for this language.</p>`;
  return `<details><summary>Experimental sentence structure</summary><p>${escHtml(syntax.note)}</p><ul>${syntax.arcs.map(arc => {
    const word = r.tokens[arc.token_index]?.surface || "";
    const head = arc.head_token_index == null ? "ROOT" : r.tokens[arc.head_token_index]?.surface || "";
    return `<li>${arc.sentence_index == null ? "" : `Sentence ${arc.sentence_index + 1} · `}${escHtml(head)} → ${escHtml(word)}: ${escHtml(arc.relation)} (predicted)</li>`;
  }).join("")}</ul>${syntax.skipped_sentences.length ? `<p>Some sentences exceeded the parser's token limit.</p>` : ""}</details>`;
}

function renderAdvanced(r, lang) {
  const body = document.getElementById("body-advanced");
  if (!r) { body.innerHTML = `<div class="adv-limited" style="margin-top:0">No data.</div>`; return; }

  const rows = (r.tokens || []).map(t => {
    const f = Object.entries(Object.keys(t.ud_features || {}).length ? t.ud_features : (t.features || {})).map(([k,v])=>`${k}=${v}`).join(", ");
    return `<tr>
      <td><b>${escHtml(t.surface)}</b></td>
      <td style="color:var(--text-2)">${["heuristic", "corpus_attested", "predicted"].includes(t.root_status) ? escHtml(t.root) : "—"}</td>
      <td><span class="pos-pill pos-${t.pos}">${t.pos}</span></td>
      <td style="font-family:'SF Mono',monospace;font-size:0.75rem;color:var(--text-2)">${escHtml((t.morphemes||[]).join(" + "))||"—"}</td>
      <td style="color:var(--text-3);font-size:0.75rem">${escHtml(f)||"—"}</td>
      <td>${escHtml((t.subwords||[]).join(" + "))||"—"}</td>
      <td>${analysisLabel(t)}${renderPosUncertainty(t)}${renderEvidence(t)}</td>
    </tr>`;
  }).join("");

  body.innerHTML = `${r.capabilities?.tokenization === "characters" ? `<p class="adv-limited">Character-level tokenization: these are not validated words or morphemes. POS labels use limited heuristics.</p>` : ""}<div class="adv-table-wrap"><table>
    <thead><tr><th>Surface</th><th>Lemma / proposed stem</th><th>POS</th><th>Proposed morphemes</th><th>Features</th><th>BPE subwords</th><th>Analysis &amp; evidence</th></tr></thead>
    <tbody>${rows}</tbody>
  </table></div>${renderFormRelationships(r)}${renderSyntax(r)}
  <div class="adv-limited">${escHtml(r.analysis_note || "Analyses need review.")} ${r.limited ? escHtml(r.limited_note || "") : ""}</div>`;
}

// ── Render: attention heatmap ─────────────────────────────────────────────────
function renderAttention(attn) {
  const body   = document.getElementById("body-attention");
  const badge  = document.getElementById("attn-badge");
  if (!attn || !attn.matrix || attn.matrix.length === 0) {
    body.innerHTML = `<div class="adv-limited" style="margin-top:0">No attention data — translate a sentence first.</div>`;
    if (badge) badge.textContent = "Source → target alignment";
    return;
  }

  const { matrix, src_tokens, tgt_tokens, entropy } = attn;
  const T_out = matrix.length;
  const T_src = src_tokens.length;

  // Clean BPE tokens for display: strip </w> marker, join subwords
  const cleanSrc = src_tokens.map(t => t.replace(/<\/w>$/, ""));
  const cleanTgt = tgt_tokens.map(t => t.replace(/<\/w>$/, ""));

  if (badge) badge.textContent = `${T_src} source × ${T_out} output tokens`;

  // Cell size — shrink for large sequences
  const cellW = Math.max(18, Math.min(42, Math.floor(560 / T_src)));
  const cellH = Math.max(18, Math.min(36, Math.floor(400 / T_out)));
  const labelW = 90;
  const headerH = 56;
  const svgW = labelW + T_src * cellW + 4;
  const svgH = headerH + T_out * cellH + 4;

  // Build SVG heatmap
  let cells = "";
  for (let r = 0; r < T_out; r++) {
    for (let c = 0; c < T_src; c++) {
      const val = matrix[r][c];
      const alpha = Math.pow(val, 0.5);   // gamma-correct for perceptual linearity
      const x = labelW + c * cellW;
      const y = headerH + r * cellH;
      cells += `<rect x="${x}" y="${y}" width="${cellW - 1}" height="${cellH - 1}"
        fill="rgba(93,214,192,${alpha.toFixed(3)})" rx="2"/>`;
    }
  }

  // Source token labels (rotated, top)
  let srcLabels = "";
  for (let c = 0; c < T_src; c++) {
    const x = labelW + c * cellW + cellW / 2;
    srcLabels += `<text x="${x}" y="${headerH - 6}" text-anchor="end"
      transform="rotate(-45 ${x} ${headerH - 6})"
      font-size="10" fill="var(--text-3)" font-family="monospace">${escHtml(cleanSrc[c])}</text>`;
  }

  // Target token labels (left side) + confidence bar
  let tgtLabels = "";
  for (let r = 0; r < T_out; r++) {
    const y = headerH + r * cellH + cellH / 2 + 4;
    const conf = entropy ? (1 - entropy[r]) : 1;
    const barW = Math.round(conf * 20);
    const confColor = conf > 0.7 ? "var(--green)" : conf > 0.4 ? "var(--yellow)" : "var(--accent)";
    tgtLabels += `<text x="${labelW - 26}" y="${y}" text-anchor="end"
      font-size="10" fill="var(--text-2)" font-family="monospace">${escHtml(cleanTgt[r])}</text>
      <rect x="${labelW - 24}" y="${y - 8}" width="${barW}" height="7" rx="2" fill="${confColor}" opacity="0.7"/>`;
  }

  // Peak attention marker — brightest cell in each row gets a dot
  let peaks = "";
  for (let r = 0; r < T_out; r++) {
    const row = matrix[r];
    const peak = row.indexOf(Math.max(...row));
    const px = labelW + peak * cellW + cellW / 2;
    const py = headerH + r * cellH + cellH / 2;
    peaks += `<circle cx="${px}" cy="${py}" r="2.5" fill="white" opacity="0.8"/>`;
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg"
    width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">
    ${srcLabels}
    ${cells}
    ${peaks}
    ${tgtLabels}
  </svg>`;

  // Entropy legend
  const avgConf = entropy
    ? Math.round((1 - entropy.reduce((a, b) => a + b, 0) / entropy.length) * 100)
    : null;

  body.innerHTML = `
    <div class="attn-explainer">
      Each row = one output token. Each column = one source token.
      Brightness = how much attention the model paid to that source position.
      <b>White dot</b> = peak alignment. <b>Left bars</b> = attention concentration (green = more focused, yellow = more diffuse).
      ${avgConf !== null ? `<b>Avg attention concentration: ${avgConf}%</b>` : ""}
    </div>
    <div class="attn-heatmap-wrap">
      ${svg}
    </div>
    <div class="attn-note">
      Cross-attention from the last decoder layer, averaged over ${4} heads.
      Attention patterns are model diagnostics, not verified word alignments or probabilities of correctness.
    </div>`;
}

// ── Microphone (Web Speech API) ───────────────────────────────────────────────
const localAudio = document.getElementById('local-audio');
const localTranscribe = document.getElementById('local-transcribe');
const localSpeechStatus = document.getElementById('local-speech-status');
if (localTranscribe && localSpeechStatus && localAudio) {
  api('/api/speech').then(status => {
    localSpeechStatus.textContent = `${status.message}${status.checkpoint_name ? ' Checkpoint: '+status.checkpoint_name+'.' : ''} Model language: ${status.language || "unspecified"}. Format: ${status.format}`;
    localTranscribe.disabled = !status.available;
  }).catch(error => {
    localSpeechStatus.textContent = `Local speech status unavailable: ${error.message}`;
    localTranscribe.disabled = true;
  });
  localTranscribe.addEventListener('click', async () => {
    const file = localAudio.files[0];
    if (!file) { localSpeechStatus.textContent = 'Choose a 16 kHz PCM WAV first.'; return; }
    if (file.size > 2000000) { localSpeechStatus.textContent = 'WAV exceeds the 2 MB limit.'; return; }
    const initialText = document.getElementById('main-input').value;
    const initialSource = _srcCode;
    localTranscribe.disabled = true;
    localSpeechStatus.textContent = 'Running your local CTC model…';
    try {
      const response = await fetch('/api/transcribe', {method: 'POST', headers: {'Content-Type': 'audio/wav'}, body: file});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Transcription failed');
      if (!result.transcript) { localSpeechStatus.textContent = 'The local model produced no transcript; no text was replaced.'; return; }
      if (document.getElementById('main-input').value !== initialText || _srcCode !== initialSource) {
        localSpeechStatus.textContent = `Input changed while transcribing. Transcript: ${result.transcript}`;
        return;
      }
      document.getElementById('main-input').value = result.transcript;
      _speechSource = result.input_evidence || {method:'local_ctc', language:result.language};
      srcPicker.setLabel(result.language, (_allLangs.find(lang => lang.code === result.language) || {}).name || result.language);
      document.getElementById('results').classList.add('hidden');
      localSpeechStatus.textContent = `Local CTC transcript (audio: ${result.duration_s.toFixed(2)} s) using the ${result.language} model${result.checkpoint_name ? ' · '+result.checkpoint_name : ''}. Accuracy is unverified; review the text, then select Translate & Explain.`;
    } catch (error) {
      localSpeechStatus.textContent = `Local transcription unavailable: ${error.message}`;
    } finally { localTranscribe.disabled = false; }
  });
}

const micBtn    = document.getElementById("mic-btn");
const micStatus = document.getElementById("mic-status");
let _micActive  = false;
let _recognition = null;

if (!("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
  micBtn.style.opacity = "0.4";
  micBtn.title = "Voice input needs Chrome or Edge";
  micBtn.addEventListener("click", () => alert("Voice input works in Chrome or Edge. Please open LinguaForge in one of those browsers."));
} else {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  micBtn.addEventListener("click", () => {
    if (_micActive) { _recognition?.stop(); return; }
    _recognition = new SR();
    _recognition.lang = _srcCode ? (SPEECH_TAGS[_srcCode] || _srcCode) : "";
    const recognitionLanguage = _recognition.lang;
    _recognition.interimResults = false;
    _recognition.maxAlternatives = 1;
    _recognition.onstart  = () => { _micActive = true;  micBtn.classList.add("recording");    micStatus.classList.remove("hidden"); };
    _recognition.onresult = e  => {
      document.getElementById("main-input").value = e.results[0][0].transcript;
      _speechSource = {method:'browser_speech', ...(recognitionLanguage ? {language:recognitionLanguage} : {})};
      document.getElementById('results').classList.add('hidden');
    };
    let speechError = false;
    micStatus.textContent = "🔴 Listening… speak now";
    _recognition.onerror  = e  => {
      if (e.error !== "aborted") {
        speechError = true; micStatus.textContent = `Browser speech input unavailable: ${e.error}`;
        micStatus.classList.remove("hidden");
      }
    };
    _recognition.onend = () => {
      _micActive = false; micBtn.classList.remove("recording");
      if (!speechError) micStatus.classList.add("hidden");
    };
    try { _recognition.start(); } catch (error) {
      micStatus.textContent = `Browser speech input unavailable: ${error.message}`;
      micStatus.classList.remove("hidden");
    }
  });
}

// ── Flashcard drawer ──────────────────────────────────────────────────────────
function openFlashcards() {
  document.getElementById("fc-drawer").classList.remove("hidden");
  document.getElementById("fc-overlay").classList.remove("hidden");
  refreshDeck().catch(error => {
    document.getElementById("deck-stats-inline").textContent = `— unavailable: ${error.message}`;
    document.getElementById("review-area").textContent = `Cannot load cards: ${error.message}`;
  });
}
function closeFlashcards() {
  document.getElementById("fc-drawer").classList.add("hidden");
  document.getElementById("fc-overlay").classList.add("hidden");
}

document.getElementById("open-flashcards").addEventListener("click", openFlashcards);
document.getElementById("close-flashcards").addEventListener("click", closeFlashcards);
document.getElementById("fc-overlay").addEventListener("click", closeFlashcards);

document.getElementById("add-btn").addEventListener("click", async () => {
  const front = document.getElementById("add-front").value.trim();
  const back  = document.getElementById("add-back").value.trim();
  const lang  = document.getElementById("add-lang").value;
  if (!front || !back) return;
  const button = document.getElementById("add-btn"); button.disabled = true;
  try {
    await api("/api/add_card", { front, back, lang });
    document.getElementById("add-front").value = "";
    document.getElementById("add-back").value  = "";
    button.textContent = "Add";
    await refreshDeck();
  } catch (error) { button.textContent = "Save failed — retry"; button.title = error.message; }
  finally { button.disabled = false; }
});

function renderCardContext(card) {
  const context = card.analysis?.context;
  if (!context) return "";
  const findings = context.grammar_findings || [];
  const note = findings.length ? `${findings.length} heuristic rule finding(s) in this word's context` :
    context.grammar_status === "incomplete" ? "Source grammar checks incomplete" : "No linked rule findings; correctness is not established";
  const comparisons = (context.form_relationships || []).flatMap(link => link.contrasts.map(c => `${link.lemma}: ${c.feature} ${c.left} → ${c.right} (${link.status})`)).join("; ");
  const syntax = context.syntax;
  const arcs = (syntax?.arcs || []).map(arc => `${arc.head_surface || "ROOT"} → ${arc.surface}: ${arc.relation}`).join("; ");
  const sentences = (context.sentences || []).map(s => s.text).filter(Boolean).join(" · ");
  return `<div class="muted-sm">${escHtml(sentences || card.example || "")}<br>${escHtml(note)}${comparisons ? `<br>Related form annotations: ${escHtml(comparisons)}` : ""}${arcs ? `<br>Experimental syntax predictions: ${escHtml(arcs)}. Uncalibrated; grammatical links do not establish meaning.` : ""}</div>`;
}

async function refreshDeck(preserveReview = false) {
  const data = await api("/api/deck");
  const s    = data.stats;
  document.getElementById("deck-stats-inline").textContent = `— ${s.cards} cards · ${s.due} due`;

  const tbody = document.querySelector("#deck-table tbody");
  tbody.innerHTML = "";
  for (const c of data.cards) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escHtml(c.front)}</td>
      <td style="color:var(--text-2)">${escHtml(c.back)}${renderCardContext(c)}</td>
      <td style="color:var(--text-3)">${escHtml(c.lang)}</td>
      <td>${c.is_due ? `<span class="due-pill">DUE</span>` : c.srs.due_date}</td>
      <td style="color:var(--text-2)">${c.srs.ease_factor}</td>
      <td style="color:var(--text-2)">${c.srs.interval}d</td>
      <td></td>`;
    const d = document.createElement("button");
    d.className = "btn-del btn-small"; d.textContent = "✕";
    d.setAttribute("aria-label", `Delete card: ${c.front}`);
    d.addEventListener("click", async () => { await api("/api/delete_card", { card_id: c.id }); refreshDeck(); });
    tr.querySelector("td:last-child").appendChild(d);
    tbody.appendChild(tr);
  }

  if (!preserveReview) {
    _reviewQ   = data.cards.filter(c => c.is_due);
    _reviewIdx = 0;
  }
  renderReview();
  const chooser = document.getElementById("exercise-card");
  if (chooser) {
    const previous = chooser.value;
    chooser.innerHTML = "";
    for (const card of data.cards) {
      const option = document.createElement("option");
      option.value = card.id; option.textContent = `${card.front} (${card.lang})`;
      chooser.appendChild(option);
    }
    if (data.cards.some(card => card.id === previous)) chooser.value = previous;
    document.getElementById("exercise-start").disabled = !data.cards.length;
  }
}

function renderExercise(exercise) {
  const area = document.getElementById("exercise-area");
  const prompt = exercise.prompt || (exercise.scrambled || []).join(" ");
  area.innerHTML = `<p>${escHtml(prompt)}</p>${exercise.kind === "mcq"
    ? `<label for="practice-response">Your choice</label><select id="practice-response"><option value="">Choose…</option>${exercise.choices.map((choice,index)=>`<option value="${index}">${escHtml(choice)}</option>`).join("")}</select>`
    : `<label for="practice-response">Your response</label><input id="practice-response" autocomplete="off">`}
    <button class="btn-secondary" id="exercise-reveal" type="button">Show saved answer</button><p id="exercise-answer"></p>`;
  document.getElementById("exercise-reveal").addEventListener("click", () => {
    const answer = exercise.kind === "mcq" ? exercise.choices[exercise.answer_index] : exercise.answer;
    document.getElementById("exercise-answer").textContent = `Saved reference: ${answer}. Compare with your response; other valid expressions may exist.`;
  });
}

document.getElementById("exercise-start")?.addEventListener("click", async () => {
  const card_id = document.getElementById("exercise-card").value;
  if (!card_id) return;
  const button = document.getElementById("exercise-start"); button.disabled = true;
  try {
    renderExercise(await api("/api/exercise", {card_id,kind:document.getElementById("exercise-kind").value}));
  } catch (error) {
    document.getElementById("exercise-area").textContent = `Exercise unavailable: ${error.message}`;
  } finally { button.disabled = false; }
});

function renderReview() {
  const area = document.getElementById("review-area");
  if (!_reviewQ.length) {
    area.innerHTML = `<p class="muted-sm">Nothing due — add cards or come back later.</p>`;
    return;
  }
  if (_reviewIdx >= _reviewQ.length) {
    area.innerHTML = `<p class="muted-sm">Session done ✓ — ${_reviewQ.length} card${_reviewQ.length>1?"s":""} reviewed.</p>`;
    return;
  }
  const card = _reviewQ[_reviewIdx];
  area.innerHTML = `
    <div class="rc-wrap">
      <div class="rc-front">${escHtml(card.front)}</div>
      <button class="btn-secondary" id="reveal" type="button">Show answer</button>
    </div>`;
  document.getElementById("reveal").addEventListener("click", () => {
    area.innerHTML = `
      <div class="rc-wrap">
        <div class="rc-front">${escHtml(card.front)}</div>
        <div class="rc-back">${escHtml(card.back)}</div>
        ${renderCardContext(card)}
        <div class="rc-grade">
          <button data-btn="again" type="button">Again</button>
          <button data-btn="hard"  type="button">Hard</button>
          <button data-btn="good"  type="button">Good</button>
          <button data-btn="easy"  type="button">Easy</button>
        </div>
        <p class="rc-prog">${_reviewIdx + 1} / ${_reviewQ.length}</p>
      </div>`;
    area.querySelectorAll(".rc-grade button").forEach(b => {
      b.addEventListener("click", async () => {
        const buttons = area.querySelectorAll(".rc-grade button");
        buttons.forEach(button => { button.disabled = true; });
        try { await api("/api/review", { card_id: card.id, button: b.dataset.btn }); }
        catch (error) {
          buttons.forEach(button => { button.disabled = false; });
          b.textContent = "Review failed — retry"; b.title = error.message; return;
        }
        _reviewIdx++; renderReview();
        refreshDeck(true).catch(error => {
          document.getElementById("deck-stats-inline").textContent = `— refresh unavailable: ${error.message}`;
        });
      });
    });
  });
}

refreshDeck().catch(error => {
  document.getElementById("deck-stats-inline").textContent = `— unavailable: ${error.message}`;
});
