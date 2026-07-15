// LinguaForge app.js — vanilla JS, no framework, no build step

// ── Utilities ─────────────────────────────────────────────────────────────────
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

const DEEP_LANGS    = new Set(["en","es","fr","de","ru","ar"]);
const CONTENT_POS   = new Set(["NOUN","PROPN","VERB","ADJ","ADV"]);
const POS_PLAIN     = { NOUN:"noun", PROPN:"proper noun", VERB:"verb", AUX:"aux. verb", ADJ:"adjective", ADV:"adverb", DET:"article", PREP:"preposition", CONJ:"conjunction", SCONJ:"conjunction", PRON:"pronoun" };
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
    b.innerHTML = `<span class="lp-item-code">${l.code}</span><span class="lp-item-name">${l.name}</span>${DEEP_LANGS.has(l.code) ? '<span class="lp-item-deep">⚙</span>' : ""}`;
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
    dd.appendChild(el("div", "lp-footer", "🔬 Deep analysis (⚙): <b>EN · ES · FR · DE · RU · AR</b><br>Contribute a tokenizer or lexicon to unlock your language"));

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
const srcPicker = LangPicker({ btnId:"src-btn", dropdownId:"src-dropdown", nameId:"src-name", allowAuto:true, onChange:(c,n)=>{ _srcCode=c; _srcName=n; } });
const tgtPicker = LangPicker({ btnId:"tgt-btn", dropdownId:"tgt-dropdown", nameId:"tgt-name", allowAuto:false, onChange:(c,n)=>{ _tgtCode=c; _tgtName=n; } });

// Swap source ↔ target languages
document.getElementById("swap-btn").addEventListener("click", () => {
  // Can't swap when source is auto-detect (nothing concrete to put in target)
  if (!_srcCode) return;
  const [prevSrcCode, prevSrcName, prevTgtCode, prevTgtName] =
        [_srcCode, _srcName, _tgtCode, _tgtName];
  srcPicker.setLabel(prevTgtCode, prevTgtName);
  tgtPicker.setLabel(prevSrcCode, prevSrcName);
  // Also swap the textarea content with the last translation if present
  const transText = document.querySelector(".trans-text");
  if (transText && transText.textContent.trim()) {
    document.getElementById("main-input").value = transText.textContent.trim();
    document.getElementById("results").classList.add("hidden");
  }
});

// Load language list from server
(async () => {
  try {
    const data = await api("/api/languages");
    _allLangs = data.languages;
  } catch { /* fall back to empty list — pickers still work, just no grouped list */ }
})();

// ── Main pipeline ─────────────────────────────────────────────────────────────
document.getElementById("go-btn").addEventListener("click", runTranslate);
document.getElementById("main-input").addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runTranslate();
});

async function runTranslate() {
  const text = document.getElementById("main-input").value.trim();
  if (!text) return;

  const goBtn     = document.getElementById("go-btn");
  const goLabel   = document.getElementById("go-label");
  const goSpinner = document.getElementById("go-spinner");
  goBtn.disabled = true;
  goLabel.textContent = "Analyzing…";
  goSpinner.classList.remove("hidden");

  try {
    const detectResult = await api("/api/detect", { text });
    const effectiveSrc = _srcCode || detectResult.language || "en";

    const [morphRes, grammarRes, transRes] = await Promise.allSettled([
      api("/api/morphology", { text, lang: effectiveSrc }),
      api("/api/grammar",    { text, lang: effectiveSrc }),
      api("/api/translate",  { text, direction: `${effectiveSrc}→${_tgtCode}`, beam: 4 }),
    ]);

    const morph = morphRes.status   === "fulfilled" ? morphRes.value   : null;
    const gram  = grammarRes.status === "fulfilled" ? grammarRes.value  : null;
    const trans = transRes.status   === "fulfilled" ? transRes.value    : { error: transRes.reason?.message || "Unavailable" };

    renderDetect(detectResult);
    renderTranslation(trans, effectiveSrc);
    renderWords(morph, effectiveSrc);
    renderGrammar(gram, effectiveSrc);
    renderVocab(morph, effectiveSrc);
    renderAdvanced(morph, effectiveSrc);
    renderAttention(trans?.attention || null);

    // Badge for Words
    if (morph) {
      const words = (morph.tokens||[]).filter(t => t.pos !== "PUNCT" && t.surface.trim());
      setBadge("badge-words", `${words.length} words`);
    }

    const results = document.getElementById("results");
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (e) {
    document.getElementById("detect-card").innerHTML = `<div class="low-conf-warn">⚠ ${e.message}</div>`;
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
function renderDetect(r) {
  const pct = Math.round(r.confidence * 100);
  const cls = r.confidence < 0.3 ? "low" : r.confidence < 0.65 ? "med" : "";

  document.getElementById("detect-card").innerHTML = `
    <div class="det-row">
      <div class="det-lang-badge">
        <span class="det-code">${r.language}</span>
        <span class="det-name">${r.name}</span>
      </div>
      <div class="det-meta">
        <span>Script: <b>${r.script}</b></span>
        ${r.dialect ? `<span class="det-dialect-tag">🗺 ${r.dialect}</span>` : ""}
        ${r.is_mixed ? `<span style="color:var(--yellow)">⚠ Mixed script</span>` : ""}
      </div>
    </div>
    <div class="conf-row">
      <span class="conf-label">Confidence</span>
      <div class="conf-track"><div class="conf-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>
    ${r.confidence < 0.3 ? `<div class="low-conf-warn" style="margin-top:0.65rem">⚠ Low confidence — try a longer sample with common words for a reliable result.</div>` : ""}`;
}

// ── Render: translation ───────────────────────────────────────────────────────
function renderTranslation(r, srcLang) {
  const card = document.getElementById("trans-card");

  if (r.error) {
    const noModel = r.error.includes("No trained model") || r.error.includes("no such endpoint");
    card.innerHTML = `
      <div class="trans-header"><span class="trans-dir-label">Translation → ${_tgtName}</span></div>
      <div class="trans-unavail">
        ${noModel
          ? `Translation <b>${srcLang} → ${_tgtCode}</b> isn't available yet for this pair.<br>
             <span style="color:var(--text-3);font-size:0.8rem">Available: Spanish → English · Train more: <code>scripts/phase7_train.py</code></span>`
          : r.error
        }
      </div>`;
    return;
  }

  const normalizedNote = r.normalized_input
    ? `<div class="trans-normalized">Normalized: <em>${escHtml(r.normalized_input)}</em></div>`
    : "";

  const literalBlock = r.literal_fallback
    ? `<div class="trans-literal">
         <span class="trans-literal-label">Literal (word-by-word):</span>
         ${escHtml(r.literal_fallback)}
       </div>`
    : "";

  const confidenceFlag = r.low_confidence
    ? `<span class="trans-low-conf" title="Coverage check flagged this translation as potentially incomplete">⚠ low confidence</span>`
    : "";

  card.innerHTML = `
    <div class="trans-header">
      <span class="trans-dir-label">Translation · ${srcLang} → ${_tgtCode}</span>
      ${confidenceFlag}
      <button class="tts-btn" id="tts-btn" type="button">🔊 Play</button>
    </div>
    ${normalizedNote}
    <div class="trans-text">${r.translation || "<em style='color:var(--text-3)'>No output</em>"}</div>
    ${literalBlock}
    <div class="trans-model-note">From-scratch Transformer NMT · Phase 7 · beam search · <a href="#" id="open-fc-note">Save to flashcards ↗</a></div>`;

  document.getElementById("open-fc-note")?.addEventListener("click", e => { e.preventDefault(); openFlashcards(); });

  const ttsBtn = document.getElementById("tts-btn");
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
    window.speechSynthesis.speak(u);
  });
}

// ── Render: word-by-word breakdown ────────────────────────────────────────────
function feats(f) {
  const p = [];
  if (f.Tense  === "Past") p.push("past");
  if (f.Tense  === "Pres") p.push("present");
  if (f.Number === "Plur") p.push("plural");
  if (f.Gender === "Fem")  p.push("feminine");
  if (f.Gender === "Masc") p.push("masculine");
  return p.join(" · ");
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
    const showRoot  = t.root && t.root !== t.surface && !r.limited;
    const isVerb    = (t.pos === "VERB" || t.pos === "AUX") && lang === "es" && !r.limited;
    const verbForm  = isVerb ? (t.root || t.surface) : null;

    html += `<div class="${chipCls}${isVerb ? ' is-verb-clickable' : ''}" ${isVerb ? `data-verb="${verbForm}" title="Click to see full conjugation table"` : ""}>
      <span class="wc-surface">${t.surface}</span>
      ${showRoot  ? `<span class="wc-root">${t.root}</span>` : ""}
      ${posLabel && !r.limited ? `<span class="wc-pos ${posCls}">${posLabel}</span>` : ""}
      ${ft && !r.limited       ? `<span class="wc-feats">${ft}</span>` : ""}
      ${isVerb ? `<span class="wc-conj-hint">conjugation ↓</span>` : ""}
    </div>`;
  }
  html += `</div>`;

  if (r.limited) {
    html += `<div class="limited-note">
      <b>Basic tokenization only — ${lang.toUpperCase()} language profile not loaded.</b><br>
      Full word-by-word analysis (roots, grammar role, tense) is available for EN · ES · FR · DE · RU · AR.
      Other languages need a dedicated tokenizer profile — contribute lexical data to unlock them.
    </div>`;
  }

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
    body.innerHTML = `<p class="grammar-ok" style="padding-top:0.85rem">✓ No issues found · score ${r.score?.toFixed(2)} · ${r.n_words} words</p>`;
    setBadge("badge-grammar", "✓ Clean", "var(--green)");
    return;
  }

  setBadge("badge-grammar", `${r.errors.length} issue${r.errors.length > 1 ? "s" : ""}`, "var(--red)");
  body.innerHTML = r.errors.map(e => `
    <div class="grammar-issue ${e.severity}" style="margin-top:0.5rem">
      <span class="gi-sev">${e.severity}</span> · ${e.category}<br>
      ${e.message}
      ${e.suggestion ? `<br><span class="gi-fix">→ ${e.suggestion}</span>` : ""}
    </div>`).join("");
  body.style.paddingTop = "0.5rem";
}

// ── Render: key vocabulary ────────────────────────────────────────────────────
function renderVocab(r, lang) {
  const body = document.getElementById("body-vocab");

  if (!r || r.limited) {
    body.innerHTML = `<div class="unavail-note" style="padding-top:0.85rem">
      Auto-extraction requires a deep-analysis profile (EN · ES · FR · DE · RU · AR).<br>
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
    const info = [POS_PLAIN[t.pos], feats(t.features || {})].filter(Boolean).join(" · ");
    const root = t.root && t.root !== t.surface ? ` · root: <i>${t.root}</i>` : "";
    html += `<div class="vocab-row">
      <span class="vr-word">${t.surface}</span>
      <span class="vr-info">${info}${root}</span>
      <button class="vr-save" type="button" data-surface="${t.surface}" data-info="${info}" data-lang="${lang}">+ Save</button>
    </div>`;
  }
  html += `</div><p style="margin-top:0.6rem;font-size:0.75rem;color:var(--text-3)">Saves to your flashcard deck — open Flashcards to edit meanings or review.</p>`;
  body.innerHTML = html;

  body.querySelectorAll(".vr-save").forEach(b => {
    b.addEventListener("click", async () => {
      if (b.classList.contains("saved")) return;
      await api("/api/add_card", { front: b.dataset.surface, back: b.dataset.info, lang: b.dataset.lang, tags: ["auto"] });
      b.textContent = "✓ Saved"; b.classList.add("saved");
    });
  });
}

// ── Render: advanced / morphology ─────────────────────────────────────────────
function renderAdvanced(r, lang) {
  const body = document.getElementById("body-advanced");
  if (!r) { body.innerHTML = `<div class="adv-limited" style="margin-top:0">No data.</div>`; return; }

  const rows = (r.tokens || []).map(t => {
    const f = Object.entries(t.features || {}).map(([k,v])=>`${k}=${v}`).join(", ");
    return `<tr>
      <td><b>${t.surface}</b></td>
      <td style="color:var(--text-2)">${t.root||"—"}</td>
      <td><span class="pos-pill pos-${t.pos}">${t.pos}</span></td>
      <td style="font-family:'SF Mono',monospace;font-size:0.75rem;color:var(--text-2)">${(t.morphemes||[]).join(" + ")||"—"}</td>
      <td style="color:var(--text-3);font-size:0.75rem">${f||"—"}</td>
    </tr>`;
  }).join("");

  body.innerHTML = `<div class="adv-table-wrap"><table>
    <thead><tr><th>Surface</th><th>Root</th><th>POS</th><th>Morphemes</th><th>Features</th></tr></thead>
    <tbody>${rows}</tbody>
  </table></div>
  ${r.limited ? `<div class="adv-limited">⚠ <b>Limited</b> — ${lang.toUpperCase()} has no deep-analysis profile. POS labels are unreliable.</div>` : ""}`;
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
      <b>White dot</b> = peak alignment. <b>Left bars</b> = confidence (green = certain, yellow = unsure).
      ${avgConf !== null ? `<b>Avg confidence: ${avgConf}%</b>` : ""}
    </div>
    <div class="attn-heatmap-wrap">
      ${svg}
    </div>
    <div class="attn-note">
      Cross-attention from the last decoder layer, averaged over ${4} heads.
      A diagonal pattern = monotone alignment (expected for close language pairs).
    </div>`;
}

// ── Microphone (Web Speech API) ───────────────────────────────────────────────
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
    _recognition.interimResults = false;
    _recognition.maxAlternatives = 1;
    _recognition.onstart  = () => { _micActive = true;  micBtn.classList.add("recording");    micStatus.classList.remove("hidden"); };
    _recognition.onresult = e  => { document.getElementById("main-input").value = e.results[0][0].transcript; };
    _recognition.onerror  = e  => { if (e.error !== "aborted") micStatus.textContent = `⚠ ${e.error}`; };
    _recognition.onend    = () => { _micActive = false; micBtn.classList.remove("recording"); micStatus.classList.add("hidden"); micStatus.textContent = "🔴 Listening… speak now"; };
    _recognition.start();
  });
}

// ── Flashcard drawer ──────────────────────────────────────────────────────────
function openFlashcards() {
  document.getElementById("fc-drawer").classList.remove("hidden");
  document.getElementById("fc-overlay").classList.remove("hidden");
  refreshDeck();
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
  await api("/api/add_card", { front, back, lang });
  document.getElementById("add-front").value = "";
  document.getElementById("add-back").value  = "";
  refreshDeck();
});

async function refreshDeck() {
  const data = await api("/api/deck");
  const s    = data.stats;
  document.getElementById("deck-stats-inline").textContent = `— ${s.cards} cards · ${s.due} due`;

  const tbody = document.querySelector("#deck-table tbody");
  tbody.innerHTML = "";
  for (const c of data.cards) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${c.front}</td>
      <td style="color:var(--text-2)">${c.back}</td>
      <td style="color:var(--text-3)">${c.lang}</td>
      <td>${c.is_due ? `<span class="due-pill">DUE</span>` : c.srs.due_date}</td>
      <td style="color:var(--text-2)">${c.srs.ease_factor}</td>
      <td style="color:var(--text-2)">${c.srs.interval}d</td>
      <td></td>`;
    const d = document.createElement("button");
    d.className = "btn-del btn-small"; d.textContent = "✕";
    d.addEventListener("click", async () => { await api("/api/delete_card", { card_id: c.id }); refreshDeck(); });
    tr.querySelector("td:last-child").appendChild(d);
    tbody.appendChild(tr);
  }

  _reviewQ   = data.cards.filter(c => c.is_due);
  _reviewIdx = 0;
  renderReview();
}

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
      <div class="rc-front">${card.front}</div>
      <button class="btn-secondary" id="reveal" type="button">Show answer</button>
    </div>`;
  document.getElementById("reveal").addEventListener("click", () => {
    area.innerHTML = `
      <div class="rc-wrap">
        <div class="rc-front">${card.front}</div>
        <div class="rc-back">${card.back}</div>
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
        await api("/api/review", { card_id: card.id, button: b.dataset.btn });
        _reviewIdx++; renderReview();
      });
    });
  });
}

refreshDeck();
