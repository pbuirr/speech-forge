// Exercise the actual rendering functions without a browser/network dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../webapp/static/app.js'), 'utf8');
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      innerHTML: '', style: {}, querySelectorAll: () => [], addEventListener() {},
    });
    return elements.get(id);
  },
};
const context = vm.createContext({document, setBadge() {}});
vm.runInContext(source.slice(0, source.indexOf('function LangPicker(')) +
  source.slice(source.indexOf('function feats('), source.indexOf('function renderAttention(')), context);
context.result = {
  limited: true, limited_note: 'Linguistic morphology is unavailable.',
  analysis_note: 'Analyses are heuristics.', capabilities: {pos: 'heuristic'},
  tokens: [{surface: 'hablando', root: 'hablando', root_status: 'surface',
    pos: 'VERB', features: {}, morphemes: [], subwords: ['hab', 'lando'], method: 'unavailable'}],
};
vm.runInContext('renderWords(result, "es"); renderAdvanced(result, "es"); renderVocab(result, "es")', context);
vm.runInContext('_speechSource = {method:"local_ctc"}; renderVocab(result,"es")',context);
assert.match(elements.get('body-vocab').innerHTML,/model-generated/);
assert.doesNotMatch(elements.get('body-vocab').innerHTML,/vr-save/);
vm.runInContext('_speechSource = null; renderVocab(result,"es")',context);
assert.match(elements.get('body-words').innerHTML, /is-verb-clickable/);
assert.match(elements.get('body-words').innerHTML, /Predicted verb/);
assert.match(elements.get('body-advanced').innerHTML, /BPE subwords/);
assert.match(elements.get('body-advanced').innerHTML, /hab \+ lando/);
assert.match(elements.get('body-advanced').innerHTML, /Morphology unavailable/);
assert.match(elements.get('body-vocab').innerHTML, /Predicted verb/);
assert.doesNotMatch(elements.get('body-vocab').innerHTML, /heuristic stem/);
context.result.tokens[0].surface = '<img src=x onerror=alert(1)>';
context.result.tokens[0].subwords = ['<img>'];
vm.runInContext('renderAdvanced(result, "es"); renderWords(result, "es"); renderVocab(result, "es")', context);
for (const id of ['body-advanced', 'body-words', 'body-vocab']) {
  assert.doesNotMatch(elements.get(id).innerHTML, /<img/);
  assert.match(elements.get(id).innerHTML, /&lt;img/);
}
console.log('Rendering checks passed: separate subwords, prediction labels, Spanish conjugation, vocabulary, HTML escaping.');

context.URL = URL;
context.result = {
  limited: false, analysis_note: 'Context choices are predictions.',
  capabilities: {pos: 'corpus_lookup_and_heuristic'},
  tokens: [{surface: 'saw', root: 'see', root_status: 'predicted', pos: 'VERB', pos_status: 'predicted',
    method: 'corpus_context', features: {tense: 'Past'}, ud_features: {Tense: 'Past'},
    candidates: [{lemma: 'see', pos: 'VERB', features: {Tense: 'Past'}, observations: 2,
      source: {name: '<source>', url: 'javascript:alert(1)', declared_license: 'CC0'},
      examples: [{sentence_id: '<sentence>', token_id: 2, line: 4}]}],
    selected_candidate_indices: [0]}],
};
vm.runInContext('renderWords(result, "en"); renderAdvanced(result, "en"); renderVocab(result, "en")', context);
assert.match(elements.get('body-words').innerHTML, /Predicted lemma: see/);
assert.match(elements.get('body-words').innerHTML, /past/);
assert.match(elements.get('body-advanced').innerHTML, /Context prediction/);
assert.match(elements.get('body-advanced').innerHTML, /selected by context model/);
assert.match(elements.get('body-advanced').innerHTML, /&lt;source&gt;/);
assert.doesNotMatch(elements.get('body-advanced').innerHTML, /href="javascript:/);
assert.match(elements.get('body-vocab').innerHTML, /data-analysis=/);
assert.match(elements.get('body-vocab').innerHTML, /predicted lemma: see/);
console.log('Evidence rendering checks passed: contextual labels, source escaping, unsafe link rejection, persisted card analysis.');

context.grammarResult = {limited: true, limited_note: 'Only general rules are available.',
  score: null, errors: [], analysis_note: 'No findings does not establish correctness.'};
vm.runInContext('renderGrammar(grammarResult, "ja")', context);
assert.match(elements.get('body-grammar').innerHTML, /Only general rules/);
assert.doesNotMatch(elements.get('body-grammar').innerHTML, /score|✓ Clean/);
context.grammarResult.limited_note = 'Some grammar checks failed; results are incomplete.';
vm.runInContext('renderGrammar(grammarResult, "en")', context);
assert.match(elements.get('body-grammar').innerHTML, /incomplete/);
context.grammarResult.errors = [{severity: 'error', category: '<img>', message: '<img>', suggestion: '<img>'}];
vm.runInContext('renderGrammar(grammarResult, "ja")', context);
assert.doesNotMatch(elements.get('body-grammar').innerHTML, /<img/);
assert.match(elements.get('body-grammar').innerHTML, /&lt;img&gt;/);
assert.match(elements.get('body-grammar').innerHTML, /incomplete/);
context.result.tokens[0].grammar_issue_indices = [0];
vm.runInContext('renderWords(result, "en")', context);
assert.match(elements.get('body-words').innerHTML, /Rule finding/);
console.log('Connected analysis rendering checks passed: word/rule links, limited coverage, escaped findings.');

context.result.tokens[0].pos_uncertainty = {applies_to_returned_pos: true,
  calibration: {accuracy_estimate: 0.6, development_observations: 30, outside_observed_bins: false}};
context.result.tokens[0].pos_review_recommended = true;
vm.runInContext('renderWords(result, "en"); renderAdvanced(result, "en")', context);
assert.match(elements.get('body-words').innerHTML, /POS needs extra review/);
assert.match(elements.get('body-advanced').innerHTML, /60% on development data/);
assert.match(elements.get('body-advanced').innerHTML, /correct token and sentence boundaries/);
context.result.tokens[0].pos_uncertainty.applies_to_returned_pos = false;
vm.runInContext('renderAdvanced(result, "en")', context);
assert.doesNotMatch(elements.get('body-advanced').innerHTML, /60% on development data/);
console.log('POS reliability rendering checks passed: scope, review hint, mismatched-score suppression.');

// Exercise the actual async UI flow, including unavailable translation.
(async () => {
  for (const [target,translationFails] of [['es',false],['es',true],['en',true]]) {
    const calls = [];
    const rendered = [];
    const ui = new Map();
    const pipelineContext = vm.createContext({
      _srcCode: '', _tgtCode: target,
      document: {getElementById(id) {
        if (!ui.has(id)) ui.set(id, {value: 'He bought a apple.', innerHTML: '',
          classList: {add() {}, remove() {}}, scrollIntoView() {}});
        return ui.get(id);
      }},
      async api(url, body) {
        calls.push([url, body]);
        if (url === '/api/analyze') return {lang: 'en', detection: {},
          morphology: {tokens: []}, grammar: {errors: []}};
        if (url === '/api/translate') {
          if (translationFails) throw new Error('No trained model');
          return {translation: 'example'};
        }
        throw new Error('Unexpected separate analysis request');
      },
      setBadge() {},
    });
    vm.runInContext(source.slice(0,source.indexOf('async function api(')),pipelineContext);
    for (const name of ['renderDetect', 'renderTranslation', 'renderWords', 'renderGrammar',
      'renderVocab', 'renderAdvanced', 'renderAttention']) {
      pipelineContext[name] = (...args) => rendered.push([name, args]);
    }
    vm.runInContext(source.slice(source.indexOf('async function runTranslate()'), source.indexOf('function setBadge(')), pipelineContext);
    await vm.runInContext('runTranslate()', pipelineContext);
    const identity = target === 'en';
    assert.deepEqual(calls.map(c => c[0]), identity ? ['/api/analyze'] : ['/api/analyze', '/api/translate']);
    assert.equal(calls[0][1].lang, 'auto');
    if (!identity) assert.equal(calls[1][1].direction, 'en→es');
    else {
      const result = rendered.find(([name]) => name === 'renderTranslation')[1][0];
      assert.equal(result.method,'identity');
      assert.equal(result.translation,'He bought a apple.');
      assert.equal(result.attention,null);
    }
    assert.ok(rendered.some(([name]) => name === 'renderWords'));
    if (translationFails && !identity) assert.equal(rendered.find(([name]) => name === 'renderTranslation')[1][0].error, 'No trained model');
    assert.equal(ui.get('go-btn').disabled, false);
  }
  console.log('Unified UI flow checks passed, including translation failure.');
})().catch(error => {console.error(error); process.exitCode = 1;});

vm.runInContext(source.slice(source.indexOf('function renderCardContext('), source.indexOf('async function refreshDeck(')), context);
context.cardContextFixture = {example: '<img src=x onerror=alert(1)>', analysis: {context: {grammar_status: 'checked', grammar_findings: [{status: 'heuristic'}]}}};
const contextHtml = vm.runInContext('renderCardContext(cardContextFixture)', context);
assert.ok(contextHtml.includes('&lt;img'));
assert.ok(!contextHtml.includes('<img'));
assert.ok(contextHtml.includes('heuristic rule finding'));
context.cardContextFixture.analysis.context.grammar_findings = [];
assert.ok(vm.runInContext('renderCardContext(cardContextFixture)', context).includes('correctness is not established'));
console.log('Learning context rendering checks passed.');

context.formLinkFixture = {tokens: [{surface: '<cat>'}, {surface: 'cats'}], relationships: [{token_indices: [0,1], lemma: 'cat', status: 'predicted', note: 'No semantic identity claim', contrasts: [{feature: 'Number', left: 'Sing', right: 'Plur', interpretation: 'Grammatical number'}]}]};
const formHtml = vm.runInContext('renderFormRelationships(formLinkFixture)', context);
assert.ok(formHtml.includes('&lt;cat&gt;'));
assert.ok(formHtml.includes('Sing → Plur'));
assert.ok(formHtml.includes('predicted'));
assert.ok(formHtml.includes('No semantic identity claim'));
console.log('Form relationship rendering checks passed.');

context.syntaxFixture = {tokens: [{surface: '<cats>'}, {surface: 'sleep'}], syntax: {status: 'experimental_prediction', note: 'Uncalibrated dependencies do not establish meaning', skipped_sentences: [], arcs: [{token_index:0, head_token_index:1, relation:'nsubj'}, {token_index:1, head_token_index:null, relation:'root'}]}};
const syntaxHtml = vm.runInContext('renderSyntax(syntaxFixture)', context);
assert.ok(syntaxHtml.includes('&lt;cats&gt;'));
assert.ok(syntaxHtml.includes('nsubj (predicted)'));
assert.ok(syntaxHtml.includes('Uncalibrated'));
context.syntaxFixture.syntax.status = 'unavailable';
assert.match(vm.runInContext('renderSyntax(syntaxFixture)', context), /Syntax unavailable/);
console.log('Experimental syntax rendering checks passed.');

context.cardSentenceFixture = {example: 'Whole input should remain saved', analysis: {context: {grammar_status:'checked', grammar_findings: [], sentences: [{text:'<sentence> context'}]}}};
const sentenceCardHtml = vm.runInContext('renderCardContext(cardSentenceFixture)', context);
assert.ok(sentenceCardHtml.includes('&lt;sentence&gt; context'));
assert.ok(!sentenceCardHtml.includes('Whole input should remain saved'));
console.log('Sentence context rendering checks passed.');

context.window = {};
vm.runInContext(source.slice(source.indexOf('function renderTranslation('), source.indexOf('function feats(')), context);
context.translationResult={error:'No trained model <img src=x onerror=alert(1)> Available: none yet'};
vm.runInContext('renderTranslation(translationResult,"tr")',context);
assert.doesNotMatch(elements.get('trans-card').innerHTML, /Available: Spanish/);
assert.ok(elements.get('trans-card').innerHTML.includes('&lt;img'));
context.translationResult={translation:'<script>unsafe</script>',method:'identity'};
vm.runInContext('renderTranslation(translationResult,"es")',context);
assert.ok(elements.get('trans-card').innerHTML.includes('&lt;script&gt;'));
assert.match(elements.get('trans-card').innerHTML,/Original text/);
assert.equal(elements.get('tts-btn').disabled,true);
context.translationResult={translation:'The price is lower.',literal_fallback:'<unsafe alternative>',coverage_warnings:['20 dollars','<img>']};
vm.runInContext('renderTranslation(translationResult,"es")',context);
assert.match(elements.get('trans-card').innerHTML,/Review source numbers or amounts/);
assert.match(elements.get('trans-card').innerHTML,/Unverified word-by-word alternative \(not a correction\)/);
assert.match(elements.get('trans-card').innerHTML,/&lt;unsafe alternative&gt;/);
assert.match(elements.get('trans-card').innerHTML,/20 dollars/);
assert.doesNotMatch(elements.get('trans-card').innerHTML,/<img>/);
assert.match(elements.get('trans-card').innerHTML,/&lt;img&gt;/);

context.snapshotToken={surface:'Çocuklar',span:[0,8],grammar_issue_indices:[0]};
context.snapshotMorph={tokens:[context.snapshotToken],sentences:[{text:'Çocuklar kitap okuyor.',token_range:[0,3]}],
  syntax:{status:'experimental_prediction',arcs:[{token_index:0,head_token_index:null,relation:'root'}]}};
context.snapshotAnalysis={language_selection:'explicit',grammar:{status:'limited',errors:[{message:'heuristic'}],failed_rules:[]}};
const saved = vm.runInContext('vocabularyEvidence(snapshotToken,snapshotMorph,snapshotAnalysis)',context);
assert.equal(saved.context.sentences[0].text,'Çocuklar kitap okuyor.');
assert.equal(saved.context.syntax.arcs[0].surface,'Çocuklar');
assert.equal(saved.context.grammar_findings[0].message,'heuristic');
assert.doesNotThrow(()=>JSON.stringify(saved));

vm.runInContext(source.slice(source.indexOf('function renderExercise('),source.indexOf('document.getElementById("exercise-start")?.addEventListener')),context);
context.exerciseFixture={kind:'mcq',prompt:'<img src=x>',choices:['<script>','kitap'],answer_index:1};
vm.runInContext('renderExercise(exerciseFixture)',context);
assert.ok(elements.get('exercise-area').innerHTML.includes('&lt;img'));
assert.ok(elements.get('exercise-area').innerHTML.includes('&lt;script&gt;'));
console.log('Application usability checks passed: honest availability, escaped translation/exercises, shared saved context, unavailable browser audio.');

const microphoneElements = new Map();
const micContext = vm.createContext({
  _srcCode:'tr', SPEECH_TAGS:{tr:'tr-TR'},
  window:{SpeechRecognition:class {start(){this.onstart();this.onerror({error:'not-allowed'});this.onend();}}},
  document:{getElementById(id){
    if(!microphoneElements.has(id)) microphoneElements.set(id,{
      textContent:'',classes:new Set(['hidden']),
      classList:{add(name){microphoneElements.get(id).classes.add(name);},remove(name){microphoneElements.get(id).classes.delete(name);}},
      addEventListener(type,callback){this.click=callback;}
    });
    return microphoneElements.get(id);
  }}
});
vm.runInContext(source.slice(source.indexOf('const micBtn'),source.indexOf('// ── Flashcard drawer')),micContext);
microphoneElements.get('mic-btn').click();
assert.match(microphoneElements.get('mic-status').textContent,/not-allowed/);
assert.equal(microphoneElements.get('mic-status').classes.has('hidden'),false);
console.log('Speech error sequence check passed: permission error survives recognition end.');

vm.runInContext(source.slice(source.indexOf('function renderDetect('),source.indexOf('// ── Render: translation')),context);
context.detection={language:'gd',name:'Scottish Gaelic',script:'Latin',confidence:.95};
vm.runInContext('renderDetect(detection,"en",{method:"local_ctc",language:"en"})',context);
assert.match(elements.get('detect-card').innerHTML,/Transcript\/text hypothesis/);
assert.match(elements.get('detect-card').innerHTML,/not the original audio/);
assert.doesNotMatch(elements.get('detect-card').innerHTML,/95%/);
vm.runInContext('renderDetect(detection,"gd",null)',context);
assert.match(elements.get('detect-card').innerHTML,/95%/);
console.log('Language identification UI distinguishes transcript hypotheses from acoustic evidence.');
