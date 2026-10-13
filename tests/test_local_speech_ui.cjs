const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../webapp/static/app.js'), 'utf8');
async function fixture(available, response, changeInput = false) {
  const elements = new Map();
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: '', value: 'original', files: [{size: 100}],
      classList: {add() {}}, addEventListener(event, callback) {this.click = callback;}});
    return elements.get(id);
  }};
  let requests = 0;
  const context = vm.createContext({document, _srcCode: '', _allLangs: [{code:'tr', name:'Turkish'}],
    api: async () => ({available, language:response.language || 'en', message: 'Local status', format:'PCM WAV'}),
    srcPicker: {setLabel(code, name) {context._srcCode = code; document.getElementById('src-name').textContent = name;}},
    fetch: async (url, options) => {
      requests++;
      assert.equal(url, '/api/transcribe');
      assert.equal(options.headers['Content-Type'], 'audio/wav');
      if (changeInput) document.getElementById('main-input').value = 'edited';
      return {ok: !response.error, json: async () => response};
    }});
  vm.runInContext(source.slice(source.indexOf('const localAudio'), source.indexOf('const micBtn')), context);
  await Promise.resolve();
  return {elements, context, requests: () => requests};
}
(async () => {
  let test = await fixture(false, {});
  assert.equal(test.elements.get('local-transcribe').disabled, true);
  test = await fixture(true, {transcript:'Çocuklar kitap okuyor.', language:'tr', duration_s:1.5});
  assert.match(test.elements.get('local-speech-status').textContent,/Model language: tr/);
  await test.elements.get('local-transcribe').click();
  assert.equal(test.elements.get('main-input').value, 'Çocuklar kitap okuyor.');
  assert.equal(test.context._srcCode, 'tr');
  assert.equal(test.context._speechSource.method, 'local_ctc');
  assert.match(test.elements.get('local-speech-status').textContent, /unverified/);
  assert.match(test.elements.get('local-speech-status').textContent,/using the tr model/);
  test = await fixture(true, {error:'Invalid WAV'});
  await test.elements.get('local-transcribe').click();
  assert.equal(test.elements.get('main-input').value, 'original');
  assert.equal(test.elements.get('local-transcribe').disabled, false);
  test = await fixture(true, {transcript:'old transcript', language:'en', duration_s:1}, true);
  await test.elements.get('local-transcribe').click();
  assert.equal(test.elements.get('main-input').value, 'edited');
  assert.match(test.elements.get('local-speech-status').textContent, /old transcript/);
  console.log('Local speech UI checks passed: availability, shared language/input, errors, and stale-result protection.');
})().catch(error => {console.error(error); process.exitCode = 1;});
