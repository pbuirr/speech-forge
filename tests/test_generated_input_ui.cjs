const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'../webapp/static/app.js'),'utf8');
let hidden=true;
let callback;
const elements=new Map();
const document={getElementById(id){
  if(!elements.has(id))elements.set(id,{value:'my new text',textContent:'',
    classList:{contains(){return hidden;},add(){hidden=true;}},
    addEventListener(event,fn){callback=fn;}});
  return elements.get(id);
},querySelector(){return {textContent:'generated translation'};}};
const context=vm.createContext({document,_srcCode:'es',_srcName:'Spanish',_tgtCode:'en',_tgtName:'English',
  srcPicker:{setLabel(code,name){context._srcCode=code;context._srcName=name;hidden=true;}},
  tgtPicker:{setLabel(code,name){context._tgtCode=code;context._tgtName=name;hidden=true;}}});
vm.runInContext(source.slice(0,source.indexOf('async function api(')),context);
vm.runInContext(source.slice(source.indexOf('document.getElementById("swap-btn")'),source.indexOf('// Load language list')),context);
callback();
assert.equal(document.getElementById('main-input').value,'my new text','Hidden prior translation must not replace newly edited text');
hidden=false;
vm.runInContext('_translationEvidence={method:"local_translation",language:"en",original_input_evidence:{method:"local_ctc",model_sha256:"a".repeat(64)}}',context);
callback();
assert.equal(document.getElementById('main-input').value,'generated translation');
assert.equal(vm.runInContext('_speechSource.method',context),'local_translation');
assert.equal(vm.runInContext('_speechSource.original_input_evidence.model_sha256',context),'a'.repeat(64));
assert.match(vm.runInContext('speechEvidenceNote()',context),/Unverified model-generated/);
vm.runInContext(source.slice(source.indexOf("document.getElementById('main-input').addEventListener('input'"),source.indexOf('document.getElementById("main-input").addEventListener("keydown"')),context);
document.getElementById('main-input').value='edited generated translation';callback();
assert.equal(vm.runInContext('_speechSource.method',context),'local_translation');
document.getElementById('main-input').value='';callback();
assert.equal(vm.runInContext('_speechSource',context),null);
console.log('Generated-input UI passed: stale swap protection, translation provenance, edits and new typed workflow.');
