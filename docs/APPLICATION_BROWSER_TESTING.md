# Live application checks — 2026-10-11

Latest translation reliability pass (after user restart): actual Chrome interactions
verified numeric decoding now retains2026 and2027; the literal alternative is
explicitly unverified/not a correction. Negated conference statement still loses
negation, Pedro becomes “ped” without a review badge, and public-health news still
produces “new media” and spurious words without a badge. Explicit Spanish selection
survives a mistaken Polish detection for the name case. Local CTC remains disabled.
See [TRANSLATION_RELIABILITY.md](TRANSLATION_RELIABILITY.md) for exact inputs/outputs
and model/data measurements. No new speech transcription or flashcard write.


Latest model-replacement pass: after user restart, interacted with Spanish→English
translation in Chrome. Confirmed the everyday/news-trained model limitation label;
`2027` mistranslated as `27` gets a separate numeric review warning and literal
fallback; a subsequent station question clears stale warnings. That output remains
awkward (`Where is the station of train?`). A public-health sentence produced wrong
added words without a heuristic badge, so confidence gaps remain. Explicit Turkish
analysis completes while translation lists only es→en as installed; automatic
detection remains wrong on the short Turkish example. Default local CTC remains
disabled; no browser audio recognition is claimed. No cards changed. Full inputs,
outputs and measured limitations are in DOMAIN_ADAPTATION.md.

Additional integration pass: after the user restarted the server, Chrome verified
that `/api/speech` supplies the explicit missing-checkpoint/tokenizer message and
PCM WAV requirements, and the transcription button stays disabled. The earlier
stale-backend `not found` message is resolved. English text analysis and its
missing-translation-model response were also exercised successfully after restart.
Successful audio upload/inference remains unverified in the browser because no
trained CTC checkpoint is installed; synthetic inference and HTTP error tests
verify wiring only.
The short Turkish input `Çocuklar kitap okuyor.` was incorrectly auto-detected
as French (uncalibrated score 50%). Explicit Turkish selection routed analysis
to `tr` and retained the actual missing TR→EN translation error. Detection remains
an unresolved accuracy gap. The UI now labels the detected language and explicitly
warns when it disagrees with the selected analysis language. This warning was
verified visible in Chrome after a frontend reload and rerunning explicit Turkish.

The running application was genuinely exercised through Chrome at
`http://127.0.0.1:8765`, with an additional in-app-browser page load. Tool-launched
servers reported startup but were unreachable; the user launched and restarted
the server in their terminal. Browser interactions used the real frontend and
HTTP backend, with no mocked responses or substituted model outputs.

| Workflow | Observed result |
| --- | --- |
| Language search, source/target selection, swap | English, Turkish, Spanish and Japanese selected successfully. Target changes and swaps now hide previous results; verified live. |
| English analysis/grammar | Automatic detection, POS/morphology evidence and sentence-indexed syntax displayed. `She go to school` triggered the agreement rule. Findings retain their heuristic warning. |
| Turkish analysis | `Çocuklar kitap okuyor. Evlerde insanlar yaşıyor.` displayed corpus evidence, ambiguous alternatives and uncalibrated POS/syntax. Unknown `Evlerde` abstained from lemma extraction. Grammar coverage was limited. |
| Parser limitations | `The children read books.` incorrectly chose `children` as root and attached `read` as `acl`. The warning was visible, but the prediction is still wrong. An English input over128tokens displayed the parser-limit warning without a visible UI hang. |
| Spanish | Missing ES→EN model handled. Fallback POS labels and unavailable morphology displayed. Clicking `exploró` returned `unknown verb` from the finite conjugation inventory. Successful conjugation was not tested. |
| Japanese | Character-level tokens, unavailable morphology/syntax, and limited grammar displayed. Character tokenization is now explicitly distinguished from validated words/morphemes. |
| Translation | EN→EN, TR→EN, ES→EN and JA→EN missing-model paths exercised. Backend reported no installed model. Successful NMT output/attention quality could not be verified. |
| Flashcards | Initially empty deck. A Turkish vocabulary save failed with a Windows charmap encoding exception and left a zero-byte file. After the fix/restart, manual Turkish cards and saved corpus-analysis cards persisted across page reloads. Original sentence and predicted syntax context appeared on the saved card. |
| Review | Answer reveal and `Good` scheduling worked. Due counts originally stayed stale; after the fix they refreshed from1to0without restarting the review session. Persistence was verified after reload. |
| Exercises | Added browser access to the existing endpoint. Fill, multiple choice and sentence reconstruction exercised. Typed responses/choice selection and reference reveal worked. No-distractor error was visible; adding a distinct same-language card enabled multiple choice. Contextual cloze preserved Turkish source text. Missing sentence on a manual card reported unavailable reconstruction. |
| Speech | Browser microphone controls and explanatory text inspected. The control uses Web Speech API, not the project's CTC model. No microphone recording, external speech-service transcription, audible playback or local ASR accuracy test was performed. No successful translation existed from which to reach the playback button. |

## Fixes and regression evidence

- Decks use explicit UTF-8 bytes and atomic temporary-file replacement; failed
  replacement preserves the previous deck. Legacy Windows-encoded decks can be
  read and migrated on save. The failed empty test deck was backed up and restored;
  no existing cards were present before the test.
- Vocabulary/manual save errors are visible and preserve input for retry. Review
  submissions disable duplicate clicks, handle errors, and refresh deck statistics.
  Deck-read failures now return JSON errors instead of dropping the HTTP connection.
- Removed the false claim that Spanish translation was available. Translation text
  and errors are HTML-escaped. Detection scores are labelled uncalibrated; explicit
  analysis language is shown separately. Syntax absence and character-tokenization
  limitations are visible. Picker descriptions no longer claim affix segmentation.
- Manual card languages use the backend language list, including Turkish. Individual
  vocabulary saves retain shared sentence, syntax, form-comparison and grammar
  evidence. Practice uses existing exercise generation; reference reveal is not a
  general multilingual grammatical-answer judge.
- Browser speech failures survive the recognition-end event. Missing synthesis is
  handled and speech controls identify browser services rather than project models.
  These failure paths were tested with simulated browser events, not live audio.
- VAD HTTP input now uses the supplied sample rate instead of always16000Hz and
  rejects nonfinite, non-mono or out-of-range PCM. This was verified through the real
  handler with in-memory transport, not a browser audio-upload workflow. There is
  no VAD upload or local CTC transcription workflow in the current interface.

128 Python tests pass; Node rendering, async flow, speech-error sequence,
translation escaping, shared card context and exercise rendering checks pass.
JavaScript syntax and `git diff --check` pass. Desktop layout was inspected via
screenshots; mobile/device layouts, controlled application-load benchmarks,
successful NMT/attention, speech accuracy and learner outcomes remain unverified.

The three synthetic test cards were archived outside the repository and cleaned
up, preserving any nonmatching cards. The user's server remains running. A live
Turkish cloze screenshot is saved in this chat's `outputs/linguaforge-live-learning.jpg`.
No new dependency, pretrained model, commit or push.

## 2026-10-12 real local CTC follow-up

Chrome file upload and explicitly configured asr_expanded checkpoint were actually
used on three preselected held-out PCM WAVs. References and file hashes are in
data/speech_real/browser_checks/references.json. Outputs:

- “i hope miss milner you pass this evening at home” → “i ho mis noli yopospas y min io hon”
- “yes how many” → “as oman”
- “then don't you interfere with me sir” → “tan dotelinef fir wit ma ser”

The first transcript was explicitly passed to analysis. English source selection,
original-text display, no-correction note, uncalibrated detection disagreement and
review prompt were visible. Subsequent transcription cleared prior analysis.
All three inference requests completed; quality is poor. These checks supersede
the earlier absence of a local CTC upload interface, not the other unverified items.

## 2026-10-12 selected speech-quality package

After the user restarted the server with data/models/asr_en_quality.pt and its
matching tokenizer, all three preselected WAVs were uploaded and transcribed
through Chrome at localhost:8765. Actual outputs matched direct PCM inference:

- “i hope miss milner you pass this evening at home” → “i he mis noli you postas y min a hon”
- “yes how many” → “as homan”
- “then don't you interfere with me sir” → “then do  ine tr wit measer”

The last output contains two spaces after “do”, confirmed from the textbox value.
Each request completed and displayed the review-before-analysis prompt. The
first transcript was explicitly analyzed: English source remained selected,
original text was displayed with the no-translation/no-correction explanation,
and Welsh detection disagreement was marked uncalibrated (65%). The second
transcript cleared the previous analysis. No flashcards were saved or grammatical
corrections imposed. The third recording loses negation; aggregate evaluation
improvement does not make these transcripts reliable.

These are local WAV checks, not microphone, streaming, multilingual or load tests.

## 2026-10-12 speech provenance and unseen speaker

Installed quality checkpoint unchanged. Actually uploaded the first fixed challenge
recording (new speaker1688) as data/speech_challenge/browser-new-speaker.wav.
Reference “personally as you call it and all”; actual transcript “pasmy as e cos
and do”,3.58s. Analysis keeps English selected, labels detection uncalibrated and
shows original transcript without correction. New speech provenance note remains
above analysis/translation. Expanded vocabulary has no automatic Save buttons and
shows review requirement. English→Spanish unavailable result also retains warning.
Editing transcript retains speech provenance; clearing the input then typing
“He bought an apple.” removes speech warning, updates status and restores ordinary
vocabulary suggestions. No cards saved. Fresh transcription clears previous results.
These checks concern installed weights and new frontend controls; experimental
feature-masking/control candidates were evaluated offline, not through this server.

## 2026-10-12 diverse supervision and connected evidence

Actually tested Spanish→English “El gato duerme.” with existing local NMT: output
“The cat doubt.” remains incorrect. Swapping copies visible model output and
preserves its predicted provenance; reanalysis withholds automatic vocabulary
cards and shows reverse translation unavailable. Fixed and retested swap ordering
when language pickers hide results. Fresh typed text is not overwritten by a
hidden stale translation. No cards saved or browser-provider audio requested.

After requested restart, uploaded data/speech_dialects/browser_checks/heldout-1.wav
(Irish English). Actual transcript: “ifls ellyous of nity wep ens by to wa mor ofos
ansis whilt cos the compecon ilasion obo thit the tecter on the dofender”, displayed
7.77s. This exactly matches direct inference with asr_en_quality.pt, not the new
asr_en_diversity.pt. Do not claim new model browser verification from this run.
Analysis keeps configured English, labels detection uncalibrated, shows original
text with no translation/correction, preserves speech review warning and withholds
automatic learning cards. Requested restart with explicit diversity filenames.

### Correct diversity checkpoint restart verified

After user restarted with explicit asr_en_diversity filenames, actually uploaded
all three fixed held-out WAVs through Chrome. Results exactly match direct packaged
PCM inference in results/asr_diversity_pcm_runtime.json:

- Irish: “iflsaliuso nity wepen spoy to war mor opos insids wit cals the compeson ilation obot thit the tecter and the defender”
- Northern: “the an to al primmy ran r seved i thi tbii a fere of supet impivision of han no bet of bous”
- Scottish: “it the resit the secan bo falspon the gre the firs ther slis a bowat an ap norly wie el a grand”

Displayed 7.77/9.47/7.00s are audio durations, not measured inference latency.
Analyzed first and third; preserved configured English and speech-derived warning,
same-language output explicitly original text without translation or correction.
Scottish transcript incorrectly detected as Scottish Gaelic (uncalibrated 70%);
selection disagreement shown and analysis continues in English. Expanded vocabulary
shows automatic model-generated cards unavailable, no Save buttons. New transcription
clears old analysis. No cards saved, browser provider invoked, or console errors.
These severely distorted transcripts remain experimental. Fixed recordings were
selected before candidate test evaluation, not picked for successful output.
Screenshot speech-diversity-browser.png now shows this new checkpoint result.

### 2026-10-12 frontend uncertainty regression

Actually uploaded and analyzed the same Scottish WAV through Chrome against the
installed diversity model. The distorted transcript remains unchanged, but the
new UI labels its Scottish Gaelic detection as a transcript/text hypothesis,
withholds the generated-input percentage, and states that this cannot verify the
audio language. English analysis and the automatic-learning block are visible.
Proof: speech-language-hypothesis.png in the task's outputs directory.
The server still has the older Python process loaded; the new backend routing,
checkpoint-hash provenance and newly trained candidate are **not browser verified**.
User paused for external AI review before candidate independent evaluation/install.
