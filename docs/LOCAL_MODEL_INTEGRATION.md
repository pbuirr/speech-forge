# Local model integration

Latest real-world experiments and shared acoustic-data corrections are documented in
[Domain adaptation](DOMAIN_ADAPTATION.md). A real CTC pilot now exists locally but
its roughly 97% held-out WER is unsuitable for the default application. Unicode
normalization is opt-in and persisted; existing English tokenizer behavior is preserved.

Current translation update: a genuine locally trained original Spanish→English
checkpoint is installed and browser-verified. Only this direction has trained
translation weights on this machine. See [Real model validation](REAL_MODEL_VALIDATION.md)
for data separation, quality, timing, reproduction and significant limitations.
The speech availability and validation statements below remain applicable; statements
about absent translation weights describe the earlier integration inspection.

The web application can now accept a 16 kHz, 16-bit PCM WAV (mono/stereo,
0.025–30 seconds, at most 2 MB) through `/api/transcribe`. Audio is sent only to
the local server. The existing log-Mel frontend, original CNN/BiLSTM CTC model,
and greedy CTC decoder are used; no hosted inference or pretrained weights.
Stereo averaging and peak normalization match training preprocessing.

Configure your own trained model and its matching tokenizer:

```powershell
python -u -B -m webapp.server --port 8765 --asr-checkpoint checkpoints/best.pt --asr-tokenizer data/tokenizer.json --asr-language en
```

Checkpoints must use the original `speech_engine.train.save_checkpoint` format
(`model` state dictionary and `config` architecture dimensions). Loading is
CPU-only, lazy, uses `weights_only=True`, and serializes access to the model.
The tokenizer must be the one used to train that checkpoint. Matching vocabulary
size alone cannot establish matching character assignments. Existing training
normalization is English-oriented: selecting another language does not make an
English-trained model multilingual. Multilingual training/tokenizer normalization
still requires work and evaluation.

`GET /api/speech` reports file availability, configured language, format limits,
and that accuracy is unverified. File presence is not evidence of model quality
or compatibility. Missing files disable the UI transcription button; invalid
audio and incompatible checkpoint errors remain visible without replacing text.
Transcripts populate the shared text input and source language picker; users
review them, then run the existing analysis/translation/learning workflow.
Inputs edited during inference are preserved. The separate browser microphone
feature still uses Web Speech and may use an external provider.

Translation routing now restricts the existing multilingual checkpoint to its
supported ES/FR/DE → EN directions. `/api/languages` exposes exact installed
directions and includes multilingual weights in its capability flags. The
missing-model error uses the same direction inventory, including per-pair models.
File presence is only availability; successful real translation remains unverified
when no trained weights are installed.

Validation: original CTC checkpoint roundtrip/inference uses deliberately synthetic
weights to verify wiring, not speech accuracy. WAV validation, missing-model HTTP
errors, multilingual routing and capability consistency have regression tests.
Node tests verify transcript-to-text/language flow, unavailable state, errors,
and stale-result protection. Real speech WER/CER and translation quality have
not been measured for this integration because trained neural weights are absent.

After the user restarted the local server, Chrome verified the missing-model
message, format requirements and disabled transcription control. English analysis
continued to complete, and missing translation weights remained explicitly reported.
No successful real-audio transcription was claimed or browser-tested.
