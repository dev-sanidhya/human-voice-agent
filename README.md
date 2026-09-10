# human-voice-agent

A real-time voice agent built to sound like a person and respond fast enough
to feel like one. Runs entirely on [Groq](https://console.groq.com) - STT,
LLM, and TTS all on one provider, one network hop per turn, free-tier
friendly.

## Why this stack

Compared the actively-maintained open-source voice agent frameworks before
picking one:

- **[Pipecat](https://github.com/pipecat-ai/pipecat)** (13k+ stars,
  near-daily commits) - won. Python-first, streaming-native frame pipeline,
  first-class Groq support, and a maintained local semantic turn-detection
  model (`smart-turn-v3`) that runs on-device with no extra API.
- **[LiveKit Agents](https://github.com/livekit/agents)** - excellent at
  scale (native SIP/telephony, WebRTC infra), but that's overhead this
  project doesn't need. Right call if this ever needs to handle real phone
  trunks and hundreds of concurrent calls.
- **[Vocode](https://github.com/vocodedev/vocode-core)** - lower-level,
  less actively developed than Pipecat as of the frameworks compared.

General sentiment from developer write-ups and forum discussion tracked
during this research: Pipecat is the default recommendation for anyone
building a custom pipeline who wants control without hand-rolling transport
and turn-taking themselves; LiveKit Agents is the recommendation once you
need to scale past dozens of concurrent calls or need native telephony.

## What makes this sound human, not like a demo

1. **Instant backchannel acknowledgments** (`src/human_voice_agent/backchannel.py`).
   The moment your turn ends, a short "yeah," "got it," "right," gets queued
   for TTS *before* the LLM has produced a single token of the real answer.
   Because it's 1-2 words against a full sentence, it's synthesized and
   starts playing first - filling the model's think-time instead of leaving
   dead air. It skips short utterances, direct short questions, and never
   repeats the same phrase twice in a row - the goal is a natural tic, not a
   detectable pattern.
2. **Semantic end-of-turn detection**, not a silence timer. Pipecat's
   `smart-turn-v3` (a small local ONNX model, no API call) listens to
   *how* an utterance ends, not just whether there's been silence, so
   "so I was thinking... um..." doesn't get treated as a finished turn the
   way a fixed-timeout VAD would.
3. **A system prompt written for speech, not chat** (`src/human_voice_agent/prompts.py`).
   Most voice-agent demos sound robotic because the LLM is still writing
   like a chatbot - bullet points, "Certainly!", three-paragraph answers -
   and the TTS just reads that verbatim. Fixing the text is a bigger
   naturalness win than any TTS tuning: short turns, contractions, no
   markdown, no self-narration.
4. **One provider for the whole pipeline.** STT (`whisper-large-v3-turbo`),
   LLM (`openai/gpt-oss-20b`, `reasoning_effort=low`), and TTS
   (`canopylabs/orpheus-v1-english`) all run on Groq's LPU infrastructure.
   Fewer network hops, fewer places for latency variance to hide.

## Measured latency (this environment, real Groq API, 2026-09-10)

From `scripts/smoke_test.py` against a real ~6s customer-service audio clip:

| Stage | Time |
|---|---|
| STT (`whisper-large-v3-turbo`) | ~0.55-0.65s for a 6s utterance |
| Backchannel decision | ~0ms (local, no network) |
| LLM time-to-first-token (`gpt-oss-20b`, `reasoning_effort=low`) | ~0.31s |
| LLM total generation | ~0.39s |

`reasoning_effort` matters a lot here: left at the default, `gpt-oss-20b`
spent its *entire* token budget on hidden chain-of-thought and returned an
empty reply (confirmed live - see `config.py` comments). Setting it to
`"low"` cut that to a handful of tokens and got a real, on-topic reply in
under 400ms.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

cp .env.example .env   # add your GROQ_API_KEY - free at console.groq.com/keys
```

**One-time step before audio will play**: Groq's TTS model requires accepting
model terms in the console before first use (not something this repo can do
for you - it's an account-level action). Open
https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english
and accept.

**Windows only**: `pyaudio` needs no extra setup on recent Windows (installs
from a prebuilt wheel). On macOS you'll need `brew install portaudio` first.

## Running it

```bash
python run_local.py
```

Talks through your actual microphone and speakers - no browser, no
telephony, no Docker. Ctrl+C to end the call.

For a non-interactive check that doesn't need a live mic (CI, headless
environments, or just verifying your API key/model names work before
talking to it):

```bash
python scripts/smoke_test.py samples/your_test_clip.wav
```

Prints a per-stage latency breakdown and writes the synthesized reply to
`samples/smoke_test_output.wav`.

If a model in `config.py` ever 404s (Groq's catalog changes over time), run:

```bash
python scripts/list_models.py
```

## Project layout

```
src/human_voice_agent/
  config.py       # model choices + tunables, all env-overridable
  prompts.py       # the system prompt tuned for spoken output
  backchannel.py    # acknowledgment-word selection logic
  pipeline.py        # the actual Pipecat pipeline: STT -> backchannel -> LLM -> TTS
run_local.py          # live mic/speaker entrypoint
scripts/
  smoke_test.py        # non-interactive STT->LLM->TTS check against a WAV file
  list_models.py         # lists models currently available to your Groq key
```

## Tuning

Everything in `config.py` is overridable via env var - see `.env.example`.
Notably:

- `HVA_LLM_MODEL` - swap to `openai/gpt-oss-120b` for more reasoning depth
  at the cost of extra latency per turn.
- `HVA_BACKCHANNEL_MIN_WORDS` - raise this if backchanneling still fires on
  utterances that feel too short for it.
- `HVA_VAD_STOP_SECS` - lower for snappier turn-taking, raise if the agent
  cuts people off mid-thought.

## Known limitations

- Groq TTS terms acceptance is a one-time manual step (see Setup) - by
  design this repo doesn't and shouldn't automate account-level consent.
- Mid-utterance backchanneling (a "mm-hmm" *while* the user is still
  talking, not just after) isn't implemented - Groq's STT is
  segment-based, not streaming, so there's no interim transcript to react
  to mid-turn. Would need a streaming STT provider (e.g. Deepgram) layered
  in alongside Groq to add that.
- Not telephony-ready - this is a local mic/speaker agent. Wiring it to a
  real phone line means adding a transport (Daily, LiveKit, Twilio, or
  Asterisk - see the sibling `BPO-Demo` project in this workspace for a
  working Asterisk/ARI telephony setup that could donate its telephony
  layer to this pipeline).
