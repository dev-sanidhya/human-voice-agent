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

### Last-30-days repo scan (GitHub API, `pushed:>2026-08-10`, topic `voice-agent`, sorted by stars)

Repos with real recent activity, not just overall lifetime stars:

| Repo | Stars | Last push | Notes |
|---|---|---|---|
| [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent) | 2,396 | 2026-09-10 | A realtime voice *desktop companion* runtime (orb UI, task presence, wake word) - closer to a voice-controlled coding-agent pet than a phone-call agent. Not a fit here. |
| [Lynpoint/CyberVerse](https://github.com/Lynpoint/CyberVerse) | 1,639 | 2026-08-05 | Self-hosted digital-human agent platform (WebRTC + persona memory + optional video avatar). Heavier than needed for a voice-only agent. |
| [PatterAI/Patter](https://github.com/PatterAI/Patter) | 1,052 | 2026-08-25 | MIT, "give your AI agent a phone number" - owns the full stack including Twilio/Telnyx/Plivo telephony, 27+ provider integrations, Groq listed as an LLM provider. **Real alternative** if this project grows into needing an actual phone number instead of a local mic - see "Known limitations" below. Doesn't list Groq for TTS, so it wouldn't get you the single-provider setup this repo uses. |
| [xzf-thu/VoiceMem](https://github.com/xzf-thu/VoiceMem) | 1,222 | 2026-09-05 | Long-term memory infra for voice agents, not a conversational engine itself - orthogonal, not competing. |

Confirms the Pipecat pick: nothing in the last month's fresh activity beats
it for "local, Groq-native, low-latency conversational agent." Patter is the
one worth remembering if/when this needs a real inbound phone number - it
could sit in front of the same Groq-based logic this repo already has.

Reddit itself (reddit.com, old.reddit.com, and its search API) is hard-blocked
from this environment - both the sandboxed browser (browsing policy) and
direct fetch return errors, not just rate limits. Flagged rather than papered
over. X/Twitter search and Hacker News' comment API *are* reachable and did
turn up real, dated, attributed community commentary:

- **[@kwindla](https://x.com/kwindla/status/1952761378558915026)** (Pipecat's
  co-creator) posted about a new turn-detection model - independent
  confirmation that semantic turn detection (what this repo uses via
  `smart-turn-v3`) is the active frontier, not a solved problem.
- Krisp shipped a competing turn-detection model whose public roadmap
  explicitly calls out **backchannel handling** - "ignore acknowledgements
  like mm-hmm, stop for genuine interruptions" is described as a hard,
  unsolved part of the problem industry-wide, which is exactly what
  `backchannel.py`'s eligibility heuristics (word count, question detection,
  no-repeat) are trying to get right.
- **[HN, ilaksh, 2025-10-15](https://news.ycombinator.com/)**: "*What LLM do
  you guys use for fast inference for voice/phone agents? I feel like to get
  really good latency I need to 'cheat' with Cerebras, groq or SambaNova.*"
  - independent validation of this repo's core bet (Groq for inference speed)
  from outside this project.
- **HN, narrationbox, 2026-08-17** (within the last month) pushed back hard on
  cascaded (STT+LLM+TTS) architectures generally: *"The industry is very much
  moving towards one-model-does-all end to end... mostly for latency reasons
  and partially because the results... are just so much better."* This is a
  real, current counterargument to this repo's approach - noted honestly
  rather than ignored. The reason this repo still went cascaded: Groq doesn't
  offer a speech-to-speech model, and the brief was specifically "use a free
  model like Groq," which rules out the paid realtime engines (OpenAI
  Realtime, Gemini Live) that `narrationbox` is describing.

### Ran a second candidate locally: PatterAI/Patter

Cloned it, scaffolded a Groq-backed pipeline agent with `getpatter init`, and
ran its interactive test-mode REPL end to end with a real Groq call. Findings
from actually running it, not just reading the README:

- **Real bug**: `getpatter init --help` (and other CLI output) crashes on
  Windows with `UnicodeEncodeError` - it prints a `→` character into a
  cp1252-default console. Workaround: `PYTHONUTF8=1`. Worth a heads-up if
  anyone else here ever picks this up on Windows.
- **Real friction**: even pure-text "test mode" (no telephony, no STT/TTS,
  by its own docs) still eagerly validates and requires placeholder
  `TWILIO_ACCOUNT_SID`/`TWILIO_AUTH_TOKEN` and API keys for every configured
  provider before it'll construct the agent object at all.
- **Real limitation**: Patter's *built-in* test-mode conversational loop is
  hardcoded to OpenAI - a Groq-configured pipeline agent falls back to
  "`[No on_message handler or LLM loop configured]`" unless you hand-roll
  your own `on_message` callback that calls Groq directly. Did exactly that
  (see the flow below) and got a real, coherent, context-aware conversation
  through Groq - it works, it's just not the zero-code path for a
  Groq-first setup the way it is for OpenAI.
- Once wired manually, a real two-turn conversation through Patter's harness
  via Groq worked correctly and stayed on-topic ("need to check my order
  status" -> asked for the order number -> answered "what are your hours"
  correctly on the next turn), at 2.0s cold / 0.74s warm per turn (full
  non-streamed completion, not comparable directly to this repo's streamed
  TTFT numbers above).
- **Genuinely interesting for later**: Patter's own examples include an
  [`openclaw-phone-agent`](https://github.com/PatterAI/Patter/tree/main/examples/openclaw-phone-agent)
  recipe - Patter as the telephony/voice shell, an OpenClaw agent as the
  brain over a loopback OpenAI-compatible endpoint. That's directly relevant
  to this workspace (OpenClaw is already used for enrichment in the main
  Agency pipeline, and telephony infra already exists in the sibling
  `BPO-Demo` project) if this project ever needs a real inbound phone number
  instead of a local mic.

**Verdict unchanged**: Pipecat stays the right pick for *this* repo's brief -
local, Groq-native (including TTS, which Patter doesn't offer), no carrier
credentials required to run. Patter is the one to reach for the day this
needs an actual phone number.

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

**Audio works out of the box even before you do anything about Groq's TTS
terms.** Groq's Orpheus TTS (the intended production voice) requires
accepting model terms in the console before first use - confirmed by hand
that this is a login-session-gated console page, not something reachable
via API key, so it's not something this repo can or should do for you.
`ResilientTTSService` (`src/human_voice_agent/tts_fallback.py`) tries Groq
first on every utterance and, if it hits that exact wall, automatically
falls back to a free, no-signup voice (`edge-tts`) for the rest of the run
and logs a clear one-line warning when it does. Verified live end to end:
mic in, real Groq STT + LLM, fallback voice out through actual speakers,
`Bot started speaking` -> `Bot stopped speaking` with no errors.

The fallback is noticeably slower (~3s time-to-first-audio vs Groq's
~150-200ms) since it's a full-utterance, non-streamed synthesis path - it
exists for availability, not as the target experience. To get the fast,
intended voice: open
https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english,
accept the terms, and the very next run uses Groq TTS with no config change
- the fallback only triggers on that specific error.

Needs `ffmpeg` on PATH for the fallback path only (edge-tts returns MP3;
the pipeline needs raw PCM). The primary Groq path has no such dependency.

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
