"""Central config: model choices and tunables, all overridable via env vars.

Everything runs on Groq (STT + LLM + TTS) on purpose: one provider means one
network hop per round trip instead of three, which is where most of the
latency budget in a cascaded voice pipeline actually goes.
"""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

# whisper-large-v3-turbo: Groq's fastest transcription model, tuned for
# real-time use (large-v3 is more accurate but ~3x slower on Groq's own
# benchmarks - not worth it for a live conversation).
STT_MODEL = os.getenv("HVA_STT_MODEL", "whisper-large-v3-turbo")

# openai/gpt-oss-20b: smallest/fastest chat model currently on Groq. Swap to
# openai/gpt-oss-120b via HVA_LLM_MODEL if you want more reasoning depth and
# can spend the extra latency per turn. (Groq's catalog changes - run
# `scripts/list_models.py` if either model 404s.)
LLM_MODEL = os.getenv("HVA_LLM_MODEL", "openai/gpt-oss-20b")

# Orpheus TTS on Groq - fully hosted, no separate TTS provider/key needed.
TTS_MODEL = os.getenv("HVA_TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("HVA_TTS_VOICE", "hannah")

# "en" (default), "hi" (Devanagari Hindi), or "hinglish" (Latin-script
# code-switched Hindi-English). Picks the system prompt (prompts.py) and
# backchannel phrase pool (backchannel.py) for the whole conversation.
LANGUAGE = os.getenv("HVA_LANGUAGE", "en")

# sonic-turbo picked over the newer sonic-3.6 specifically for latency -
# confirmed live sonic-turbo hits first audio in ~0.38s vs sonic-3.6's
# ~0.53s, even though sonic-3.6 is Cartesia's newest model with marketed
# Hindi/Hinglish improvements. Swap via HVA_CARTESIA_MODEL if accent
# quality matters more than the ~150ms difference for your use case.
CARTESIA_MODEL = os.getenv("HVA_CARTESIA_MODEL", "sonic-turbo")

# Real native-Hindi voices from Cartesia's voice library (confirmed live via
# their /voices?language=hi endpoint - these are actual UUIDs, not
# placeholders). Siya (feminine) is the default; Kabir (masculine) is a
# ready-made alternative.
CARTESIA_VOICE_SIYA = "4459a9a5-69d6-4680-b970-e13dc51845b6"
CARTESIA_VOICE_KABIR = "cb9c954d-bcaa-43ed-82bf-aeb5e88a3cb5"

# Groq's TTS model is explicitly English-only by its own name
# (orpheus-v1-english) - Groq offers no Hindi voice. Cartesia Sonic does,
# with real streaming (no ffmpeg decode needed, unlike edge-tts) and,
# confirmed live, ~2-3x faster time-to-first-audio than edge-tts for the
# same Hindi text - see tts_fallback.py for the measured numbers. edge-tts
# stays as the universal last-resort fallback for all three languages if
# Cartesia has no key configured or a request fails.
TTS_VOICE_BY_LANGUAGE = {
    "en": ("groq", TTS_VOICE),
    "hi": ("cartesia", os.getenv("HVA_TTS_VOICE_HI", CARTESIA_VOICE_SIYA)),
    "hinglish": ("cartesia", os.getenv("HVA_TTS_VOICE_HINGLISH", CARTESIA_VOICE_SIYA)),
}

VAD_STOP_SECS = float(os.getenv("HVA_VAD_STOP_SECS", "0.2"))

# Below this many words, an instant acknowledgment ("yeah", "got it") before
# the real answer reads as noise, not naturalness - real humans don't say
# "mm-hmm" after "hi". See backchannel.py.
BACKCHANNEL_MIN_WORDS = int(os.getenv("HVA_BACKCHANNEL_MIN_WORDS", "5"))
