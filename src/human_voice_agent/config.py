"""Central config: model choices and tunables, all overridable via env vars.

Everything runs on Groq (STT + LLM + TTS) on purpose: one provider means one
network hop per round trip instead of three, which is where most of the
latency budget in a cascaded voice pipeline actually goes.
"""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

VAD_STOP_SECS = float(os.getenv("HVA_VAD_STOP_SECS", "0.2"))

# Below this many words, an instant acknowledgment ("yeah", "got it") before
# the real answer reads as noise, not naturalness - real humans don't say
# "mm-hmm" after "hi". See backchannel.py.
BACKCHANNEL_MIN_WORDS = int(os.getenv("HVA_BACKCHANNEL_MIN_WORDS", "5"))
