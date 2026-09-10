"""Non-interactive pipeline smoke test.

Exercises the real Groq STT -> LLM -> TTS chain end to end against a sample
WAV file (stand-in for "user finished speaking") and prints a latency
breakdown per stage, plus the total time-to-first-audio-byte - the number
that actually determines how the agent *feels* to talk to.

This exists because a live microphone isn't available in every environment
this repo gets tested from (CI, sandboxes, etc). `run_local.py` is the real
thing - a live mic/speaker conversation - run that when you have a
microphone and want to hear it for real.

Usage:
    python scripts/smoke_test.py samples/smoke_test_input.wav
"""
import asyncio
import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from groq import AsyncGroq

from human_voice_agent.backchannel import choose_backchannel
from human_voice_agent.config import GROQ_API_KEY, LLM_MODEL, STT_MODEL, TTS_MODEL, TTS_VOICE
from human_voice_agent.prompts import SYSTEM_PROMPT
from human_voice_agent.text_normalize import normalize_speech_text


async def main():
    load_dotenv()
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set - copy .env.example to .env and add your key.")
        sys.exit(1)

    wav_path = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/smoke_test_input.wav")
    if not wav_path.exists():
        print(f"No input file at {wav_path}")
        sys.exit(1)

    client = AsyncGroq(api_key=GROQ_API_KEY)
    t_start = time.monotonic()

    # 1. STT
    with open(wav_path, "rb") as f:
        stt_resp = await client.audio.transcriptions.create(
            file=(wav_path.name, f.read()),
            model=STT_MODEL,
        )
    t_stt = time.monotonic()
    transcript = stt_resp.text.strip()
    print(f"[STT  {t_stt - t_start:6.3f}s] whisper heard: {transcript!r}")

    if not transcript:
        print("Empty transcript - can't continue the smoke test past STT.")
        return

    # 2. Backchannel decision (instant - no network call, this is the whole point)
    filler = choose_backchannel(transcript, has_spoken_before=True)
    t_backchannel = time.monotonic()
    print(f"[BACK {t_backchannel - t_stt:6.3f}s] backchannel: {filler!r}")

    # 3. LLM (streamed, time to first token is what matters for perceived latency)
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        stream=True,
        max_completion_tokens=120,
        reasoning_effort="low",
    )
    first_token_time = None
    reply_chunks = []
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.monotonic()
            reply_chunks.append(delta)
    t_llm_done = time.monotonic()
    reply = normalize_speech_text("".join(reply_chunks).strip())
    ttft = (first_token_time - t_backchannel) if first_token_time else float("nan")
    print(f"[LLM  ttft={ttft:6.3f}s total={t_llm_done - t_backchannel:6.3f}s] reply: {reply!r}")

    # 4. TTS on the real reply (the backchannel clip would normally already be
    # playing by now - see backchannel.py / pipeline.py for how that overlaps
    # with steps 3-4 in the live pipeline instead of running after them).
    try:
        tts_resp = await client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply,
            response_format="wav",
        )
        t_tts = time.monotonic()
        out_path = Path("samples/smoke_test_output.wav")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_bytes(await tts_resp.read())
        print(f"[TTS  {t_tts - t_llm_done:6.3f}s] wrote {out_path} ({out_path.stat().st_size} bytes)")
    except Exception as exc:
        t_tts = time.monotonic()
        print(f"[TTS  SKIPPED] {exc}")
        print(
            "  -> Groq TTS models require a one-time terms acceptance in the "
            "console before first use: open the URL in the error above, or "
            "https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english "
            "and click accept. This is a one-time account action, not something "
            "this script can or should do for you."
        )

    total = t_tts - t_start
    print(f"\nTotal STT->LLM->TTS: {total:.3f}s (excludes network to your mic/speakers)")
    print(f"Time-to-first-spoken-word if backchannel fires: ~{t_backchannel - t_stt:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
