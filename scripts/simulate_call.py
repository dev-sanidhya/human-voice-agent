"""Simulates a full back-and-forth phone call using the real pipeline logic
(real Groq STT-less turn text standing in for the caller, real backchannel
selection, real Groq LLM replies, real Groq TTS for both voices) and
renders it to one downloadable audio file - a call you can actually listen
to end to end, not a single-turn clip.

The caller's lines are scripted text (standing in for what STT would have
produced from a live mic, which this environment doesn't have) - everything
downstream of that (backchannel decision, LLM reply, TTS for both speakers)
is the real pipeline code and a real Groq API call every step.

Usage:
    python scripts/simulate_call.py
"""
import asyncio
import io
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from groq import AsyncGroq

from human_voice_agent.backchannel import choose_backchannel
from human_voice_agent.config import GROQ_API_KEY, LLM_MODEL, STT_MODEL, TTS_MODEL
from human_voice_agent.prompts import SYSTEM_PROMPT
from human_voice_agent.text_normalize import normalize_speech_text

AGENT_VOICE = "hannah"
CALLER_VOICE = "troy"

# Scripted caller turns - stands in for live mic + STT. Everything after
# this (backchannel, LLM, TTS) is the real pipeline running for real.
CALLER_TURNS = [
    "Hey, hi, I'm calling about a package I ordered last week, it still hasn't shown up.",
    "It's order number 48219. I ordered it like eight days ago and the tracking hasn't moved since Tuesday.",
    "No I haven't, is that actually going to help? I already tried calling the courier directly.",
    "Okay yeah let's do that. How long does a replacement usually take to ship out?",
    "Alright that works. Thanks for sorting this out, appreciate it.",
]


def sh(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


async def synth(client: AsyncGroq, text: str, voice: str, out_path: Path):
    resp = await client.audio.speech.create(
        model=TTS_MODEL, voice=voice, input=text, response_format="wav"
    )
    out_path.write_bytes(await resp.read())


async def main():
    load_dotenv()
    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set.")
        sys.exit(1)

    client = AsyncGroq(api_key=GROQ_API_KEY)
    work = Path("samples/call_sim")
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*.wav"):
        f.unlink()

    history = []
    clips = []
    turn_idx = 0

    for caller_text in CALLER_TURNS:
        turn_idx += 1
        t0 = time.monotonic()

        # 1. Caller's line, in the caller's voice.
        caller_clip = work / f"{turn_idx:02d}a_caller.wav"
        await synth(client, caller_text, CALLER_VOICE, caller_clip)
        clips.append(caller_clip)
        print(f"[turn {turn_idx}] caller: {caller_text!r}  ({time.monotonic()-t0:.2f}s)")

        # 2. Real backchannel decision (same function the live pipeline uses).
        phrase = choose_backchannel(caller_text, has_spoken_before=turn_idx > 1)
        if phrase:
            back_clip = work / f"{turn_idx:02d}b_backchannel.wav"
            await synth(client, phrase, AGENT_VOICE, back_clip)
            clips.append(back_clip)
            print(f"          backchannel: {phrase!r}")

        # 3. Real Groq LLM reply, streamed, same system prompt and
        # reasoning_effort as the live pipeline.
        history.append({"role": "user", "content": caller_text})
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
            stream=True,
            max_completion_tokens=120,
            reasoning_effort="low",
        )
        reply_chunks = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                reply_chunks.append(delta)
        reply = normalize_speech_text("".join(reply_chunks).strip())
        history.append({"role": "assistant", "content": reply})
        print(f"          agent: {reply!r}  (total turn {time.monotonic()-t0:.2f}s)")

        # 4. Real Groq TTS for the agent's reply.
        reply_clip = work / f"{turn_idx:02d}c_agent.wav"
        await synth(client, reply, AGENT_VOICE, reply_clip)
        clips.append(reply_clip)

    # Concatenate every clip with a short natural gap into one call recording.
    gap = work / "_gap.wav"
    sh(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.4", str(gap), "-loglevel", "error"])

    concat_list = work / "_concat.txt"
    lines = []
    for c in clips:
        lines.append(f"file '{c.resolve().as_posix()}'")
        lines.append(f"file '{gap.resolve().as_posix()}'")
    concat_list.write_text("\n".join(lines), encoding="utf-8")

    out_path = Path("samples/simulated_call.wav")
    sh([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        str(out_path), "-loglevel", "error",
    ])
    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
