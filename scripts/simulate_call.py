"""Simulates a full back-and-forth phone call using the real pipeline logic
and renders it to one downloadable audio file, with each turn's silence gap
set to the *actual measured latency* for that turn - not a fake fixed pause.

Closed loop, not just scripted text played through the agent side: each
caller line is synthesized to audio, then re-transcribed through real Groq
STT (timed), exactly like a live mic would produce it. Backchannel decision
and TTS synthesis run concurrently with the LLM call (asyncio.gather),
matching how the real pipeline overlaps them - the gap before the agent's
reply plays is however long that real overlapped race actually took.

Usage:
    python scripts/simulate_call.py                  # English
    python scripts/simulate_call.py --language hi        # Hindi
    python scripts/simulate_call.py --language hinglish   # Hinglish
"""
import argparse
import asyncio
import io
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from groq import AsyncGroq

from human_voice_agent.backchannel import choose_backchannel, get_backchannel_phrases
from human_voice_agent.config import (
    CARTESIA_API_KEY,
    CARTESIA_MODEL,
    CARTESIA_VOICE_KABIR,
    GROQ_API_KEY,
    LLM_MODEL,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE_BY_LANGUAGE,
)
from human_voice_agent.prompts import get_system_prompt
from human_voice_agent.text_normalize import normalize_speech_text
from human_voice_agent.tts_fallback import ResilientTTSService

CALLER_VOICE_BY_PROVIDER = {
    "groq": "troy",
    "cartesia": CARTESIA_VOICE_KABIR,  # male, distinct from the agent's Siya voice
    "edge": "hi-IN-MadhurNeural",
}

CALLER_TURNS = {
    "en": [
        "Hey, hi, I'm calling about a package I ordered last week, it still hasn't shown up.",
        "It's order number 48219. I ordered it like eight days ago and the tracking hasn't moved since Tuesday.",
        "No I haven't, is that actually going to help? I already tried calling the courier directly.",
        "Okay yeah let's do that. How long does a replacement usually take to ship out?",
        "Alright that works. Thanks for sorting this out, appreciate it.",
    ],
    "hi": [
        "नमस्ते, मैं पिछले हफ़्ते मंगाए गए एक पैकेज के बारे में कॉल कर रहा हूँ, वो अभी तक नहीं आया।",
        "ऑर्डर नंबर 48219 है। मैंने आठ दिन पहले ऑर्डर किया था और ट्रैकिंग मंगलवार से रुकी हुई है।",
        "नहीं देखा, क्या इससे सच में मदद मिलेगी? मैंने कूरियर वालों को खुद भी कॉल किया था।",
        "ठीक है चलिए करते हैं। रिप्लेसमेंट भेजने में आमतौर पर कितना समय लगता है?",
        "ठीक है, बढ़िया। इसे सुलझाने के लिए शुक्रिया।",
    ],
    "hinglish": [
        "Hey, mai last week order kiya hua ek package ke baare mein call kar raha hoon, abhi tak aaya nahi.",
        "Order number 48219 hai. Maine eight din pehle order kiya tha aur tracking Tuesday se stuck hai.",
        "Nahi dekha, kya isse actually help milegi? Maine courier wale ko khud bhi call kiya tha.",
        "Okay chalo karte hai. Replacement bhejne mein usually kitna time lagta hai?",
        "Theek hai, badhiya. Isse sort karne ke liye thanks.",
    ],
}


def sh(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def make_tts_service(provider: str, voice: str) -> ResilientTTSService:
    tts = ResilientTTSService(
        groq_api_key=GROQ_API_KEY,
        groq_model=TTS_MODEL,
        groq_voice=voice if provider == "groq" else "hannah",
        edge_voice=voice if provider == "edge" else "en-US-AndrewNeural",
        cartesia_api_key=CARTESIA_API_KEY,
        cartesia_model=CARTESIA_MODEL,
        cartesia_voice_id=voice if provider == "cartesia" else None,
        cartesia_language="hi" if provider == "cartesia" else "en",
        provider=provider,
        sample_rate=24000,
    )
    tts._sample_rate = 24000  # only finalizes on a real StartFrame; pin it for standalone use
    return tts


async def synth(tts: ResilientTTSService, text: str, out_path: Path) -> tuple[float, float]:
    """Synthesizes via the real shipped ResilientTTSService (not a
    standalone reimplementation - an earlier version of this script had its
    own separate edge-tts synthesis code that quietly drifted out of sync
    with a real latency fix made in tts_fallback.py, so the numbers it
    reported were stale. Using the real service directly means this script
    can never again report numbers the shipped code doesn't actually
    produce.

    Takes an existing `tts` instance rather than building a fresh one per
    call - the backchannel cache (see tts_fallback.py's warm_cache) only
    helps if the same instance, with the same warmed cache, is reused
    across the whole call instead of being thrown away and rebuilt (with an
    empty cache) on every single utterance.

    Returns (time_to_first_frame, total_time).
    """
    t0 = time.monotonic()
    first_frame_t = None
    pcm_chunks = []
    async for frame in tts.run_tts(text, context_id="sim"):
        if hasattr(frame, "audio"):
            if first_frame_t is None:
                first_frame_t = time.monotonic() - t0
            pcm_chunks.append(frame.audio)
    total_t = time.monotonic() - t0

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        for chunk in pcm_chunks:
            wf.writeframes(chunk)

    return first_frame_t or total_t, total_t


def add_background_noise(in_path: Path, out_path: Path, level: float = 0.035):
    """Mixes a faint, continuous room-tone noise bed under the whole call.

    This does not change the actual measured latency numbers anywhere in
    this file - it's a UX technique, not a latency fix. Dead digital
    silence during a real gap reads as "did the call drop?"; the same gap
    with a faint continuous texture under it reads as "the line is still
    open, someone's about to talk." Real call center and IVR systems do
    this on purpose. Low amplitude brown noise (soft, rumbly, not hissy
    like white/pink noise) approximates a quiet room/office line tone.
    """
    with wave.open(str(in_path), "rb") as wf:
        duration = wf.getnframes() / wf.getframerate()

    noise_path = in_path.with_name(in_path.stem + "_noise.wav")
    sh([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anoisesrc=color=brown:sample_rate=24000:amplitude=1.0",
        "-t", f"{duration:.3f}",
        str(noise_path), "-loglevel", "error",
    ])
    sh([
        "ffmpeg", "-y",
        "-i", str(in_path), "-i", str(noise_path),
        "-filter_complex",
        f"[1:a]volume={level}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0",
        "-ar", "24000", "-ac", "1",
        str(out_path), "-loglevel", "error",
    ])
    noise_path.unlink()


def make_silence(seconds: float, out_path: Path):
    seconds = max(0.05, seconds)
    sh(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
            "-t", f"{seconds:.3f}", str(out_path), "-loglevel", "error",
        ]
    )


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", choices=["en", "hi", "hinglish"])
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("GROQ_API_KEY not set.")
        sys.exit(1)

    caller_turns = CALLER_TURNS[args.language]
    system_prompt = get_system_prompt(args.language)
    agent_provider, agent_voice = TTS_VOICE_BY_LANGUAGE[args.language]
    caller_voice = CALLER_VOICE_BY_PROVIDER[agent_provider]

    client = AsyncGroq(api_key=GROQ_API_KEY)
    work = Path(f"samples/call_sim_{args.language}")
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        f.unlink()

    # One persistent TTS instance per speaker for the whole call, not a
    # fresh one per utterance - the agent's cache (see tts_fallback.py)
    # only pays off if it's actually reused across turns.
    agent_tts = make_tts_service(agent_provider, agent_voice)
    caller_tts = make_tts_service(agent_provider, caller_voice)
    t_warm = time.monotonic()
    await agent_tts.warm_cache(get_backchannel_phrases(args.language))
    print(f"[cache] warmed {len(agent_tts._cache)} backchannel phrases in "
          f"{time.monotonic() - t_warm:.2f}s (paid once, not per-turn)\n")

    history = []
    timeline = []
    turn_idx = 0
    total_latency = 0.0
    total_first_sound = 0.0

    for caller_text in caller_turns:
        turn_idx += 1

        # 1. Caller's scripted line, synthesized to real audio.
        caller_clip = work / f"{turn_idx:02d}a_caller.wav"
        await synth(caller_tts, caller_text, caller_clip)
        timeline.append(caller_clip)

        # 2. Real Groq STT on that audio - closes the loop for real, exactly
        # what a live mic turn would produce. Timed.
        t_stt0 = time.monotonic()
        with open(caller_clip, "rb") as f:
            stt_resp = await client.audio.transcriptions.create(
                file=(caller_clip.name, f.read()), model=STT_MODEL
            )
        stt_time = time.monotonic() - t_stt0
        heard_text = stt_resp.text.strip() or caller_text

        print(f"[turn {turn_idx}] caller said:  {caller_text!r}")
        print(f"          STT heard:    {heard_text!r}  ({stt_time:.3f}s)")

        # 3. Backchannel decision (instant) + its TTS synthesis, run
        # CONCURRENTLY with the LLM call - same overlap the real pipeline
        # gets from BackchannelProcessor sitting before the LLM stage.
        phrase = choose_backchannel(
            heard_text, has_spoken_before=turn_idx > 1, language=args.language
        )
        history.append({"role": "user", "content": heard_text})

        async def run_llm():
            t0 = time.monotonic()
            stream = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, *history],
                stream=True,
                max_completion_tokens=120,
                reasoning_effort="low",
            )
            first_token_t = None
            chunks = []
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token_t is None:
                        first_token_t = time.monotonic()
                    chunks.append(delta)
            reply_text = normalize_speech_text("".join(chunks).strip())
            llm_time = time.monotonic() - t0
            return reply_text, llm_time

        async def run_backchannel():
            if not phrase:
                return None, 0.0
            clip = work / f"{turn_idx:02d}b_backchannel.wav"
            # Time-to-first-frame, not total synth time - streaming means
            # the caller hears it start well before synthesis finishes, and
            # that's the number that actually determines perceived latency.
            # Cache hits (see tts_fallback.py) make this ~0 for any phrase
            # already warmed, since there's no network call at all.
            first_frame_t, _total_t = await synth(agent_tts, phrase, clip)
            return clip, first_frame_t

        (reply_text, llm_time), (back_clip, back_time) = await asyncio.gather(
            run_llm(), run_backchannel()
        )
        history.append({"role": "assistant", "content": reply_text})

        # 4. Reply TTS - only starts once the LLM text is ready. Same
        # time-to-first-frame reasoning as the backchannel above.
        reply_clip = work / f"{turn_idx:02d}c_agent.wav"
        reply_tts_time, reply_tts_total = await synth(agent_tts, reply_text, reply_clip)

        # Real overlapped timeline: STT, then backchannel-TTS and
        # LLM-generation race in parallel, then reply-TTS starts only once
        # the LLM text is ready. The caller hears the backchannel as soon as
        # its own path finishes; the full reply as soon as *its* path
        # finishes - whichever is later determines when the reply plays.
        backchannel_ready_at = stt_time + back_time
        reply_ready_at = stt_time + llm_time + reply_tts_time
        gap_before_backchannel = stt_time
        gap_before_reply = max(0.0, reply_ready_at - backchannel_ready_at) if phrase else reply_ready_at

        turn_latency = reply_ready_at
        first_sound_at = backchannel_ready_at if phrase else reply_ready_at
        total_latency += turn_latency
        total_first_sound += first_sound_at

        print(f"          backchannel:  {phrase!r} (ready @ {backchannel_ready_at:.3f}s)"
              if phrase else "          backchannel:  (none this turn)")
        print(f"          agent reply:  {reply_text!r}")
        print(f"          LLM {llm_time:.3f}s + TTS-first-frame {reply_tts_time:.3f}s "
              f"(full synth {reply_tts_total:.3f}s), reply ready @ {reply_ready_at:.3f}s")
        print(f"          -> time to first audible sound: {first_sound_at:.3f}s "
              f"| time to full reply: {turn_latency:.3f}s\n")

        gap1 = work / f"{turn_idx:02d}_gap1.wav"
        make_silence(gap_before_backchannel, gap1)
        timeline.append(gap1)

        if phrase and back_clip:
            timeline.append(back_clip)

        gap2 = work / f"{turn_idx:02d}_gap2.wav"
        make_silence(gap_before_reply, gap2)
        timeline.append(gap2)
        timeline.append(reply_clip)

        # Small fixed breath before the next caller turn - not part of the
        # measured agent-latency number, just so turns don't run together.
        breath = work / f"{turn_idx:02d}_breath.wav"
        make_silence(0.5, breath)
        timeline.append(breath)

    concat_list = work / "_concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in timeline), encoding="utf-8"
    )

    raw_path = work / "_concatenated.wav"
    sh([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        str(raw_path), "-loglevel", "error",
    ])

    out_path = Path(f"samples/simulated_call_{args.language}.wav")
    add_background_noise(raw_path, out_path)

    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Average time to FIRST audible sound: {total_first_sound/turn_idx:.3f}s "
          f"across {turn_idx} turns")
    print(f"Average time to FULL reply: {total_latency/turn_idx:.3f}s across {turn_idx} turns")


if __name__ == "__main__":
    asyncio.run(main())
