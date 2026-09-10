"""Calls the real shipped ResilientTTSService directly and writes its output
to a WAV file - proof that the fallback path in pipeline.py (not a separate
ad-hoc script) produces real audio right now, with zero manual setup.
"""
import asyncio
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from pipecat.frames.frames import TTSAudioRawFrame

from human_voice_agent.config import GROQ_API_KEY, TTS_MODEL, TTS_VOICE
from human_voice_agent.tts_fallback import ResilientTTSService


async def main():
    load_dotenv()
    # sample_rate only finalizes once a StartFrame flows through a real
    # pipeline (it reports 0 beforehand) - pin it explicitly here since this
    # script calls the service directly, outside a pipeline.
    tts = ResilientTTSService(
        groq_api_key=GROQ_API_KEY, groq_model=TTS_MODEL, groq_voice=TTS_VOICE, sample_rate=48000
    )
    tts._sample_rate = 48000

    text = "Right, hey, sorry to hear that. What happened with the service?"
    frames = []
    async for frame in tts.run_tts(text, context_id="verify"):
        if isinstance(frame, TTSAudioRawFrame):
            frames.append(frame)

    if not frames:
        print("No audio frames produced.")
        return

    out_path = Path("samples/fallback_verified.wav")
    out_path.parent.mkdir(exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(frames[0].num_channels)
        wf.setsampwidth(2)
        wf.setframerate(frames[0].sample_rate)
        for f in frames:
            wf.writeframes(f.audio)

    print(f"Wrote {out_path} from the real ResilientTTSService ({len(frames)} frame(s)).")


if __name__ == "__main__":
    asyncio.run(main())
