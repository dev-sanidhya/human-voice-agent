"""Live entrypoint: talk to the agent through your actual microphone and
speakers. No browser, no telephony, no Docker - just your machine's default
audio devices.

    python run_local.py                  # English (default, or HVA_LANGUAGE)
    python run_local.py --language hi        # Hindi (Devanagari)
    python run_local.py --language hinglish   # Hinglish (Latin script)

Ctrl+C to end the call.
"""
import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from human_voice_agent.config import GROQ_API_KEY, LANGUAGE
from human_voice_agent.pipeline import build_pipeline
from human_voice_agent.phrase_bank import GREETING


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=LANGUAGE, choices=["en", "hi", "hinglish"])
    args = parser.parse_args()

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set - copy .env.example to .env and add your key.")
        sys.exit(1)

    # VAD and semantic turn-detection live on the user context aggregator
    # (see pipeline.py) - the transport just needs raw audio in/out enabled.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    logger.info("Warming phrase cache...")
    _, task, _ = await build_pipeline(transport, language=args.language)

    # Cached greeting, not an LLM call - confirmed live this was costing a
    # full LLM+TTS round trip (1-2s+) before the caller heard anything at
    # all, on every single call, for text that's the same handful of
    # options every time. append_to_context=True keeps it in the
    # conversation history so the LLM sees what was actually said.
    greeting = random.choice(GREETING.get(args.language, GREETING["en"]))
    await task.queue_frames([TTSSpeakFrame(greeting, append_to_context=True)])

    runner = PipelineRunner()
    logger.info("Listening - start talking. Ctrl+C to end the call.")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
