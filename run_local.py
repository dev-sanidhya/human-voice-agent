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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from human_voice_agent.config import GROQ_API_KEY, LANGUAGE
from human_voice_agent.pipeline import build_pipeline

_GREETING_INSTRUCTION = {
    "en": "[The call just connected. Greet the caller in one short, "
    "natural sentence and ask how you can help.]",
    "hi": "[कॉल अभी कनेक्ट हुई है। कॉलर का एक छोटे, स्वाभाविक वाक्य में स्वागत "
    "करें और पूछें कि आप कैसे मदद कर सकते हैं।]",
    "hinglish": "[Call abhi connect hui hai. Caller ko ek chhote, natural "
    "sentence mein greet karo aur pucho ki tum kaise help kar sakte ho.]",
}


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

    _, task, context = build_pipeline(transport, language=args.language)

    context.add_message({"role": "user", "content": _GREETING_INSTRUCTION[args.language]})

    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    logger.info("Listening - start talking. Ctrl+C to end the call.")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
