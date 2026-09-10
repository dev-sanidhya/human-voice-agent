"""Live entrypoint: talk to the agent through your actual microphone and
speakers. No browser, no telephony, no Docker - just your machine's default
audio devices.

    python run_local.py

Ctrl+C to end the call.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from human_voice_agent.config import GROQ_API_KEY
from human_voice_agent.pipeline import build_pipeline


async def main():
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set - copy .env.example to .env and add your key.")
        sys.exit(1)

    # VAD and semantic turn-detection live on the user context aggregator
    # (see pipeline.py) - the transport just needs raw audio in/out enabled.
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    _, task, context = build_pipeline(transport)

    context.add_message(
        {
            "role": "user",
            "content": "[The call just connected. Greet the caller in one short, "
            "natural sentence and ask how you can help.]",
        }
    )

    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    logger.info("Listening - start talking. Ctrl+C to end the call.")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
