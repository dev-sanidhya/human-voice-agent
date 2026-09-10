"""Print the Groq models currently available to this API key.

Groq's model catalog changes over time (models get deprecated/replaced).
If config.py's defaults 404, run this to see what's actually live right now
and update HVA_STT_MODEL / HVA_LLM_MODEL / HVA_TTS_MODEL accordingly.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from groq import AsyncGroq

from human_voice_agent.config import GROQ_API_KEY


async def main():
    load_dotenv()
    client = AsyncGroq(api_key=GROQ_API_KEY)
    models = await client.models.list()
    for m in sorted(models.data, key=lambda m: m.id):
        print(m.id)


if __name__ == "__main__":
    asyncio.run(main())
