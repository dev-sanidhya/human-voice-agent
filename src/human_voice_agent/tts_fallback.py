"""TTS with an automatic, zero-signup fallback.

Groq's Orpheus TTS is the intended production voice (see config.py) and
gets tried first on every utterance. But it's gated behind a one-time
terms-acceptance click in Groq's console (a login-session-only action - no
API exists for it, confirmed by hand), which some environments running this
agent won't have done yet.

Rather than the whole agent going silent until that click happens, this
service catches exactly that failure once, logs a clear warning, and falls
back to Microsoft Edge's free neural TTS (via the `edge-tts` package - no
account, no key, no signup) for the rest of the call. It's a stand-in voice,
not the shipped default - swap HVA_TTS_MODEL/HVA_TTS_VOICE back to Groq's
Orpheus the moment the console terms are accepted and this fallback stops
triggering on its own (Groq is retried fresh on every new run of the agent).

Requires `ffmpeg` on PATH for the fallback path only (edge-tts returns MP3;
pipecat needs raw PCM). The primary Groq path has no such dependency.
"""
import asyncio
import io
import shutil
import wave
from typing import AsyncGenerator

import edge_tts
from groq import AsyncGroq
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class ResilientTTSService(TTSService):
    """Groq Orpheus TTS with an automatic edge-tts fallback."""

    def __init__(
        self,
        *,
        groq_api_key: str,
        groq_model: str,
        groq_voice: str,
        edge_voice: str = "en-US-AndrewNeural",
        sample_rate: int = 48000,
        **kwargs,
    ):
        # This service picks between two real providers itself rather than
        # wrapping one via pipecat's settings framework, so there's no single
        # "model"/"voice" to report - `extra` documents both for anyone
        # inspecting settings at runtime.
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(
                model=None,
                voice=None,
                language=None,
                extra={"groq_model": groq_model, "groq_voice": groq_voice, "edge_voice": edge_voice},
            ),
            **kwargs,
        )
        self._groq_client = AsyncGroq(api_key=groq_api_key)
        self._groq_model = groq_model
        self._groq_voice = groq_voice
        self._edge_voice = edge_voice
        # Once Groq TTS proves terms-gated for this process, stop retrying it
        # every single utterance - go straight to the fallback for the rest
        # of the call instead of eating a failed round trip each time.
        self._groq_terms_blocked = False

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()

        if not self._groq_terms_blocked:
            try:
                async for frame in self._run_groq(text, context_id):
                    yield frame
                return
            except Exception as exc:
                if "model_terms_required" in str(exc):
                    logger.warning(
                        "Groq TTS blocked: model terms not accepted yet in the "
                        "Groq console (this is a one-time, login-only step - see "
                        "README). Falling back to a free edge-tts voice for the "
                        "rest of this run."
                    )
                    self._groq_terms_blocked = True
                else:
                    logger.warning(f"Groq TTS failed ({exc}), falling back for this line.")

        async for frame in self._run_edge(text, context_id):
            yield frame

    async def _run_groq(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        resp = await self._groq_client.audio.speech.create(
            model=self._groq_model,
            voice=self._groq_voice,
            input=text,
            response_format="wav",
        )
        await self.stop_ttfb_metrics()
        pcm, rate, channels = _wav_bytes_to_pcm(await resp.read())
        yield TTSAudioRawFrame(pcm, rate, channels, context_id=context_id)

    async def _run_edge(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not shutil.which("ffmpeg"):
            yield ErrorFrame(error="edge-tts fallback needs ffmpeg on PATH and it wasn't found.")
            return

        communicate = edge_tts.Communicate(text, self._edge_voice)
        mp3_chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.extend(chunk["data"])

        await self.stop_ttfb_metrics()

        if not mp3_chunks:
            yield ErrorFrame(error="edge-tts returned no audio")
            return

        pcm = await _mp3_to_pcm(bytes(mp3_chunks), self.sample_rate)
        yield TTSAudioRawFrame(pcm, self.sample_rate, 1, context_id=context_id)


def _wav_bytes_to_pcm(data: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()


async def _mp3_to_pcm(mp3_bytes: bytes, sample_rate: int) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pcm, stderr = await proc.communicate(input=mp3_bytes)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3->pcm decode failed: {stderr.decode(errors='replace')}")
    return pcm
