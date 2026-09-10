"""TTS with automatic fallbacks and a per-language provider choice.

Three providers, picked per language (see config.py's TTS_VOICE_BY_LANGUAGE):

- **Groq Orpheus** (English default) - gated behind a one-time
  terms-acceptance click in Groq's console (a login-session-only action, no
  API exists for it, confirmed by hand). Falls back to edge-tts if that
  wall is hit, so the agent never just goes silent.
- **Cartesia Sonic** (Hindi/Hinglish default) - real streaming WebSocket
  TTS with native Hindi/Hinglish support and, confirmed live, ~0.4-0.5s
  time-to-first-audio - roughly 2-3x faster than the edge-tts path it
  replaced for these languages (which measured ~0.9-1.5s time-to-first-audio
  because Microsoft's TTS only returns MP3, which then needs an extra
  decode step; Cartesia streams raw PCM directly, no decode needed at all).
- **edge-tts** - the universal fallback. Free, no signup, works if the
  other two are unavailable, but real-measured as the slowest of the three.

Requires `ffmpeg` on PATH for the edge-tts path only (it returns MP3;
pipecat needs raw PCM). Groq and Cartesia have no such dependency - both
already speak PCM.
"""
import asyncio
import io
import shutil
import wave
from typing import AsyncGenerator, Optional

import edge_tts
from cartesia import AsyncCartesia
from groq import AsyncGroq
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from human_voice_agent.text_normalize import normalize_speech_text


async def _normalize_transform(text: str, _aggregation_type) -> str:
    return normalize_speech_text(text)


class ResilientTTSService(TTSService):
    """Groq Orpheus / Cartesia Sonic, each with an automatic edge-tts
    fallback, selected per language by `provider`.
    """

    def __init__(
        self,
        *,
        groq_api_key: str,
        groq_model: str,
        groq_voice: str,
        edge_voice: str = "en-US-AndrewNeural",
        cartesia_api_key: Optional[str] = None,
        cartesia_model: str = "sonic-turbo",
        cartesia_voice_id: Optional[str] = None,
        cartesia_language: str = "en",
        sample_rate: int = 48000,
        provider: str = "groq",
        **kwargs,
    ):
        # This service picks between real providers itself rather than
        # wrapping one via pipecat's settings framework, so there's no single
        # "model"/"voice" to report - `extra` documents all of them for
        # anyone inspecting settings at runtime.
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(
                model=None,
                voice=None,
                language=None,
                extra={
                    "groq_model": groq_model,
                    "groq_voice": groq_voice,
                    "edge_voice": edge_voice,
                    "cartesia_model": cartesia_model,
                    "cartesia_voice_id": cartesia_voice_id,
                },
            ),
            # Fixes the audible symptom of the glued-sentence defect
            # documented in text_normalize.py before it ever reaches TTS.
            text_transforms=[("*", _normalize_transform)],
            **kwargs,
        )
        self._provider = provider
        self._groq_client = AsyncGroq(api_key=groq_api_key)
        self._groq_model = groq_model
        self._groq_voice = groq_voice
        self._edge_voice = edge_voice
        self._cartesia_client = AsyncCartesia(api_key=cartesia_api_key) if cartesia_api_key else None
        self._cartesia_model = cartesia_model
        self._cartesia_voice_id = cartesia_voice_id
        self._cartesia_language = cartesia_language
        # Once a primary provider proves unusable for this process (Groq's
        # terms wall, or a Cartesia failure), stop retrying it every single
        # utterance - go straight to the fallback for the rest of the call.
        self._primary_blocked = False

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()

        if not self._primary_blocked:
            try:
                if self._provider == "cartesia":
                    async for frame in self._run_cartesia(text, context_id):
                        yield frame
                    return
                else:
                    async for frame in self._run_groq(text, context_id):
                        yield frame
                    return
            except Exception as exc:
                if self._provider == "groq" and "model_terms_required" in str(exc):
                    logger.warning(
                        "Groq TTS blocked: model terms not accepted yet in the "
                        "Groq console (this is a one-time, login-only step - see "
                        "README). Falling back to a free edge-tts voice for the "
                        "rest of this run."
                    )
                    self._primary_blocked = True
                else:
                    logger.warning(
                        f"{self._provider} TTS failed ({exc}), falling back to edge-tts "
                        "for this line."
                    )

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

    async def _run_cartesia(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Real WebSocket streaming TTS - Cartesia sends raw PCM chunks
        directly over the wire, so unlike the edge-tts path there's no
        intermediate container format and no decode step at all. Confirmed
        live: ~0.4-0.5s to first audio chunk for Hindi text on `sonic-turbo`
        (picked over the newer `sonic-3.6` specifically for lower latency -
        sonic-3.6 measured ~0.53s vs sonic-turbo's ~0.38s in a direct
        side-by-side test, even though sonic-3.6 is Cartesia's newest model
        with marketed Hindi/Hinglish improvements - swap
        HVA_CARTESIA_MODEL if accent quality matters more than the ~150ms
        difference for your use case).
        """
        if not self._cartesia_client:
            raise RuntimeError("No CARTESIA_API_KEY configured")

        ws = await self._cartesia_client.tts.websocket()
        try:
            gen = await ws.send(
                model_id=self._cartesia_model,
                transcript=text,
                voice={"id": self._cartesia_voice_id},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                language=self._cartesia_language,
            )
            first_frame = True
            async for out in gen:
                if out.audio:
                    if first_frame:
                        await self.stop_ttfb_metrics()
                        first_frame = False
                    yield TTSAudioRawFrame(out.audio, self.sample_rate, 1, context_id=context_id)
        finally:
            await ws.close()

    async def _run_edge(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Streams edge-tts's mp3 straight into ffmpeg and yields PCM chunks
        as they come out, instead of waiting for the entire utterance to
        finish synthesizing and decoding first.

        Confirmed live this was the actual latency bug, not edge-tts itself:
        profiled edge-tts's own network stream in isolation and its first
        audio chunk arrives in ~0.9s - but the old code buffered the *whole*
        stream (~1.4s) and then ran a *separate* full decode after that
        before yielding anything, so time-to-first-audio was really the
        full synthesis+decode time (up to ~2.7s on longer Hindi replies),
        throwing away the head start the network stream already gave it.
        """
        if not shutil.which("ffmpeg"):
            yield ErrorFrame(error="edge-tts fallback needs ffmpeg on PATH and it wasn't found.")
            return

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate),
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def feed_stdin():
            communicate = edge_tts.Communicate(text, self._edge_voice)
            got_audio = False
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    got_audio = True
                    proc.stdin.write(chunk["data"])
                    await proc.stdin.drain()
            proc.stdin.close()
            return got_audio

        feed_task = asyncio.create_task(feed_stdin())

        first_frame = True
        CHUNK_BYTES = 4800  # ~100ms of 24kHz mono s16le - small enough for low latency
        try:
            while True:
                pcm_chunk = await proc.stdout.read(CHUNK_BYTES)
                if not pcm_chunk:
                    break
                if first_frame:
                    await self.stop_ttfb_metrics()
                    first_frame = False
                yield TTSAudioRawFrame(pcm_chunk, self.sample_rate, 1, context_id=context_id)
        finally:
            got_audio = await feed_task
            returncode = await proc.wait()
            if first_frame:
                # Nothing was ever yielded - either edge-tts gave no audio
                # or ffmpeg failed outright. stop_ttfb_metrics wasn't called
                # above, so call it now to avoid leaving metrics hanging.
                await self.stop_ttfb_metrics()
                stderr = (await proc.stderr.read()).decode(errors="replace")
                if not got_audio:
                    yield ErrorFrame(error="edge-tts returned no audio")
                elif returncode != 0:
                    yield ErrorFrame(error=f"ffmpeg mp3->pcm decode failed: {stderr}")


def _wav_bytes_to_pcm(data: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()
