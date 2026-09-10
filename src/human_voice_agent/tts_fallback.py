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
        cartesia_api_key_2: Optional[str] = None,
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
        # Round-robined across however many Cartesia keys are configured.
        # Confirmed live with one key: this account's Cartesia concurrency
        # cap is 2, and warm_cache's concurrency=3+ was tripping it, sending
        # a few phrases to the edge-tts fallback instead of the real voice.
        # A second account key doubles effective concurrent capacity rather
        # than raising it past what any one account allows.
        self._cartesia_clients = [
            AsyncCartesia(api_key=key) for key in (cartesia_api_key, cartesia_api_key_2) if key
        ]
        self._cartesia_rr = 0
        # Indices into _cartesia_clients that have proven dead this process
        # (out of credits, or otherwise rejected) - confirmed live: without
        # this, round-robin keeps sending roughly half of every request to a
        # broke key, which fails and falls back to edge-tts every time,
        # dragging down real per-turn latency instead of the two keys
        # actually adding capacity.
        self._cartesia_blocked: set[int] = set()
        self._cartesia_model = cartesia_model
        self._cartesia_voice_id = cartesia_voice_id
        self._cartesia_language = cartesia_language
        # Once a primary provider proves unusable for this process (Groq's
        # terms wall, or a Cartesia failure), stop retrying it every single
        # utterance - go straight to the fallback for the rest of the call.
        self._primary_blocked = False
        # Pre-synthesized audio for known short phrases (backchannel words -
        # see warm_cache below). A cache hit skips the network entirely:
        # no STT/LLM is involved for these, so this is the one place actual
        # zero-network latency is achievable, not just "faster than before."
        self._cache: dict[str, bytes] = {}

    @property
    def _rate(self) -> int:
        """`sample_rate` only finalizes to a real value once a StartFrame
        flows through the pipeline (it reports 0 before that, confirmed
        live - it broke warm_cache() with a real "unsupported sample rate:
        0" error from Cartesia the first time this ran, since warm_cache is
        meant to run before the pipeline starts). The constructor's
        `sample_rate` argument is preserved separately by the base class as
        `_init_sample_rate` specifically for this - use that until the real
        one is set.
        """
        return self.sample_rate or self._init_sample_rate

    async def warm_cache(self, phrases: list[str], concurrency: int = 4) -> None:
        """Synthesizes each phrase once, through the real provider path
        (including fallback if the primary is unavailable), and stores the
        PCM bytes for instant reuse. Call this once at startup/pipeline
        build time, not per-call - the whole point is paying the network
        cost exactly once instead of on every single utterance.

        Runs up to `concurrency` requests in parallel - confirmed live this
        mattered once the phrase bank grew past just backchannels (5-8
        phrases) to include greetings too (priority_phrases is ~28 phrases
        for English now, which blocks call start): sequential warming at
        ~0.4-0.6s/phrase was adding several real seconds to every call's
        startup before this. English moved off Groq to Cartesia (see
        config.py), whose per-account concurrency cap is 2 - with two
        account keys now round-robined in _run_cartesia (also see
        config.py's CARTESIA_API_KEY_2), 4 is the matching combined
        capacity. Confirmed live: 3 against a single Cartesia key was still
        tipping a couple of phrases into the edge-tts fallback every run.
        """
        to_warm = [p for p in phrases if p not in self._cache]
        if not to_warm:
            return

        semaphore = asyncio.Semaphore(concurrency)

        async def warm_one(phrase: str) -> None:
            async with semaphore:
                async def collect() -> list[bytes]:
                    out = []
                    async for frame in self.run_tts(phrase, context_id=f"warm-{phrase}"):
                        if hasattr(frame, "audio"):
                            out.append(frame.audio)
                    return out

                try:
                    # edge-tts's underlying network stream has no timeout of
                    # its own - confirmed live it can hang indefinitely under
                    # load (5 concurrent fallback calls after a burst of Groq
                    # 429s), which without this wrapper freezes the whole
                    # asyncio.gather() below forever, not just this phrase.
                    chunks = await asyncio.wait_for(collect(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timed out warming phrase {phrase!r} after 15s, skipping it.")
                    return
                except Exception as exc:
                    # Confirmed live: edge-tts can also raise outright
                    # (NoAudioReceived) instead of hanging, e.g. when several
                    # fallback calls hit Microsoft's endpoint at once. Left
                    # uncaught, that exception propagates out of gather() and
                    # kills every other in-flight phrase too, not just this
                    # one - one flaky network call shouldn't take down the
                    # whole warm-up.
                    logger.warning(f"Failed warming phrase {phrase!r} ({exc}), skipping it.")
                    return
                if chunks:
                    self._cache[phrase] = b"".join(chunks)
                    logger.debug(f"Cached phrase {phrase!r} ({len(self._cache[phrase])} bytes)")

        await asyncio.gather(*(warm_one(p) for p in to_warm))

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        await self.start_ttfb_metrics()

        cached = self._cache.get(text)
        if cached is not None:
            # No provider call at all - this is the only place in the
            # pipeline where "latency" isn't reduced, it's genuinely zero
            # (beyond local audio-chunking overhead). Chunked rather than
            # yielded as one frame so downstream behaves the same as a
            # streamed synthesis, not a single giant frame.
            await self.stop_ttfb_metrics()
            CHUNK_BYTES = 4800
            for i in range(0, len(cached), CHUNK_BYTES):
                yield TTSAudioRawFrame(
                    cached[i : i + CHUNK_BYTES], self._rate, 1, context_id=context_id
                )
            return

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
        if not self._cartesia_clients:
            raise RuntimeError("No CARTESIA_API_KEY configured")

        available = [i for i in range(len(self._cartesia_clients)) if i not in self._cartesia_blocked]
        if not available:
            raise RuntimeError("All configured Cartesia keys are blocked (out of credits or rejected)")

        idx = available[self._cartesia_rr % len(available)]
        self._cartesia_rr += 1
        client = self._cartesia_clients[idx]
        try:
            ws = await client.tts.websocket()
        except Exception as exc:
            if _is_dead_key_error(exc):
                logger.warning(f"Cartesia key #{idx + 1} looks dead ({exc}), blocking it for the rest of this run.")
                self._cartesia_blocked.add(idx)
            raise
        try:
            gen = await ws.send(
                model_id=self._cartesia_model,
                transcript=text,
                voice={"id": self._cartesia_voice_id},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self._rate,
                },
                language=self._cartesia_language,
            )
            first_frame = True
            async for out in gen:
                if out.audio:
                    if first_frame:
                        await self.stop_ttfb_metrics()
                        first_frame = False
                    yield TTSAudioRawFrame(out.audio, self._rate, 1, context_id=context_id)
        except Exception as exc:
            if _is_dead_key_error(exc):
                logger.warning(f"Cartesia key #{idx + 1} looks dead ({exc}), blocking it for the rest of this run.")
                self._cartesia_blocked.add(idx)
            raise
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
            "-f", "s16le", "-ac", "1", "-ar", str(self._rate),
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
                yield TTSAudioRawFrame(pcm_chunk, self._rate, 1, context_id=context_id)
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


def _is_dead_key_error(exc: Exception) -> bool:
    """True for a Cartesia error that means this specific key is done for
    the rest of the process (out of credits, subscription/payment issue) -
    not a transient one like a per-minute rate limit, which a later request
    on the same key could still succeed at. Confirmed live: an out-of-credit
    key returns either an "Insufficient credits" message or an outright
    HTTP 402 on the websocket handshake.
    """
    msg = str(exc)
    return "insufficient credits" in msg.lower() or "402" in msg
