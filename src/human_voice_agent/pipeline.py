"""Builds the live voice pipeline: mic -> VAD -> smart-turn -> STT -> LLM ->
TTS -> speakers, all on Groq, with instant backchannel acknowledgments
layered in front of the LLM stage.

One provider (Groq) for STT + LLM + TTS on purpose: three separate vendors
means three separate network hops with independent latency variance. One
provider, one region, one connection pool.
"""
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transports.base_transport import BaseTransport

from human_voice_agent.backchannel import choose_backchannel
from human_voice_agent.phrase_bank import priority_phrases
from human_voice_agent.config import (
    CARTESIA_API_KEY,
    CARTESIA_API_KEY_2,
    CARTESIA_MODEL,
    GROQ_API_KEY,
    LANGUAGE,
    LLM_MODEL,
    STT_MODEL,
    TTS_MODEL,
    TTS_VOICE,
    TTS_VOICE_BY_LANGUAGE,
    VAD_STOP_SECS,
)
from human_voice_agent.prompts import get_system_prompt
from human_voice_agent.tts_fallback import ResilientTTSService


class BackchannelProcessor(FrameProcessor):
    """Speaks a short acknowledgment the instant the user's turn ends,
    before the LLM has produced any of the real answer.

    Sits between the user context aggregator and the LLM in the pipeline.
    A TTSSpeakFrame pushed here reaches the TTS service directly (the LLM
    service passes frames it doesn't own straight through), so the
    acknowledgment gets synthesized and starts playing while the LLM is
    still generating tokens for the actual reply - the two overlap instead
    of running back to back.
    """

    def __init__(self, language: str = "en"):
        super().__init__()
        self._has_spoken_before = False
        self._language = language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            phrase = choose_backchannel(
                frame.text, has_spoken_before=self._has_spoken_before, language=self._language
            )
            if phrase:
                logger.debug(f"backchannel: {phrase!r}")
                await self.push_frame(
                    TTSSpeakFrame(phrase, append_to_context=False), direction
                )
            self._has_spoken_before = True

        await self.push_frame(frame, direction)


async def build_pipeline(
    transport: BaseTransport, language: str = LANGUAGE
) -> tuple[Pipeline, PipelineTask, LLMContext]:
    """Assemble the full voice pipeline around a given transport.

    `language`: "en" (default), "hi" (Devanagari Hindi), or "hinglish"
    (Latin-script code-switched Hindi-English). Picks the system prompt,
    backchannel phrase pool, and TTS voice/provider for the whole call.

    Async because it pre-warms the phrase-bank audio cache (see
    ResilientTTSService.warm_cache / phrase_bank.py) - a one-time synthesis
    pass so the greeting and acknowledgment phrases are cache hits during
    the call instead of paying network latency on each one. Only warms
    `priority_phrases` (greeting + ack) - deliberately *not*
    `all_cacheable_phrases`, which also includes CLOSING: that category
    has no real trigger in the pipeline yet (no call-end flow exists), and
    warming phrases nothing can select was confirmed live to be pure
    wasted spend on the previous HOLD category before it got merged into
    ack - not repeating that mistake with CLOSING.
    """
    stt = GroqSTTService(api_key=GROQ_API_KEY, settings=GroqSTTService.Settings(model=STT_MODEL))

    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        settings=GroqLLMService.Settings(
            model=LLM_MODEL,
            # gpt-oss models on Groq default to spending their whole token
            # budget on hidden chain-of-thought before emitting any visible
            # reply (confirmed live: default settings burned 120/120 tokens
            # on `reasoning` and returned empty content). "low" cuts that to
            # a handful of tokens - essential for voice, where the visible
            # reply is all that matters and every reasoning token is added
            # latency the caller sits through in silence.
            extra={"reasoning_effort": "low"},
        ),
    )

    # All three languages route to Cartesia Sonic now (see config.py's
    # TTS_VOICE_BY_LANGUAGE). English started on Groq Orpheus, but its
    # free-tier 10 requests/minute cap turned out to be a real live problem:
    # confirmed live, warming the cacheable phrases at call start burns
    # through that limit, and the live per-turn replies (never cacheable -
    # fresh LLM text every turn) then queue behind the same limit, producing
    # 20s+ turns instead of ~1-2s. Cartesia has no such per-minute wall on
    # this account and was already proven ~2-3x faster time-to-first-audio
    # than edge-tts for Hindi/Hinglish - moving English onto it removes a
    # rate-limited code path instead of keeping a third one alive.
    # edge-tts remains the universal last-resort fallback for all three
    # languages if a Cartesia request fails.
    tts_provider, tts_voice = TTS_VOICE_BY_LANGUAGE.get(language, ("groq", TTS_VOICE))
    tts = ResilientTTSService(
        groq_api_key=GROQ_API_KEY,
        groq_model=TTS_MODEL,
        groq_voice=TTS_VOICE if tts_provider == "groq" else tts_voice,
        edge_voice=tts_voice if tts_provider == "edge" else "en-US-AndrewNeural",
        cartesia_api_key=CARTESIA_API_KEY,
        cartesia_api_key_2=CARTESIA_API_KEY_2,
        cartesia_model=CARTESIA_MODEL,
        cartesia_voice_id=tts_voice if tts_provider == "cartesia" else None,
        cartesia_language="hi" if language in ("hi", "hinglish") else "en",
        provider=tts_provider,
    )
    await tts.warm_cache(priority_phrases(language))

    context = LLMContext(messages=[{"role": "system", "content": get_system_prompt(language)}])
    # No explicit turn_analyzer here: LLMContextAggregatorPair's stop strategy
    # already defaults to TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)
    # when none is given - semantic end-of-turn detection out of the box.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        ),
    )

    backchannel = BackchannelProcessor(language=language)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            backchannel,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    return pipeline, task, context
