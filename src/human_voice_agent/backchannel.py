"""Instant acknowledgment words - the "sounds good", "yeah", "mm-hmm" layer.

The trick that makes this feel human isn't the phrase list, it's the timing:
the acknowledgment is queued for TTS the instant the user's turn ends, before
the LLM has produced a single token of the real answer. Since it's 1-2 words
against a full sentence, it's synthesized and starts playing first, filling
the LLM's think time instead of leaving dead air. See BackchannelProcessor in
pipeline.py for where this actually gets wired into the frame flow.

Real humans don't back-channel after every single thing you say, and they
never do it after "hi" or a one-word answer to a direct question. Getting
*when to stay quiet* right matters more than the phrase list.
"""
import random

from human_voice_agent.config import BACKCHANNEL_MIN_WORDS

_PHRASES_BY_LANGUAGE = {
    "en": ["Mm-hmm,", "Right,", "Got it,", "Yeah,", "Okay,", "I see,", "Sure,", "Gotcha,"],
    # Devanagari - real Hindi acknowledgments, not transliterated English ones.
    "hi": ["हाँ,", "ठीक है,", "अच्छा,", "समझ गया,", "जी,"],
    # Latin script, the way these actually get typed/said in casual Hinglish -
    # not a literal translation of the English list, the phrases people
    # actually use ("theek hai", "haan", "acha") when code-switching.
    "hinglish": ["Haan,", "Theek hai,", "Acha,", "Samajh gaya,", "Arre haan,"],
}

# Module-level so a single process remembers what it last said - avoids
# "yeah... yeah... yeah..." across consecutive turns, which reads as a bug,
# not a personality. Keyed by language so switching languages mid-process
# (shouldn't normally happen, but tests do) doesn't cross-contaminate.
_last_phrase: dict[str, str | None] = {}


def get_backchannel_phrases(language: str = "en") -> list[str]:
    """The full phrase pool for a language - for pre-warming the TTS cache
    (see ResilientTTSService.warm_cache), not for runtime selection.
    """
    return list(_PHRASES_BY_LANGUAGE.get(language, _PHRASES_BY_LANGUAGE["en"]))


def _is_direct_short_question(text: str) -> bool:
    """A quick factual question deserves an answer, not a "hmm" first."""
    stripped = text.strip()
    return stripped.endswith("?") and len(stripped.split()) < 8


def choose_backchannel(
    user_text: str, *, has_spoken_before: bool, language: str = "en"
) -> str | None:
    """Return an acknowledgment phrase to speak first, or None to stay quiet.

    Returns None (stay quiet) when:
    - this is the opening line of the call (nothing to "acknowledge" yet)
    - the utterance is short (< BACKCHANNEL_MIN_WORDS words) - "yeah" after
      "hi" or "yes" is the single most common way these demos sound robotic
    - the utterance is a short direct question - answer it, don't stall it
    - on a random ~35% of otherwise-eligible turns, so it doesn't become a
      tic the user notices and gets annoyed by

    `language` picks the phrase pool: "en" (English), "hi" (Devanagari
    Hindi), or "hinglish" (Latin-script code-switched Hindi-English). Word
    counting is whitespace-based, which holds up fine for Hindi and Hinglish
    too since both are space-separated in normal writing.
    """
    phrases = _PHRASES_BY_LANGUAGE.get(language, _PHRASES_BY_LANGUAGE["en"])

    if not has_spoken_before:
        return None

    word_count = len(user_text.split())
    if word_count < BACKCHANNEL_MIN_WORDS:
        return None

    if _is_direct_short_question(user_text):
        return None

    if random.random() > 0.65:
        return None

    last = _last_phrase.get(language)
    choices = [p for p in phrases if p != last] or phrases
    phrase = random.choice(choices)
    _last_phrase[language] = phrase
    return phrase
