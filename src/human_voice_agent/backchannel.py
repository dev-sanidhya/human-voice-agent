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

_PHRASES = [
    "Mm-hmm,",
    "Right,",
    "Got it,",
    "Yeah,",
    "Okay,",
    "I see,",
    "Sure,",
    "Gotcha,",
]

# Module-level so a single process remembers what it last said - avoids
# "yeah... yeah... yeah..." across consecutive turns, which reads as a bug,
# not a personality.
_last_phrase: str | None = None


def _is_direct_short_question(text: str) -> bool:
    """A quick factual question deserves an answer, not a "hmm" first."""
    stripped = text.strip()
    return stripped.endswith("?") and len(stripped.split()) < 8


def choose_backchannel(user_text: str, *, has_spoken_before: bool) -> str | None:
    """Return an acknowledgment phrase to speak first, or None to stay quiet.

    Returns None (stay quiet) when:
    - this is the opening line of the call (nothing to "acknowledge" yet)
    - the utterance is short (< BACKCHANNEL_MIN_WORDS words) - "yeah" after
      "hi" or "yes" is the single most common way these demos sound robotic
    - the utterance is a short direct question - answer it, don't stall it
    - on a random ~35% of otherwise-eligible turns, so it doesn't become a
      tic the user notices and gets annoyed by
    """
    global _last_phrase

    if not has_spoken_before:
        return None

    word_count = len(user_text.split())
    if word_count < BACKCHANNEL_MIN_WORDS:
        return None

    if _is_direct_short_question(user_text):
        return None

    if random.random() > 0.65:
        return None

    choices = [p for p in _PHRASES if p != _last_phrase] or _PHRASES
    phrase = random.choice(choices)
    _last_phrase = phrase
    return phrase
