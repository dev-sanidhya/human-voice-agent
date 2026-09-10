"""Defensive text cleanup before TTS.

Confirmed live against the real Groq API (openai/gpt-oss-20b,
reasoning_effort=low): roughly 1 in 8 replies glue two sentences together
with no space after the period - typically the model narrating a pending
action ("Let me check the order.") and then, in the same completion,
hallucinating its resolution ("Got it, it shipped yesterday.") with nothing
between them. Tried fixing this by instructing the model not to do it in
the system prompt - measured that this made it *worse* (roughly 1 in 2 over
a 12-sample retest, up from 1 in 8), so that approach was reverted rather
than shipped. A stochastic habit in a small, low-reasoning-effort model
isn't reliably fixable by asking nicely.

This fixes the audible symptom instead: without a space, TTS reads
"...at.Got it..." as one run-on breath with no pause. Inserting the space
makes the two sentences read as two sentences again, with the pause a
period should produce - it doesn't stop the model from occasionally saying
something slightly odd, but it stops the audio from sounding broken.

Also strips parenthetical stage directions. Confirmed live in Hindi mode:
the model wrote "ठीक है, एक मिनट रुकिए... (रुकें) आपकी ट्रैकिंग..." - a
literal "(pause)" stage direction in the text, which TTS has no way to
interpret as an instruction and just reads aloud as the word "pause". Same
"fix the code, not the prompt" reasoning as the glued-sentence fix above:
telling the model not to do this is one more instruction for a small,
low-reasoning-effort model to occasionally ignore under load, so this
strips anything in parens before it ever reaches TTS instead.
"""
import re

# Devanagari (Hindi) has no case distinction, so "next sentence starts with
# an uppercase letter" doesn't apply - instead this looks for the Hindi
# sentence-ending danda (।) or Latin ./!/? immediately followed by any
# Devanagari or Latin letter, glued with no space.
_GLUED_SENTENCE = re.compile(r"([.!?।])([A-Za-zऀ-ॿ])")
_PARENTHETICAL = re.compile(r"[(（][^)）]*[)）]")
_EXTRA_SPACE = re.compile(r" {2,}")


def normalize_speech_text(text: str) -> str:
    text = _GLUED_SENTENCE.sub(r"\1 \2", text)
    text = _PARENTHETICAL.sub("", text)
    return _EXTRA_SPACE.sub(" ", text).strip()
