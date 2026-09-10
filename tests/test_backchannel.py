import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from human_voice_agent import backchannel


def test_skips_opening_line():
    assert backchannel.choose_backchannel("hello there how are you doing today", has_spoken_before=False) is None


def test_skips_short_utterance():
    assert backchannel.choose_backchannel("yes please", has_spoken_before=True) is None


def test_skips_short_direct_question():
    assert backchannel.choose_backchannel("what's the capital of France?", has_spoken_before=True) is None


def test_never_repeats_consecutively():
    backchannel._last_phrase.clear()
    seen = []
    for _ in range(40):
        phrase = backchannel.choose_backchannel(
            "so I was calling about the invoice from last month that never arrived",
            has_spoken_before=True,
        )
        if phrase:
            if seen:
                assert phrase != seen[-1]
            seen.append(phrase)
    assert seen, "expected at least one backchannel to fire across 40 tries"


def test_hindi_and_hinglish_phrase_pools_differ_from_english():
    backchannel._last_phrase.clear()
    seen = {"en": set(), "hi": set(), "hinglish": set()}
    text = "haan yaar mujhe bhi lagta hai ki ye sahi rahega, chalo karte hain"
    for lang in seen:
        for _ in range(40):
            phrase = backchannel.choose_backchannel(text, has_spoken_before=True, language=lang)
            if phrase:
                seen[lang].add(phrase)
    assert seen["en"] and seen["hi"] and seen["hinglish"]
    assert seen["en"].isdisjoint(seen["hi"])
    assert seen["en"].isdisjoint(seen["hinglish"])


if __name__ == "__main__":
    test_skips_opening_line()
    test_skips_short_utterance()
    test_skips_short_direct_question()
    test_never_repeats_consecutively()
    test_hindi_and_hinglish_phrase_pools_differ_from_english()
    print("all tests passed")
