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
    backchannel._last_phrase = None
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


if __name__ == "__main__":
    test_skips_opening_line()
    test_skips_short_utterance()
    test_skips_short_direct_question()
    test_never_repeats_consecutively()
    print("all tests passed")
