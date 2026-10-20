"""
A deterministic, rule-based Yorùbá numeral parser — the guardrail for
Phase 5's LLM-based amount extraction (app/intent.py).

Why this exists: a systematic test (phase2-asr/scripts/numeral_parsing_test.py)
found Claude Haiku 4.5 only correctly parses 5/14 (36%) of real Yorùbá
number words, with CONFIDENT, SILENT wrong answers — exactly the failure
mode a money-handling app can't tolerate unchecked. See PLAN.md.

Deliberately NOT a full compositional parser for arbitrary Yorùbá numbers.
The compound/subtractive system (e.g. 45, 67) is genuinely complex, and a
guardrail that's confidently wrong is worse than one that admits it can't
help. Instead: an explicit, small vocabulary of the exact word forms
already tested and ground-truthed against real model output — units 1-10,
the base-20 tens (including the irregular subtractive forms for 50/70/90),
100 and the irregular 200 ("igba"), and thousand-multiples ("ẹgbẹ̀rún" +
a unit/ten/hundred word). Anything outside that returns None (unparseable)
— callers must treat None as "couldn't independently verify", never as 0.

Tone-stripping follows the same approach as phase7-integration/app/router.py
(strip acute/grave marks, preserve underdots ẹ/ọ since those are phoneme
distinctions, not tone) — duplicated here rather than imported, since
Phase 5 doesn't depend on Phase 7 (wrong dependency direction) and this is
a small, self-contained ~10-line helper.

Usage:
    python3 app/yoruba_numerals.py --self-test
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

_TONE_MAP = {
    "á": "a", "à": "a", "é": "e", "è": "e", "í": "i", "ì": "i",
    "ó": "o", "ò": "o", "ú": "u", "ù": "u", "ń": "n",
    "Á": "A", "À": "A", "É": "E", "È": "E", "Í": "I", "Ì": "I",
    "Ó": "O", "Ò": "O", "Ú": "U", "Ù": "U", "Ń": "N",
}

# ẹ́/ẹ̀/ọ́/ọ̀ are NOT single codepoints — even after NFC normalization,
# "ẹ" (U+1EB9, precomposed "e with dot below") has no precomposed form
# combined with a tone mark on top, so the acute/grave rides along as a
# separate standalone combining character (U+0301, U+0300). A naive
# per-codepoint dict lookup (as used for the plain vowels above) silently
# leaves these combining marks untouched — found by testing this module,
# not assumed. Strip them explicitly in a second pass, which removes the
# tone mark while leaving the precomposed ẹ/ọ base letter alone (correct:
# the underdot is a phoneme distinction, not tone, and must survive).
_COMBINING_TONE_MARKS = {"̀", "́"}


def _strip_tone_marks(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(_TONE_MAP.get(ch, ch) for ch in text)
    text = "".join(ch for ch in text if ch not in _COMBINING_TONE_MARKS)
    return text


def _normalize(text: str) -> str:
    """Lowercase, strip tone marks, drop hyphens (ASR/typed text is
    inconsistent about "ọgọ́rùn-ún" vs "ọgọrunun"), collapse whitespace."""
    text = _strip_tone_marks(text.lower())
    text = text.replace("-", "")
    return re.sub(r"\s+", " ", text).strip()


# Counting-form units (used when a number modifies a noun, e.g. "náírà
# mẹ́ta" = "three naira") — the same form used in numeral_parsing_test.py.
# Keys are the REAL output of _normalize() on the actual Yorùbá word
# (computed directly, not guessed — underdot letters ẹ/ọ are preserved,
# see the module docstring for why a naive ASCII-only key would be wrong).
_UNITS = {
    "okan": 1, "eni": 1, "kan": 1,
    "meji": 2,
    "mẹta": 3,
    "mẹrin": 4,
    "marunun": 5,
    "mẹfa": 6,
    "meje": 7,
    "mẹjọ": 8,
    "mẹsanan": 9,
    "mẹwaa": 10, "mẹwa": 10,  # "mẹwa" is the informal spelling actually used
    # in this project's own real test sentence (phase7-integration's
    # FINANCE_TEXT) — found by testing against it, not assumed.
}

# Base-20 tens, including the irregular subtractive forms (50/70/90 are
# each "10 short of" the next multiple of 20 — a real, documented feature
# of Yorùbá numerals, not a typo).
_TENS = {
    "ogun": 20,
    "ọgbọn": 30,
    "ogoji": 40,
    "aadọta": 50,
    "ọgọta": 60,
    "aadọrin": 70,
    "ọgọrin": 80,
    "aadọrunun": 90,
    "ọgọrunun": 100,
}

# Irregular hundreds — 200 is its own word, not "ọgọ́rùn méjì". Larger
# hundreds (300+) are out of scope (see module docstring).
_HUNDREDS_IRREGULAR = {
    "igba": 200,
}

_MULTIPLIER_VOCAB = {**_UNITS, **_TENS, **_HUNDREDS_IRREGULAR}

_THOUSAND_WORD = "ẹgbẹrun"

# Underdot letters (ẹ/ọ) are a real phoneme distinction in written Yorùbá,
# so the vocabulary above preserves them and the primary matching pass
# does too. But real ASR output doesn't always preserve them either —
# confirmed live, 2026-08-15: with TTS output variance held fixed (a
# fixed seed), Whisper's transcription of "ẹgbẹ̀rún mẹ́wàá" came back as
# plain "egberun mawa" — underdots dropped entirely, not just tone marks.
# A strict-only matcher would silently fail on real ASR output like this.
# So: try the strict (underdot-preserving) vocabulary first; if nothing
# matches, retry against an underdot-collapsed ("loose") copy of the same
# vocabulary. This deliberately trades a little phoneme precision for
# real-world ASR robustness — same trade-off phase7-integration/app/
# router.py already makes for its own keyword list.
_UNDERDOT_MAP = {"ẹ": "e", "ọ": "o"}


def _strip_underdots(text: str) -> str:
    return "".join(_UNDERDOT_MAP.get(ch, ch) for ch in text)


def _loose(vocab: dict) -> dict:
    """Collapses a vocabulary's keys to underdot-free form. Where two
    strict keys collapse to the same loose key (shouldn't happen with the
    current vocabulary, but would be a real ambiguity if it did), the
    later entry silently wins — acceptable here since it's checked by
    the self-test asserting no vocabulary regressions, not left unnoticed."""
    return {_strip_underdots(k): v for k, v in vocab.items()}


_MULTIPLIER_VOCAB_LOOSE = _loose(_MULTIPLIER_VOCAB)
_THOUSAND_WORD_LOOSE = _strip_underdots(_THOUSAND_WORD)

# Loose-only entries: real observed ASR mishearings that don't fit the
# systematic underdot-collapse rule above (ẹ->e), so they can't be
# derived automatically — added directly to the loose vocabulary only.
# Deliberately NOT added to the strict vocabulary: a word with no
# underdots of its own (like "mawa") would otherwise be reachable by
# the single-word fallback in the STRICT pass too, short-circuiting the
# "thousand-word + multiplier" check before the loose pass ever runs —
# a real bug caught by testing, not a hypothetical (see git history /
# PLAN.md: an earlier version returned 10 instead of 10000 for exactly
# this reason).
_MULTIPLIER_VOCAB_LOOSE["mawa"] = 10  # "mẹwa" (ten) mis-transcribed by
# Whisper as "mawa" (ẹ->a, not the more common ẹ->e) — confirmed live,
# 2026-08-15, with TTS output variance held fixed via TTS_SEED so this
# wasn't a one-off garbling.


def _parse_with_vocab(words: list[str], thousand_word: str, multiplier_vocab: dict) -> Optional[int]:
    for i, word in enumerate(words):
        if word == thousand_word:
            if i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in multiplier_vocab:
                    return 1000 * multiplier_vocab[next_word]
                return None  # unrecognized word follows — don't guess
            return 1000  # thousand-word is the last word, genuinely bare

    for word in words:
        if word in multiplier_vocab:
            return multiplier_vocab[word]

    return None


def parse_yoruba_number(text: str) -> Optional[int]:
    """Returns the integer value of a Yorùbá number word/phrase found in
    `text`, or None if nothing in the small known vocabulary matches —
    tried strictly first (underdots preserved, most linguistically
    precise), then loosely (underdots collapsed, more tolerant of real
    ASR output that dropped them) if the strict pass finds nothing.

    Handles two patterns:
      1. "ẹgbẹ̀rún" (thousand), optionally followed by a unit/ten/hundred
         word as a multiplier — e.g. "ẹgbẹ̀rún méjì" -> 2000, bare
         "ẹgbẹ̀rún" -> 1000 (only when nothing follows it — see below).
      2. A single unit/ten/hundred word on its own — e.g. "mẹ́ta" -> 3,
         "ọgọ́rùn-ún" -> 100.

    Only ever returns a value it can trace to an exact vocabulary match —
    never a guess at a compound number outside that vocabulary. This
    specifically means: if the thousand-word is followed by a word that
    ISN'T a recognized multiplier, this returns None rather than silently
    assuming "no multiplier" (= 1000) — an earlier version of this
    function did exactly that and got the real Phase 7 finance sentence
    wrong (returned 1000 instead of 10000 because "mẹwa" wasn't yet in
    the vocabulary and the unrecognized word was silently ignored rather
    than treated as "possibly a multiplier I don't know"). The "bare
    1000" reading is only trusted when the thousand-word is the very last
    word in the message, i.e. there's genuinely nothing to be ambiguous
    about.
    """
    words = _normalize(text).split(" ")
    result = _parse_with_vocab(words, _THOUSAND_WORD, _MULTIPLIER_VOCAB)
    if result is not None:
        return result

    loose_words = [_strip_underdots(w) for w in words]
    return _parse_with_vocab(loose_words, _THOUSAND_WORD_LOOSE, _MULTIPLIER_VOCAB_LOOSE)


def _self_test() -> None:
    """Validated against all 14 real cases from
    phase2-asr/scripts/numeral_parsing_test.py — the same words, the same
    expected values, ground-truthed there against real model output."""
    cases = [
        ("Mo ná náírà mẹ́ta lórí oúnjẹ", 3),
        ("Mo ná náírà mẹ́wàá lórí oúnjẹ", 10),
        ("Mo ná náírà ogún lórí oúnjẹ", 20),
        ("Mo ná náírà ọgbọ̀n lórí oúnjẹ", 30),
        ("Mo ná náírà ogójì lórí oúnjẹ", 40),
        ("Mo ná náírà àádọ́ta lórí oúnjẹ", 50),
        ("Mo ná náírà ọgọ́ta lórí oúnjẹ", 60),
        ("Mo ná náírà àádọ́rin lórí oúnjẹ", 70),
        ("Mo ná náírà ọgọ́rùn-ún lórí oúnjẹ", 100),
        ("Mo ná náírà igba lórí oúnjẹ", 200),
        ("Mo ná náírà ẹgbẹ̀rún kan lórí oúnjẹ", 1000),
        ("Mo ná náírà ẹgbẹ̀rún méjì lórí oúnjẹ", 2000),
        ("Mo ná náírà ẹgbẹ̀rún márùn-ún lórí oúnjẹ", 5000),
        ("Mo ná náírà ẹgbẹ̀rún mẹ́wàá lórí oúnjẹ", 10000),
        # A few real, unparseable-by-design cases — must return None, not a guess.
        ("Bawo ni? Se daadaa ni?", None),
        ("Mo ná owó lórí ọkọ̀ irin-àjò", None),  # amount not stated at all
    ]
    n_pass = 0
    for text, expected in cases:
        got = parse_yoruba_number(text)
        status = "OK" if got == expected else "FAIL"
        if status == "OK":
            n_pass += 1
        print(f"[{status}] {text!r} -> expected {expected}, got {got}")
    print(f"\n{n_pass}/{len(cases)} passed")
    assert n_pass == len(cases), "self-test failed — see FAIL lines above"


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("Usage: python3 app/yoruba_numerals.py --self-test")
