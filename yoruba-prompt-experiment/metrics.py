"""
Heuristic, automatic metrics for the Yoruba system-prompt experiment.

These are NOT a substitute for native-speaker judgment — they are cheap,
reproducible proxies that let us flag gross code-switching and give a rough
fluency signal before a human rates naturalness/accuracy. See README.md for
how the human rating step slots in.
"""
import re
import unicodedata

# Common English function/connector words. If these appear as whole tokens in
# a response that's supposed to be 100% Yoruba, that's a strong code-switch
# signal — native Yoruba text doesn't need "the", "however", "so", etc.
ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "so", "because", "however", "therefore", "although",
    "if", "then", "when", "while", "since", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "it",
    "you", "your", "yours", "i", "we", "they", "he", "she", "them", "his",
    "her", "its", "our", "my", "me", "us", "not", "no", "yes", "can", "will",
    "would", "should", "could", "may", "might", "must", "do", "does", "did",
    "have", "has", "had", "also", "just", "very", "really", "here", "there",
    "first", "second", "third", "step", "note", "important", "example",
    "additionally", "furthermore", "overall", "finally", "conclusion",
}

# Characters that only appear in Yoruba orthography (underdotted vowels/
# consonant, tone marks on vowels). Their presence/density is a proxy for
# whether the model is writing "real" diacritized Yoruba vs. flattened
# ASCII-only text (a common degraded mode).
YORUBA_SPECIAL_CHARS = set("ẹẸọỌṣṢáÁàÀéÉèÈíÍìÌóÓòÒúÚùÙńŃ")

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def code_switch_rate(text: str) -> dict:
    """Fraction of tokens that are recognizable English function words.

    Single-character tokens ("a", "i") are excluded from matching: Yoruba
    orthography uses standalone single vowels as real words/particles (e.g.
    "a" = "we"), so a bare "a" or "i" is not a reliable code-switch signal —
    counting them produced false positives on genuinely fluent Yoruba text.
    """
    tokens = _tokenize(text.lower())
    if not tokens:
        return {"rate": 0.0, "hits": [], "token_count": 0}
    hits = [t for t in tokens if len(t) >= 2 and t in ENGLISH_FUNCTION_WORDS]
    return {
        "rate": round(len(hits) / len(tokens), 4),
        "hits": hits,
        "token_count": len(tokens),
    }


def ascii_only_word_ratio(text: str) -> float:
    """
    Fraction of alphabetic tokens (len >= 3) containing no Yoruba diacritics
    and no underdotted letters. High values can indicate the model is writing
    "flattened" Yoruba (ASCII substitutes) rather than proper orthography —
    itself a fluency/naturalness signal, though not proof of code-switching
    (some correct Yoruba words are diacritic-free).
    """
    tokens = [t for t in _tokenize(text) if len(t) >= 3]
    if not tokens:
        return 0.0
    plain = [t for t in tokens if not any(ch in YORUBA_SPECIAL_CHARS for ch in t)]
    return round(len(plain) / len(tokens), 4)


def diacritic_density(text: str) -> float:
    """Yoruba special characters per 100 alphabetic characters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    special = [ch for ch in letters if ch in YORUBA_SPECIAL_CHARS]
    return round(100 * len(special) / len(letters), 2)


def non_latin_script_chars(text: str) -> int:
    """
    Count characters outside Latin script (excluding whitespace/punctuation/
    digits). A non-zero count on a Yoruba task is almost always a bug/error
    (e.g. the model switching to another script entirely) rather than a
    subtle fluency issue, so it's reported separately from code-switching.
    """
    count = 0
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "LATIN" not in name:
            count += 1
    return count


def basic_stats(text: str) -> dict:
    tokens = _tokenize(text)
    return {
        "char_count": len(text),
        "word_count": len(tokens),
        "sentence_count": len(re.findall(r"[.!?]+", text)) or 1,
    }


def compute_all_metrics(text: str) -> dict:
    cs = code_switch_rate(text)
    return {
        "code_switch_rate": cs["rate"],
        "code_switch_hits": cs["hits"],
        "ascii_only_word_ratio": ascii_only_word_ratio(text),
        "diacritic_density_per_100_chars": diacritic_density(text),
        "non_latin_script_char_count": non_latin_script_chars(text),
        **basic_stats(text),
    }
