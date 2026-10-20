"""
Domain router: given transcribed user text, decide whether this turn should
go to Phase 5's finance-tracker logic or Phase 3's general conversational
LLM. This is a DELIBERATE placeholder — a small keyword heuristic, not a
model call — so that routing doesn't cost a full LLM round trip before the
real handling even starts. It is honestly not robust: real Yorùbá/Igbo/Hausa
finance language varies far more than a fixed keyword list can capture.
Treat this as the seam where a better router (a fast, cheap intent
classification call, or a small fine-tuned classifier) plugs in later — see
PLAN.md Step 2.

Keyword lists cover Yorùbá, Igbo, Hausa, and English financial vocabulary
(amounts, spending/receiving verbs, "balance"/"summary" equivalents),
sourced from the phrases already proven to work in Phase 5's live tests
plus straightforward translation for the other two languages — the
Igbo/Hausa entries are NOT independently native-speaker-verified the way
the Yorùbá ones are (Phase 5's live tests), and should be checked before
relying on them for real Igbo/Hausa users (see Phase 6's native-speaker
testing step).
"""
from __future__ import annotations

import re
import unicodedata


def _strip_tone_marks(text: str) -> str:
    """Strip acute/grave tone marks but preserve underdots (ẹọṣ).
    This makes keyword matching robust to ASR diacritic variations."""
    # Map tone-marked vowels to plain (same logic as phase2-asr/normalize.py)
    tone_map = {
        "á": "a", "à": "a", "é": "e", "è": "e", "í": "i", "ì": "i",
        "ó": "o", "ò": "o", "ú": "u", "ù": "u", "ń": "n",
        "Á": "A", "À": "A", "É": "E", "È": "E", "Í": "I", "Ì": "I",
        "Ó": "O", "Ò": "O", "Ú": "U", "Ù": "U", "Ń": "N",
        # Underdot + tone combinations
        "ẹ́": "ẹ", "ẹ̀": "ẹ", "ọ́": "ọ", "ọ̀": "ọ",
    }
    text = unicodedata.normalize("NFC", text)
    return "".join(tone_map.get(ch, ch) for ch in text)

FINANCE_KEYWORDS = {
    "yoruba": [
        # Core money/currency terms (with ASR-robust variants)
        "owo", "owó", "náírà", "naira", "nara", "náàra", "náara",
        # Spending/receiving verbs
        "inawo", "ìnáwó", "ná ", "na ",  # "na" = spend (with space to avoid false positives)
        "gba owo", "gbà owó",
        # Query terms
        "elo", "ẹ́lòó", "melo",  # how much
        "akotan", "àkótán",  # summary
        "iye owo", "ìyè owó",  # amount of money
        # ASR-robust variants (common mishearings from Phase 7 testing)
        "igberun", "ẹgbẹrun", "ẹgbẹ̀rùn", "ègbànrùn", "ègbàn",  # thousand variants
        "ogorun", "ọgọrun",  # hundred
    ],
    "igbo": [
        "ego", "nnukwu ego", "mmefu", "mefuru", "nweta ego", "ole ego", "nchikota",
    ],
    "hausa": [
        "kudi", "kuɗi", "naira", "nara", "kashe", "samu kudi", "nawa", "taƙaice",
    ],
    "english": [
        "money", "naira", "nara", "spent", "spend", "earned", "income", "expense",
        "balance", "summary", "budget", "paid", "bought", "salary",
    ],
}


def route(text: str, lang: str = "yoruba") -> str:
    """Returns "finance" or "chat". Case-insensitive WORD-BOUNDARY match
    against the language's keyword list, plus English (mixed-language input
    is common in real Nigerian speech, so always also check English terms
    regardless of the primary language).

    Matching is done on TONE-STRIPPED text to handle ASR diacritic variations
    (e.g., "náàra" vs "naira" vs "nara" all match). Underdots (ẹọṣ) are
    preserved since they change phoneme identity.

    Word-boundary matching (not naive substring) matters concretely here:
    an earlier substring-based version of this function misrouted "Se o
    ranti ohun ti mo beere lowo re?" ("Do you remember what I asked you?")
    to "finance", because "lowo" (a common Yorùbá preposition meaning
    roughly "from/at") contains "owo" (money) as a substring. Caught by
    testing against Phase 5's own real live-test phrases, not a
    hypothetical — see PLAN.md.
    """
    # Normalize: lowercase + strip tone marks for robust matching
    normalized = _strip_tone_marks(text.lower())
    keywords = FINANCE_KEYWORDS.get(lang, []) + FINANCE_KEYWORDS["english"]
    
    for kw in keywords:
        kw = _strip_tone_marks(kw.strip().lower())
        if not kw:
            continue
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", normalized):
            return "finance"
    return "chat"
