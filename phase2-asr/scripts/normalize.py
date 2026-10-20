"""
Phase 2 prototype: tone-aware vs. tone-agnostic text normalization + WER/CER
scoring for Yorùbá ASR evaluation.

Why this exists (see RESEARCH_OPEN_WEIGHT_ASR.md §4): Yorùbá is tonal, and
its orthography marks tone with diacritics (á à é è í ì ó ò ú ù ń) plus
underdots that change vowel/consonant identity (ẹ ọ ṣ vs e o s — these are
NOT stylistic, "ọkọ" and "oko" are different words). A WER script that
doesn't normalize Unicode consistently will report inflated error rates
from encoding differences alone (NFC vs NFD), and a script that silently
strips diacritics will make every model look artificially good while hiding
the actual hard part of the task. This module makes both modes explicit and
requires you to report both, rather than picking whichever looks better.

Usage:
    python3 scripts/normalize.py --self-test
    python3 scripts/normalize.py --ref refs.txt --hyp hyps.txt
"""
from __future__ import annotations

import argparse
import unicodedata

# Yorùbá tone marks (combining or precomposed) and the underdot diacritic.
# Precomposed vowels with tone marks that appear after NFC normalization:
_TONE_MARKED_VOWELS = set("áàéèíìóòúùṣńẹọẸỌṢÁÀÉÈÍÌÓÒÚÙŃ")
# Underdot-bearing letters that change phoneme identity, not just tone —
# these must NEVER be stripped even in "tone-agnostic" mode, only the pure
# pitch-accent marks (acute/grave) should be strippable.
_UNDERDOT_LETTERS = {"ẹ": "e", "ọ": "o", "ṣ": "s", "Ẹ": "E", "Ọ": "O", "Ṣ": "S"}

# Acute (tone: high) and grave (tone: low) combining marks, and their
# precomposed-form base-vowel mapping for the small fixed vowel set.
_ACUTE_GRAVE_MAP = {
    "á": "a", "à": "a", "é": "e", "è": "e", "í": "i", "ì": "i",
    "ó": "o", "ò": "o", "ú": "u", "ù": "u", "ń": "n",
    "Á": "A", "À": "A", "É": "E", "È": "E", "Í": "I", "Ì": "I",
    "Ó": "O", "Ò": "O", "Ú": "U", "Ù": "U", "Ń": "N",
}


def normalize_text(text: str, mode: str) -> str:
    """
    mode="tone_sensitive": NFC-normalize only. Diacritics (tone marks AND
        underdots) are preserved and scored as real characters — this is
        the honest, harder number.
    mode="tone_agnostic": NFC-normalize, then strip ONLY acute/grave tone
        marks (pitch accent). Underdots are still preserved, because ẹ/ọ/ṣ
        vs e/o/s is a phoneme distinction, not a tone distinction — folding
        those together would be a different, larger, less honest kind of
        stripping than "ignore tone."
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower() if mode == "tone_agnostic" else text.strip()

    if mode == "tone_sensitive":
        return text
    if mode == "tone_agnostic":
        return "".join(_ACUTE_GRAVE_MAP.get(ch, ch) for ch in text)
    raise ValueError(f"Unknown normalization mode: {mode!r}")


def compute_wer_cer(references: list[str], hypotheses: list[str], mode: str) -> dict:
    import jiwer

    norm_refs = [normalize_text(r, mode) for r in references]
    norm_hyps = [normalize_text(h, mode) for h in hypotheses]

    wer = jiwer.wer(norm_refs, norm_hyps)
    cer = jiwer.cer(norm_refs, norm_hyps)
    return {"mode": mode, "wer": round(wer, 4), "cer": round(cer, 4), "n": len(references)}


def evaluate_both_modes(references: list[str], hypotheses: list[str]) -> dict:
    """The function Phase 2's benchmark harness should actually call — always
    reports both modes, so a model can't be scored favorably by accident."""
    return {
        "tone_sensitive": compute_wer_cer(references, hypotheses, "tone_sensitive"),
        "tone_agnostic": compute_wer_cer(references, hypotheses, "tone_agnostic"),
    }


def _self_test():
    """No network/model needed — proves the normalization logic itself is
    correct against hand-checked Yorùbá examples."""
    cases = [
        # (input, tone_sensitive_expected, tone_agnostic_expected)
        ("Ọmọ mi ń lọ sí ilé", "Ọmọ mi ń lọ sí ilé", "ọmọ mi n lọ si ile"),
        ("ọkọ", "ọkọ", "ọkọ"),   # underdot preserved in BOTH modes — different word from "oko"
        ("oko", "oko", "oko"),
        ("Ẹ ṣé", "Ẹ ṣé", "ẹ ṣe"),  # underdot preserved, tone mark stripped
    ]
    failures = 0
    for text, expected_sensitive, expected_agnostic in cases:
        got_sensitive = normalize_text(text, "tone_sensitive")
        got_agnostic = normalize_text(text, "tone_agnostic")
        ok_s = got_sensitive == expected_sensitive
        ok_a = got_agnostic == expected_agnostic
        status = "OK" if (ok_s and ok_a) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] {text!r}")
        print(f"    tone_sensitive: got={got_sensitive!r} expected={expected_sensitive!r} {'✓' if ok_s else '✗'}")
        print(f"    tone_agnostic:  got={got_agnostic!r} expected={expected_agnostic!r} {'✓' if ok_a else '✗'}")

    # Sanity check: ọkọ and oko must NOT collapse to the same string in
    # either mode — if they do, the underdot-preservation guarantee is broken.
    assert normalize_text("ọkọ", "tone_agnostic") != normalize_text("oko", "tone_agnostic"), \
        "BUG: underdot letters collapsed under tone_agnostic mode — this must never happen"

    # WER/CER smoke test on a tiny synthetic pair
    refs = ["Ọmọ mi ń lọ sí ilé"]
    hyps = ["Omo mi n lo si ile"]  # a diacritic-free ASR hypothesis
    result = evaluate_both_modes(refs, hyps)
    print("\nWER/CER on 1 synthetic pair (diacritic-free hypothesis vs diacritic-rich reference):")
    print(f"  tone_sensitive: WER={result['tone_sensitive']['wer']} CER={result['tone_sensitive']['cer']}")
    print(f"  tone_agnostic:  WER={result['tone_agnostic']['wer']} CER={result['tone_agnostic']['cer']}")
    assert result["tone_sensitive"]["wer"] >= result["tone_agnostic"]["wer"], \
        "BUG: tone_sensitive WER should be >= tone_agnostic WER for a diacritic-free hypothesis"

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures == 0


def main():
    parser = argparse.ArgumentParser(description="Yorùbá tone-aware normalization + WER/CER")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ref", help="Path to reference transcripts, one per line")
    parser.add_argument("--hyp", help="Path to hypothesis transcripts, one per line")
    args = parser.parse_args()

    if args.self_test:
        ok = _self_test()
        raise SystemExit(0 if ok else 1)

    if not args.ref or not args.hyp:
        parser.error("--ref and --hyp are required unless --self-test is given")

    with open(args.ref, encoding="utf-8") as f:
        refs = [line.strip() for line in f if line.strip()]
    with open(args.hyp, encoding="utf-8") as f:
        hyps = [line.strip() for line in f if line.strip()]
    if len(refs) != len(hyps):
        raise SystemExit(f"Mismatched line counts: {len(refs)} refs vs {len(hyps)} hyps")

    result = evaluate_both_modes(refs, hyps)
    print(f"n={result['tone_sensitive']['n']}")
    print(f"tone_sensitive: WER={result['tone_sensitive']['wer']}  CER={result['tone_sensitive']['cer']}")
    print(f"tone_agnostic:  WER={result['tone_agnostic']['wer']}  CER={result['tone_agnostic']['cer']}")


if __name__ == "__main__":
    main()
