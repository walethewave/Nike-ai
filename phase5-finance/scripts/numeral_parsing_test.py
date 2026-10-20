"""
Focused follow-up to the numeral-misparsing finding in live_intent_test.py
(2026-08-15): that test found ONE real case of a wrong amount ("ẹgbẹ̀rún
mẹ́wàá" / ten thousand parsed as 20,000) on clean, non-ASR text. This script
maps out WHERE the LLM's Yorùbá-numeral parsing breaks down, by holding the
sentence template fixed and varying only the number word, across simple
units, irregular subtractive tens (a known hard part of Yorùbá's base-20
counting system), hundreds, and thousands.

Number words below follow standard Yorùbá numeral formation. Not
independently native-speaker-verified — same caveat as live_intent_test.py.

Usage:
    python3 scripts/numeral_parsing_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import intent  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# (yoruba number word, expected numeric value, category of numeral)
NUMERAL_CASES = [
    ("mẹ́ta", 3, "unit"),
    ("mẹ́wàá", 10, "unit"),
    ("ogún", 20, "regular ten (2x10)"),
    ("ọgbọ̀n", 30, "regular ten (3x10)"),
    ("ogójì", 40, "regular ten (4x10)"),
    ("àádọ́ta", 50, "irregular subtractive ten (60-10)"),
    ("ọgọ́ta", 60, "regular ten (3x20)"),
    ("àádọ́rin", 70, "irregular subtractive ten (80-10)"),
    ("ọgọ́rùn-ún", 100, "hundred"),
    ("igba", 200, "irregular hundred (not 'ọgọ́rùn méjì')"),
    ("ẹgbẹ̀rún kan", 1000, "thousand (1x)"),
    ("ẹgbẹ̀rún méjì", 2000, "thousand (2x)"),
    ("ẹgbẹ̀rún márùn-ún", 5000, "thousand (5x)"),
    ("ẹgbẹ̀rún mẹ́wàá", 10000, "thousand x ten — the original failing case"),
]

SENTENCE_TEMPLATE = "Mo ná náírà {word} lórí oúnjẹ"  # "I spent {word} naira on food"


def get_client():
    import anthropic
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return anthropic.AnthropicBedrock(
        aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
        aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
        aws_region=region,
    )


def main():
    client = get_client()
    results = []

    for word, expected, category in NUMERAL_CASES:
        sentence = SENTENCE_TEMPLATE.format(word=word)
        print(f"=== [{category}] {word!r} (expect {expected}) ===")
        print(f"    {sentence!r}")
        t0 = time.time()
        try:
            parsed = intent.parse_intent(client, MODEL, sentence)
            error = None
        except Exception as exc:
            parsed = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.time() - t0, 2)

        got = parsed.get("amount") if parsed else None
        correct = (got == expected)
        status = "OK" if correct else ("PARSE_FAIL" if error else "WRONG")
        print(f"    [{status}] got={got}\n")

        results.append({
            "word": word, "category": category, "sentence": sentence,
            "expected": expected, "got": got, "correct": correct,
            "raw_intent": parsed.get("intent") if parsed else None,
            "error": error, "elapsed_sec": elapsed,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "numeral_parsing_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in results if r["correct"])
    print(f"\n{n_ok}/{len(results)} correct.")
    print("\nWrong cases:")
    for r in results:
        if not r["correct"]:
            print(f"  [{r['category']}] {r['word']!r}: expected {r['expected']}, got {r['got']}")

    print(f"\nFull log: {out_path}")


if __name__ == "__main__":
    main()
