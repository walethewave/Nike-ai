"""
The REAL, live intent-parsing test: real Yorùbá finance utterances -> real
Claude call via Bedrock (app/intent.py's parse_intent) -> real SQLite writes
(app/db.py) for the two intents that produce one. Every stage is real; this
was blocked until now by the AWS credential issue documented in PLAN.md and
phase3-conversational/PLAN.md — this script is the first live re-test since
that got fixed.

Uses an isolated temp SQLite file (not phase5-finance/data/finance.db) so
running this doesn't pollute the app's real data.

Test phrases are hand-written for this test, covering all five intents.
They are NOT native-speaker-verified (same caveat as phase6/phase7's
non-Yorùbá keyword lists) — they're a reasonable attempt at natural Yorùbá,
not a linguistically certified test set.

Usage:
    python3 scripts/live_intent_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import db, intent  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

TEST_CASES = [
    {"label": "add_expense", "text": "Mo ná ẹgbẹ̀rún méjì fún oúnjẹ lánàá", "expect_intent": "add_expense"},
    {"label": "add_income", "text": "Mo gba ẹgbẹ̀rún mẹ́wàá gẹ́gẹ́ bí owó oṣù", "expect_intent": "add_income"},
    {"label": "get_balance", "text": "Èló ni mo ní lọ́wọ́ mi nísisìyí?", "expect_intent": "get_balance"},
    {"label": "get_summary", "text": "Ṣe àkótán ìnáwó mi fún mi", "expect_intent": "get_summary"},
    {"label": "unclear_offtopic", "text": "Ǹjẹ́ o mọ ibi tí mo fi bàtà mi sí?", "expect_intent": "unclear"},
    {"label": "add_expense_no_amount", "text": "Mo ná owó lórí ọkọ̀ irin-àjò", "expect_intent": "add_expense"},
]


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
    tmp_db = Path(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)
    print(f"Using isolated test DB: {tmp_db}\n")

    results = []
    for case in TEST_CASES:
        print(f"=== {case['label']}: {case['text']!r} ===")
        t0 = time.time()
        try:
            parsed = intent.parse_intent(client, MODEL, case["text"])
            error = None
        except Exception as exc:
            parsed = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.time() - t0, 2)

        record = {
            "label": case["label"],
            "input_text": case["text"],
            "expect_intent": case["expect_intent"],
            "parsed": parsed,
            "error": error,
            "elapsed_sec": elapsed,
        }

        if error:
            print(f"  FAILED: {error}")
        else:
            got = parsed["intent"]
            match = "OK" if got == case["expect_intent"] else "MISMATCH"
            print(f"  [{match}] intent={got} amount={parsed.get('amount')} category={parsed.get('category')}")

            if got in ("add_expense", "add_income") and parsed.get("amount") is not None:
                db.add_transaction(db.Transaction(
                    kind="expense" if got == "add_expense" else "income",
                    amount=float(parsed["amount"]),
                    category=parsed.get("category"),
                    description=parsed.get("description"),
                ), db_path=tmp_db)
                record["balance_after_write"] = db.get_balance(db_path=tmp_db)
                print(f"  DB write OK — balance now {record['balance_after_write']}")

        results.append(record)
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "live_intent_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in results if not r["error"] and r["parsed"]["intent"] == r["expect_intent"])
    print(f"\n{n_ok}/{len(results)} matched expected intent. Full log: {out_path}")

    tmp_db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
