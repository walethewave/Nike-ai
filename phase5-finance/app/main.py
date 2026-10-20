"""
Phase 5 prototype: a Yorùbá-language finance tracker built on Phase 3's
conversational layer. Text-first (same reasoning as Phase 3 — prove the
intent-parsing + DB loop works before adding the ASR/TTS layers on top;
Phase 3's app/asr.py and app/tts.py are directly reusable here once this
loop is solid, this app deliberately doesn't duplicate them).

Run:
    uvicorn app.main:app --reload --port 8001
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import db, intent

app = FastAPI(title="Naija Voice AI — Phase 5 Finance Tracker Prototype")

_client = None
MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")


def get_client():
    global _client
    if _client is None:
        import anthropic
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        _client = anthropic.AnthropicBedrock(
            aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
            aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
            aws_region=region,
        )
    return _client


class MessageRequest(BaseModel):
    text: str


class MessageResponse(BaseModel):
    intent: str
    reply: str
    balance_after: Optional[float] = None
    amount_corrected: bool = False  # True if the Yorùbá-numeral guardrail overrode the LLM's amount


def build_reply(parsed: dict, balance: float) -> str:
    """Yorùbá confirmation message per intent. Kept separate from the LLM
    call so it's deterministic — the number in the reply must always match
    what's actually in the database, not whatever the model felt like
    saying, which matters a lot more for a finance app than a chat app.

    `parsed['amount']` has already been auto-corrected upstream (see
    app/intent.py's verify_amount()) if the LLM's extracted amount
    disagreed with the deterministic Yorùbá-numeral guardrail — the
    correction isn't silent, it's surfaced in the reply below, so the
    user can see and object if the correction is itself wrong."""
    intent_name = parsed["intent"]
    correction_note = ""
    verification = parsed.get("amount_verification") or {}
    if verification.get("corrected"):
        correction_note = (
            f"(Mo tún iye owó ṣe sí ₦{parsed['amount']:,.2f} — mo gbọ́ nọ́mbà tí kò tọ́ ní àkọ́kọ́.) "
        )

    if intent_name == "add_expense":
        return (f"{correction_note}Mo ti fi kọ̀wé pé o ná ₦{parsed['amount']:,.2f} lórí {parsed['category']}. "
                f"Owó tí ó kù nísisìyí ni ₦{balance:,.2f}.")
    if intent_name == "add_income":
        return (f"{correction_note}Mo ti fi kọ̀wé pé o gba ₦{parsed['amount']:,.2f} láti ọ̀dọ̀ {parsed['category']}. "
                f"Owó tí ó kù nísisìyí ni ₦{balance:,.2f}.")
    if intent_name == "get_balance":
        return f"Owó tí ó kù lọ́wọ́ rẹ nísisìyí ni ₦{balance:,.2f}."
    if intent_name == "get_summary":
        summary = db.get_summary()
        return (f"Ìsọdipúpọ̀ owó rẹ: o gba ₦{summary['income_total']:,.2f} ({summary['income_count']} "
                f"ìgbà), o sì ná ₦{summary['expense_total']:,.2f} ({summary['expense_count']} ìgbà). "
                f"Owó tí ó kù ni ₦{summary['balance']:,.2f}.")
    return parsed.get("reply_yoruba") or "Má bìnú, n kò yé mi."


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/finance/message", response_model=MessageResponse)
def handle_message(req: MessageRequest):
    client = get_client()
    try:
        parsed = intent.parse_intent(client, MODEL, req.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Intent parsing failed: {exc}") from exc

    intent_name = parsed["intent"]

    if intent_name == "add_expense":
        if parsed["amount"] is None:
            reply = "Mélòó ni o ná? Jọ̀wọ́ sọ iye owó náà."
            return MessageResponse(intent=intent_name, reply=reply)
        db.add_transaction(db.Transaction(
            kind="expense", amount=float(parsed["amount"]),
            category=parsed["category"], description=parsed["description"],
        ))
        balance = db.get_balance()
        return MessageResponse(
            intent=intent_name, reply=build_reply(parsed, balance), balance_after=balance,
            amount_corrected=parsed["amount_verification"]["corrected"],
        )

    if intent_name == "add_income":
        if parsed["amount"] is None:
            reply = "Mélòó ni o gbà? Jọ̀wọ́ sọ iye owó náà."
            return MessageResponse(intent=intent_name, reply=reply)
        db.add_transaction(db.Transaction(
            kind="income", amount=float(parsed["amount"]),
            category=parsed["category"], description=parsed["description"],
        ))
        balance = db.get_balance()
        return MessageResponse(
            intent=intent_name, reply=build_reply(parsed, balance), balance_after=balance,
            amount_corrected=parsed["amount_verification"]["corrected"],
        )

    if intent_name in ("get_balance", "get_summary"):
        balance = db.get_balance()
        return MessageResponse(intent=intent_name, reply=build_reply(parsed, balance), balance_after=balance)

    return MessageResponse(intent="unclear", reply=build_reply(parsed, db.get_balance()))
