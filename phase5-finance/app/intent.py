"""
Parses a Yorùbá (or English) natural-language financial statement into one
of four structured intents using Claude via Bedrock.

Deliberately does NOT use `output_config.format` (Anthropic's structured-
output / JSON-schema-enforcement param). That param returned a 400 ("Extra
inputs are not permitted") on this project's Bedrock account/SDK combo when
tried for the LLM-judge in yoruba-prompt-experiment/llm_judge.py — same
account, same client library, so the same failure would very likely recur
here. Prompting for plain JSON and parsing defensively is the proven-working
pattern from that fix; reused here rather than re-discovering the same bug.

Also cross-checks any extracted `amount` against `yoruba_numerals.py`, a
small deterministic Yorùbá-numeral parser — a systematic test
(phase2-asr/scripts/numeral_parsing_test.py) found this same model only
correctly parses 5/14 (36%) of real Yorùbá number words, confidently and
silently. See PLAN.md's "Yorùbá-numeral guardrail" section.
"""
from __future__ import annotations

import json
import re

from .yoruba_numerals import parse_yoruba_number

INTENT_SYSTEM_PROMPT = """You are a financial-transaction parser for a Yorùbá-language voice \
finance tracker. The user will describe a financial event or ask about their finances, in \
Yorùbá or English. Classify their message into exactly ONE of these intents:

  add_expense   — they spent money. Extract: amount (number, in Naira), category (one short \
                  English word: food, transport, rent, utilities, health, shopping, other), \
                  description (short, in the message's own language).
  add_income    — they received money. Extract: amount (number, in Naira), category (salary, \
                  gift, business, other), description.
  get_balance   — they're asking how much money they have / their current balance. No fields \
                  needed beyond the intent itself.
  get_summary   — they're asking for a spending/income breakdown or summary. No fields needed.
  unclear       — the message doesn't clearly match any of the above. Include a `reply_yoruba` \
                  field with a short, polite Yorùbá message asking them to clarify.

Respond with ONLY a single JSON object, no markdown fences, no prose, with exactly these keys:
{"intent": "...", "amount": <number or null>, "category": "<string or null>", \
"description": "<string or null>", "reply_yoruba": "<string or null>"}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

VALID_INTENTS = {"add_expense", "add_income", "get_balance", "get_summary", "unclear"}


def _extract_json(text: str) -> dict:
    match = _JSON_OBJECT_RE.search(text)
    candidate = match.group(0) if match else text
    return json.loads(candidate)


def parse_intent(client, model: str, user_message: str) -> dict:
    """
    Returns a dict with keys: intent, amount, category, description,
    reply_yoruba. On any parse failure, falls back to intent="unclear" with
    a generic Yorùbá clarification message rather than raising — a finance
    app should never crash on a weird utterance, it should ask again.
    """
    fallback = {
        "intent": "unclear", "amount": None, "category": None, "description": None,
        "reply_yoruba": "Má bìnú, n kò yé mi. Ṣe o lè sọ ọ́ ní ọ̀nà mìíràn?",
    }

    response = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0,  # structured extraction, not creative generation — was
        # unset (API default 1.0) until this was found live, 2026-08-15: 3 calls
        # with IDENTICAL input text (confirmed via a fixed TTS seed removing all
        # upstream ASR variance) returned 3 different amounts. Not an ASR/numeral
        # problem — the LLM call itself was non-deterministic. See PLAN.md.
        system=INTENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    if response.stop_reason == "refusal":
        return fallback

    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return fallback

    if parsed.get("intent") not in VALID_INTENTS:
        return fallback

    for key in ("amount", "category", "description", "reply_yoruba"):
        parsed.setdefault(key, None)

    # Only add_expense/add_income actually use `amount` — checking it for
    # get_balance/get_summary/unclear would risk "correcting" an amount
    # that was never meant to be extracted in the first place.
    if parsed["intent"] in ("add_expense", "add_income"):
        parsed["amount_verification"] = verify_amount(user_message, parsed["amount"])
        if parsed["amount_verification"]["corrected"]:
            parsed["amount"] = parsed["amount_verification"]["deterministic_amount"]
    else:
        parsed["amount_verification"] = {"checked": False, "deterministic_amount": None,
                                          "match": None, "corrected": False}

    return parsed


def verify_amount(user_message: str, llm_amount) -> dict:
    """Cross-checks the LLM's extracted amount against a deterministic
    parse of the same message. Returns a dict describing the outcome —
    never silently drops information, so callers (main.py) can decide
    what to tell the user.

    `deterministic_amount` is only ever a value the parser can trace to
    an exact, ground-truthed vocabulary match (see yoruba_numerals.py) —
    when it's not None and disagrees with the LLM, the deterministic
    value is trusted and `corrected` is True, since a disagreement means
    the LLM is wrong, not the deterministic check (it only fires on
    words it's provably right about). When the deterministic parser
    can't find a recognized number word at all, `checked` is False and
    the LLM's amount is used as-is — this guardrail narrows the known
    failure cases, it doesn't claim to catch every possible one.
    """
    deterministic_amount = parse_yoruba_number(user_message)
    if deterministic_amount is None:
        return {"checked": False, "deterministic_amount": None,
                "match": None, "corrected": False}

    match = (llm_amount is not None) and (float(deterministic_amount) == float(llm_amount))
    return {
        "checked": True,
        "deterministic_amount": deterministic_amount,
        "match": match,
        "corrected": not match,
    }
