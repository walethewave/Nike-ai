"""
LLM layer: Claude via Amazon Bedrock. This is the one component of the
Phase 3 pipeline that's fully real and testable right now (no local GPU,
no gated dataset, no separate model download) — it reuses the exact client
pattern already verified working in yoruba-prompt-experiment/providers.py.

System prompt language: the Yoruba-system-prompt experiment (see
yoruba-prompt-experiment/RESULTS.md) found no detectable difference between
English and Yoruba system prompts on Claude Haiku 4.5, Kimi, or GLM. We
default to English here for that reason — it's one less thing to get wrong
in translation, and the experiment says it doesn't cost anything.
"""
from __future__ import annotations

import os

SYSTEM_PROMPT = """You are a helpful, friendly conversational assistant speaking with a Yorùbá \
speaker. Always respond entirely in Yorùbá — fluent, natural, grammatically correct, with \
correct diacritics (underdots and tone marks) — regardless of what language the user's message \
arrives in. Keep responses conversational and reasonably concise: this is a spoken dialogue, not \
a written document, so avoid long lists or headers. Do not switch to English unless a term has \
no natural Yorùbá equivalent (a proper noun or brand name)."""


class ConversationTurn:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationSession:
    """In-memory turn history for one conversation. Not persisted — a real
    deployment needs a store (Redis/DB) keyed by session id; this is a
    prototype, so process memory is enough to prove the loop works."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: list[ConversationTurn] = []

    def add_user_turn(self, text: str):
        self.turns.append(ConversationTurn("user", text))

    def add_assistant_turn(self, text: str):
        self.turns.append(ConversationTurn("assistant", text))

    def messages(self) -> list[dict]:
        return [t.to_message() for t in self.turns]


def build_bedrock_client():
    """Same construction pattern as yoruba-prompt-experiment/providers.py —
    verified working against real Bedrock inference profiles."""
    import anthropic

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return anthropic.AnthropicBedrock(
        aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
        aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
        aws_region=region,
    )


# Verified live (2026-07-28, see yoruba-prompt-experiment): current-tier
# Claude models on Bedrock are often INFERENCE_PROFILE-only — needs the
# region-prefixed ID, not the bare "anthropic.claude-haiku-4-5" form.
DEFAULT_MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")


def generate_reply(client, session: ConversationSession, user_text: str, model: str = DEFAULT_MODEL) -> str:
    session.add_user_turn(user_text)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=session.messages(),
    )

    if response.stop_reason == "refusal":
        reply = "Má bìnú, n kò lè dáhùn ìbéèrè yìí."  # "Sorry, I can't answer that question."
        session.add_assistant_turn(reply)
        return reply

    reply = "".join(block.text for block in response.content if block.type == "text")
    session.add_assistant_turn(reply)
    return reply
