"""
Provider abstraction: a unified `generate()` call across three systems being
compared on the Yoruba system-prompt experiment — ALL THREE hosted directly
on Amazon Bedrock, under one set of AWS credentials:

  - claude_haiku : Claude Haiku 4.5   (Anthropic, via the Bedrock Messages-API path)
  - kimi         : Kimi K2.5          (Moonshot AI, via Bedrock's Converse API)
  - glm          : GLM 5              (Z.AI, via Bedrock's Converse API)

Claude goes through `anthropic.AnthropicBedrock` (Messages-API shaped
requests, so it gets Claude-specific things like `thinking`/`output_config`).
Kimi and GLM aren't Anthropic-shaped models, so they go through Bedrock's
provider-agnostic `bedrock-runtime.converse()` API instead — same AWS
credentials, different boto3 client, different request/response envelope.

All three model IDs were verified live against this account's
`bedrock:ListFoundationModels` (2026-07-28) — not guessed from docs:
  anthropic  -> us.anthropic.claude-haiku-4-5-20251001-v1:0  (INFERENCE_PROFILE-only)
  moonshotai -> moonshotai.kimi-k2.5                          (ON_DEMAND)
  zai        -> zai.glm-5                                     (ON_DEMAND)
If your account's model access differs, re-run that check rather than
trusting the defaults below — Bedrock's third-party model catalog varies
by account and changes over time.

Clients are cached per provider and built lazily behind a lock, so this
module is safe to call concurrently from multiple threads (used by
experiment.py to fire all 3 providers in parallel).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import anthropic

_CLIENT_CACHE: dict = {}
_CLIENT_CACHE_LOCK = threading.Lock()


@dataclass
class GenerationResult:
    text: str
    stop_reason: str
    raw_usage: dict


@dataclass
class Provider:
    key: str
    display_name: str
    kind: str  # "bedrock_claude" | "bedrock_converse"
    model: str

    def is_ready(self):
        """Returns (ok: bool, note: str). `note` is a warning even when ok=True
        (e.g. falling back to boto3's default credential chain). All current
        provider kinds run on Bedrock, so this is one shared AWS check."""
        if os.environ.get("AWS_ACCESS_KEY_ID", "").strip() and os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip():
            return True, ""
        return True, (
            "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY not set in .env; relying on boto3's "
            "default credential chain (shared credentials file, SSO, instance role, ...)"
        )


PROVIDERS: dict = {
    "claude_haiku": Provider(
        key="claude_haiku",
        display_name="Claude Haiku 4.5 (Amazon Bedrock)",
        kind="bedrock_claude",
        # Haiku 4.5 on Bedrock is INFERENCE_PROFILE-only — the bare
        # "anthropic.claude-haiku-4-5" style ID 400s with "on-demand
        # throughput isn't supported"; must use the region-prefixed profile.
        model=os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ),
    "kimi": Provider(
        key="kimi",
        display_name="Kimi K2.5 (Moonshot AI, via Bedrock)",
        kind="bedrock_converse",
        model=os.environ.get("KIMI_MODEL", "moonshotai.kimi-k2.5"),
    ),
    "glm": Provider(
        key="glm",
        display_name="GLM 5 (Z.AI, via Bedrock)",
        kind="bedrock_converse",
        model=os.environ.get("GLM_MODEL", "zai.glm-5"),
    ),
}


def _resolved_region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


def _boto3_session():
    import boto3

    return boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
        region_name=_resolved_region(),
    )


def _build_bedrock_client() -> "anthropic.AnthropicBedrock":
    return anthropic.AnthropicBedrock(
        aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
        aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
        aws_region=_resolved_region(),
    )


def _build_bedrock_runtime_client():
    """Raw boto3 bedrock-runtime client for the provider-agnostic Converse
    API — used for non-Anthropic models (Kimi, GLM) that don't speak the
    Messages API shape the anthropic SDK expects."""
    return _boto3_session().client("bedrock-runtime")


def build_judge_client():
    """The faithfulness/groundedness judge always runs on Claude via Bedrock,
    independent of whichever providers are under comparison, so the same
    evaluator scores every system on the same scale."""
    return _build_bedrock_client()


def get_client(provider_key: str):
    """Thread-safe, lazily-built, cached client per provider (double-checked
    locking) — safe to call concurrently when running providers in parallel."""
    if provider_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider_key]
    with _CLIENT_CACHE_LOCK:
        if provider_key in _CLIENT_CACHE:
            return _CLIENT_CACHE[provider_key]
        provider = PROVIDERS[provider_key]
        if provider.kind == "bedrock_claude":
            client = _build_bedrock_client()
        elif provider.kind == "bedrock_converse":
            client = _build_bedrock_runtime_client()
        else:
            raise ValueError(f"Unknown provider kind: {provider.kind}")
        _CLIENT_CACHE[provider_key] = client
        return client


# Claude models that don't support output_config.effort (400 error) and only
# support thinking via an explicit {"type": "enabled", "budget_tokens": N}
# (never "adaptive"/"disabled") — so for our thinking-off use case we just
# omit both params for these rather than the disabled+low-effort pattern
# used on current-tier models.
_OLDER_CLAUDE_TIER_MARKERS = ("haiku-4-5", "haiku-3", "sonnet-4-5", "sonnet-4-0", "opus-4-1", "opus-4-0")


def _is_older_claude_tier(model: str) -> bool:
    return any(marker in model for marker in _OLDER_CLAUDE_TIER_MARKERS)


def _call_claude_bedrock(client, model: str, system_prompt: str, user_input: str, max_tokens: int) -> GenerationResult:
    kwargs = {}
    if not _is_older_claude_tier(model):
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["output_config"] = {"effort": "low"}

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}],
        **kwargs,
    )
    if response.stop_reason == "refusal":
        return GenerationResult(text="", stop_reason="refusal", raw_usage=response.usage.model_dump())
    text = "".join(b.text for b in response.content if b.type == "text")
    return GenerationResult(text=text, stop_reason=response.stop_reason, raw_usage=response.usage.model_dump())


def _call_bedrock_converse(client, model: str, system_prompt: str, user_input: str, max_tokens: int) -> GenerationResult:
    """Bedrock's provider-agnostic Converse API — same request/response shape
    regardless of which non-Anthropic model is behind `model`."""
    response = client.converse(
        modelId=model,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_input}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])
    text = "".join(block.get("text", "") for block in content_blocks if "text" in block)
    stop_reason = response.get("stopReason", "unknown")
    usage = response.get("usage", {})
    return GenerationResult(text=text, stop_reason=stop_reason, raw_usage=usage)


def generate(provider_key: str, system_prompt: str, user_input: str, max_tokens: int = 1024) -> GenerationResult:
    provider = PROVIDERS[provider_key]
    client = get_client(provider_key)
    if provider.kind == "bedrock_claude":
        return _call_claude_bedrock(client, provider.model, system_prompt, user_input, max_tokens)
    elif provider.kind == "bedrock_converse":
        return _call_bedrock_converse(client, provider.model, system_prompt, user_input, max_tokens)
    raise ValueError(f"Unknown provider kind: {provider.kind}")
