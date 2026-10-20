"""
LLM-as-judge evaluation: faithfulness and groundedness scoring.

Complements the cheap heuristics in metrics.py (code-switching, diacritic
density) and the human-rating CSV (fluency/naturalness/task_accuracy) with
an automated judgment on two dimensions that are hard to detect with regex:

  - faithfulness: does the response avoid inventing facts, figures, or
    claims that are not supportable by the reference material (or, when
    there's no reference, by common knowledge / the prompt itself)?
  - groundedness: for tasks with a source text or ground-truth answer
    (factual QA, summarization, arithmetic reasoning), does the response
    actually rest on that material rather than drifting from it?

The judge is deliberately instructed in English and reads the Yoruba
response + Yoruba input + (optional) English reference — judge-language
choice doesn't bias the thing being measured (the *generation* language
behavior), and English instructions make the judge itself more reliable.

This makes one extra API call per (item, condition) — it's opt-in via
`experiment.py --with-judge` because of the added cost/latency.

NOTE: this deliberately does NOT use `output_config.format` (structured
outputs / JSON schema enforcement). That param 400'd on this Bedrock
account/SDK combo ("output_config.format: Extra inputs are not permitted")
even though it's documented as supported — rather than depend on a feature
that isn't reliably available across every Bedrock account, the judge is
prompted to emit plain JSON and we parse it defensively.
"""
from __future__ import annotations

import json
import re

JUDGE_SYSTEM_PROMPT = """You are a careful, skeptical evaluator of AI-generated text. \
You will be shown a task given to an AI assistant (in Yoruba), the assistant's response \
(in Yoruba), and optionally reference material describing what is factually correct or \
what the response should stay faithful to. You do not need to read Yoruba fluently to judge \
faithfulness and groundedness — focus on whether claims, numbers, names, and facts in the \
response are supportable, and flag anything that looks fabricated, invented, or contradicted \
by the reference. If you are unsure whether a specific Yoruba phrase constitutes a factual \
claim, err toward not penalizing it.

Respond with ONLY a single JSON object — no markdown code fences, no prose before or after —
with exactly these keys:
{
  "faithfulness_score": <integer 1-5, 5 = no fabricated/unsupported claims, 1 = substantially made up>,
  "groundedness_score": <integer 1-5, 5 = fully grounded in the reference/task, 1 = diverges from it>,
  "hallucination_detected": <true or false>,
  "justification": "<one or two sentences in English>"
}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Judges sometimes wrap JSON in markdown fences or add stray prose
    despite instructions not to — pull out the first {...} block and parse
    that rather than requiring the whole response to be bare JSON."""
    match = _JSON_OBJECT_RE.search(text)
    candidate = match.group(0) if match else text
    return json.loads(candidate)


def build_judge_user_message(task_input: str, response_text: str, reference: str | None) -> str:
    parts = [
        f"TASK GIVEN TO THE ASSISTANT (Yoruba):\n{task_input}",
        f"\nASSISTANT'S RESPONSE (Yoruba):\n{response_text}",
    ]
    if reference:
        parts.append(f"\nREFERENCE MATERIAL (ground truth / source the response should be "
                      f"faithful to):\n{reference}")
    else:
        parts.append(
            "\nNo reference material was provided for this task (it's an open-ended "
            "task — advice, opinion, creative writing, or a how-to). Judge faithfulness "
            "as: does the response avoid stating specific, checkable facts it can't "
            "support (e.g. invented statistics, fake sources, false claims about the "
            "world)? Judge groundedness as: does the response actually address what "
            "was asked, without wandering into unrelated or invented territory?"
        )
    parts.append(
        "\nScore faithfulness_score and groundedness_score on a 1-5 integer scale."
    )
    return "\n".join(parts)


def judge_response(client, model: str, task_input: str, response_text: str, reference: str | None) -> dict:
    """Call the model as a judge and return the parsed schema dict.

    `client` is any object exposing `.messages.create(...)` compatible with
    the Anthropic Messages API (first-party client or AnthropicBedrock —
    same interface).
    """
    if not response_text.strip():
        return {
            "faithfulness_score": None,
            "groundedness_score": None,
            "hallucination_detected": None,
            "justification": "Skipped — empty response (likely a refusal).",
        }

    user_message = build_judge_user_message(task_input, response_text, reference)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    if response.stop_reason == "refusal":
        return {
            "faithfulness_score": None,
            "groundedness_score": None,
            "hallucination_detected": None,
            "justification": "Judge call was refused.",
        }

    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return {
            "faithfulness_score": None,
            "groundedness_score": None,
            "hallucination_detected": None,
            "justification": f"Judge output was not valid JSON: {text[:200]!r}",
        }

    required = {"faithfulness_score", "groundedness_score", "hallucination_detected", "justification"}
    missing = required - parsed.keys()
    if missing:
        return {
            "faithfulness_score": parsed.get("faithfulness_score"),
            "groundedness_score": parsed.get("groundedness_score"),
            "hallucination_detected": parsed.get("hallucination_detected"),
            "justification": parsed.get("justification", f"Judge JSON missing keys: {missing}"),
        }
    return parsed
