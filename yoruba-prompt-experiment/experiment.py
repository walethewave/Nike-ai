"""
Run the Yoruba system-prompt experiment across THREE systems and compare:

  - claude_haiku : Claude Haiku 4.5 (Amazon Bedrock, boto3)
  - kimi         : Kimi (Moonshot AI)
  - glm          : GLM (Zhipu AI / Z.ai)

For every test item, every ready provider is called once with system prompt
A (English) and once with system prompt B (Yoruba), using the exact same
user input. All (provider, condition) pairs for an item are fired
CONCURRENTLY via a thread pool — so Claude Haiku, Kimi, and GLM (and both
conditions of each) are all in flight at once instead of one after another.
Each provider has its own cached client instance (see providers.py), so
there's no shared connection/state across the parallel calls.

Faithfulness/groundedness judging (if enabled) always runs on Claude via
Bedrock regardless of which provider generated the response, so every
system is scored by the same independent evaluator. Judge calls for an item
also run concurrently once all the generations for that item are in.

A provider is silently skipped (with a warning) if its credentials aren't
set in .env — so you can run e.g. just Claude while you're still waiting on
Kimi/GLM API keys.

Usage:
    python experiment.py                          # run all 3 providers, all items
    python experiment.py --providers claude_haiku  # just one provider
    python experiment.py --providers kimi,glm      # just two
    python experiment.py --limit 3                 # quick smoke test, first 3 items
    python experiment.py --with-judge               # also score faithfulness/groundedness
    python experiment.py --max-workers 4            # cap concurrent in-flight calls
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from config import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_YO, TEST_ITEMS  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from llm_judge import judge_response  # noqa: E402
import providers as prov  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
# Verified live against Bedrock: like Haiku 4.5, Opus 5 is INFERENCE_PROFILE-only
# on this account — needs the "us." inference-profile prefix, not the bare ID.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL_ID", "us.anthropic.claude-opus-5")
ALL_PROVIDER_KEYS = list(prov.PROVIDERS.keys())

CONDITIONS = [
    ("A", "condition_A_english_prompt", SYSTEM_PROMPT_EN),
    ("B", "condition_B_yoruba_prompt", SYSTEM_PROMPT_YO),
]

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(msg)


MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (2, 5, 10)


def generate_and_score(provider_key: str, system_prompt: str, user_input: str) -> dict:
    """Calls the provider with its own retry loop on top of whatever the SDK
    already does — under concurrent load this sandbox has shown transient
    connection drops / timeouts that the SDK's built-in retries didn't fully
    absorb, so we retry at the application level too before giving up."""
    display = prov.PROVIDERS[provider_key].display_name
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = prov.generate(provider_key, system_prompt, user_input)
            out = {"text": result.text, "stop_reason": result.stop_reason, "usage": result.raw_usage}
            out["metrics"] = compute_all_metrics(out["text"])
            return out
        except Exception as exc:  # each provider fails differently; never let one kill the run
            last_exc = exc
            if attempt < MAX_RETRIES:
                base_delay = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
                delay = base_delay + random.uniform(0, base_delay)  # jitter so parallel retries desync
                log(f"    !! [{display}] attempt {attempt}/{MAX_RETRIES} failed ({exc}); retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                log(f"    !! [{display}] all {MAX_RETRIES} attempts failed: {exc}")

    return {"text": "", "stop_reason": "error", "usage": {}, "error": str(last_exc),
            "metrics": compute_all_metrics("")}


JUDGE_MAX_CONCURRENCY = 2  # all judge calls hit the same single model endpoint,
# unlike generation calls which spread across providers — keep this low so
# retries don't collide in a synchronized thundering herd against Bedrock.


def judge_one(judge_client, task_input: str, response_text: str, reference) -> dict:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return judge_response(judge_client, JUDGE_MODEL, task_input, response_text, reference)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                base_delay = RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
                delay = base_delay + random.uniform(0, base_delay)  # jitter so parallel retries desync
                log(f"    !! judge attempt {attempt}/{MAX_RETRIES} failed ({exc}); retrying in {delay:.1f}s")
                time.sleep(delay)
    return {"faithfulness_score": None, "groundedness_score": None,
            "hallucination_detected": None, "justification": f"Judge call failed: {last_exc}"}


def resolve_ready_providers(requested_keys: list) -> list:
    ready = []
    for key in requested_keys:
        provider = prov.PROVIDERS[key]
        ok, note = provider.is_ready()
        if not ok:
            print(f"SKIPPING provider '{key}' ({provider.display_name}): {note}", file=sys.stderr)
            continue
        if note:
            print(f"NOTE for provider '{key}' ({provider.display_name}): {note}", file=sys.stderr)
        ready.append(key)
    return ready


def process_item(item: dict, ready_providers: list, judge_client, max_workers: int) -> dict:
    """Fire every (provider, condition) call for this item concurrently,
    then — once generation is done — fire every judge call concurrently."""
    reference = item.get("reference")
    generations = {}  # (provider_key, cond_letter) -> response dict

    gen_jobs = [
        (provider_key, cond_letter, cond_key, system_prompt)
        for provider_key in ready_providers
        for cond_letter, cond_key, system_prompt in CONDITIONS
    ]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(gen_jobs)) or 1) as pool:
        futures = {
            pool.submit(generate_and_score, provider_key, system_prompt, item["input"]): (provider_key, cond_letter)
            for provider_key, cond_letter, _cond_key, system_prompt in gen_jobs
        }
        for future in as_completed(futures):
            provider_key, cond_letter = futures[future]
            display = prov.PROVIDERS[provider_key].display_name
            log(f"    [{display}] condition {cond_letter} done")
            generations[(provider_key, cond_letter)] = future.result()

    if judge_client:
        judge_jobs = [
            (provider_key, cond_letter)
            for provider_key in ready_providers
            for cond_letter, _cond_key, _sp in CONDITIONS
            if generations[(provider_key, cond_letter)]["stop_reason"] != "error"
        ]
        with ThreadPoolExecutor(max_workers=min(JUDGE_MAX_CONCURRENCY, len(judge_jobs)) or 1) as pool:
            futures = {
                pool.submit(
                    judge_one, judge_client, item["input"],
                    generations[(provider_key, cond_letter)]["text"], reference,
                ): (provider_key, cond_letter)
                for provider_key, cond_letter in judge_jobs
            }
            for future in as_completed(futures):
                provider_key, cond_letter = futures[future]
                display = prov.PROVIDERS[provider_key].display_name
                log(f"    [{display}] judged condition {cond_letter}")
                generations[(provider_key, cond_letter)]["llm_judge"] = future.result()

    results = {}
    for provider_key in ready_providers:
        results[provider_key] = {
            cond_key: generations[(provider_key, cond_letter)]
            for cond_letter, cond_key, _sp in CONDITIONS
        }

    return {
        "id": item["id"],
        "category": item["category"],
        "input": item["input"],
        "reference": reference,
        "results": results,
    }


def run_experiment(provider_keys: list, limit: int = None, with_judge: bool = False, max_workers: int = 6) -> dict:
    items = TEST_ITEMS[:limit] if limit else TEST_ITEMS
    ready_providers = resolve_ready_providers(provider_keys)

    if not ready_providers:
        print(
            "No providers ready to run — check .env for missing credentials "
            "(AWS keys for claude_haiku, KIMI_API_KEY for kimi, GLM_API_KEY for glm).",
            file=sys.stderr,
        )
        sys.exit(1)

    judge_client = prov.build_judge_client() if with_judge else None

    run = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "providers_requested": provider_keys,
        "providers_run": ready_providers,
        "provider_display_names": {k: prov.PROVIDERS[k].display_name for k in ready_providers},
        "provider_models": {k: prov.PROVIDERS[k].model for k in ready_providers},
        "with_judge": with_judge,
        "judge_model": JUDGE_MODEL if with_judge else None,
        "max_workers": max_workers,
        "system_prompt_en": SYSTEM_PROMPT_EN,
        "system_prompt_yo": SYSTEM_PROMPT_YO,
        "items": [],
    }

    print(f"Running providers in parallel: {', '.join(prov.PROVIDERS[k].display_name for k in ready_providers)}")
    print(f"Items: {len(items)}  |  Judge: {'on (' + JUDGE_MODEL + ')' if with_judge else 'off'}  "
          f"|  max_workers: {max_workers}\n")

    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {item['id']} ({item['category']})")
        item_result = process_item(item, ready_providers, judge_client, max_workers)
        run["items"].append(item_result)

    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    return run


def main():
    parser = argparse.ArgumentParser(description="Yoruba system-prompt experiment: Claude Haiku 4.5 vs Kimi vs GLM")
    parser.add_argument(
        "--providers",
        default=",".join(ALL_PROVIDER_KEYS),
        help=f"Comma-separated provider keys to run. Available: {', '.join(ALL_PROVIDER_KEYS)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N test items")
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Also run the faithfulness/groundedness LLM-judge (extra API calls, always via Claude/Bedrock)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=6,
        help="Max concurrent in-flight API calls per item (default 6 = 3 providers x 2 conditions)",
    )
    args = parser.parse_args()

    provider_keys = [p.strip() for p in args.providers.split(",") if p.strip()]
    unknown = [p for p in provider_keys if p not in prov.PROVIDERS]
    if unknown:
        print(f"Unknown provider(s): {unknown}. Valid keys: {ALL_PROVIDER_KEYS}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    started = time.monotonic()
    run = run_experiment(provider_keys, limit=args.limit, with_judge=args.with_judge, max_workers=args.max_workers)
    elapsed = time.monotonic() - started

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Saved results for providers {run['providers_run']} to {out_path}")
    print("Run `python analyze.py` to see the cross-provider comparison report.")


if __name__ == "__main__":
    main()
