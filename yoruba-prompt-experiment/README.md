# Yoruba vs English System Prompt Experiment

**Research question:** For Yoruba-language tasks, does writing the *system prompt*
in Yoruba (instead of English) produce more consistent Yoruba output, less
code-switching, and more natural/faithful/grounded responses — and does the
answer to that hold across different model families?

This compares **three systems** on the same task set, **all hosted directly
on Amazon Bedrock under one set of AWS credentials** — no separate
Moonshot/Zhipu API keys needed, verified live against `bedrock:ListFoundationModels`:

| Provider key | System | Bedrock model ID | API path |
|---|---|---|---|
| `claude_haiku` | Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Anthropic Messages API (via `anthropic[bedrock]`) |
| `kimi` | Kimi K2.5 (Moonshot AI) | `moonshotai.kimi-k2.5` | Bedrock Converse API (via `boto3`) |
| `glm` | GLM 5 (Z.AI) | `zai.glm-5` | Bedrock Converse API (via `boto3`) |

Claude goes through the Anthropic-shaped Messages API (so it gets
Claude-specific params like `thinking`/`output_config`); Kimi and GLM aren't
Anthropic models, so they go through Bedrock's provider-agnostic
`bedrock-runtime.converse()` API instead — same credentials, different
request/response shape. See `providers.py` for the split.

Each is run under the same two conditions (English vs Yoruba system prompt),
on the same 10 Yoruba test items, so you get both an A/B result per model
*and* a cross-model comparison. **All 3 providers × 2 conditions (up to 6
API calls) fire concurrently per item**, via a thread pool — see
"Parallel execution" below.

## Design

- **Same 10 Yoruba-language user inputs**, spanning factual Q&A, advice,
  explanation, creative writing, opinion/reasoning, step-by-step instructions,
  summarization, casual conversation, and arithmetic reasoning (`config.py`).
- **Two conditions per provider**, identical model, identical user input,
  only the system prompt differs:
  - **Condition A** — system prompt written in **English**, instructing the
    model to respond entirely in Yoruba.
  - **Condition B** — the same instructions, written **natively in Yoruba**.
- Four items (`qa_factual_1`, `explain_1`, `summarize_1`, `reasoning_1`) carry
  a `reference` — a ground-truth answer or source text — used for faithfulness/
  groundedness scoring. Open-ended items (advice, opinion, creative, casual
  chat) have no reference by design.
- A provider with missing credentials is **skipped, not fatal** — run
  whichever subset you currently have keys for.

## What gets measured

1. **Code-switching rate** (automatic, heuristic) — fraction of tokens that
   match common English function words (the, and, however, therefore, ...).
2. **Orthographic fluency proxies** (automatic) — diacritic density and
   ASCII-only word ratio.
3. **Stray non-Latin characters** (automatic) — sanity check for script bugs.
4. **Faithfulness & groundedness** (automatic, **optional**, LLM-as-judge —
   `llm_judge.py`) — scores each response 1-5 on:
   - *faithfulness*: does it avoid inventing facts/figures/claims it can't
     support?
   - *groundedness*: for items with a `reference`, does the response rest on
     that material? For open-ended items, does it stay within what was asked?
   - plus a `hallucination_detected` boolean and a short justification.
   **The judge always runs on Claude via Bedrock**, independent of which
   provider generated the response being judged — so Kimi and GLM outputs
   aren't scored by themselves or by each other, keeping the evaluation
   consistent across all three systems. Costs one extra API call per (item,
   condition, provider) — enable with `--with-judge`.
5. **Fluency / naturalness / task accuracy / faithfulness / groundedness**
   (**manual, native-speaker rating** — this is the metric that actually
   matters). `analyze.py` exports a CSV with one row per (item, provider,
   condition) and empty columns for a native Yoruba speaker to score 1–5,
   alongside the LLM-judge's own scores as a reference point.

Automatic metrics (heuristic and LLM-judge alike) are cheap sanity checks,
not a substitute for human judgment. Treat the CSV rating step as the actual
finding — and note that code-switch rate / diacritic density aren't perfectly
comparable *across* models with different natural verbosity/style, so lean on
the human ratings especially for the cross-provider comparison.

## Setup

```bash
cd yoruba-prompt-experiment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit `.env` (already created, gitignored) and fill in your AWS credentials —
that's the only credential set needed for all three providers:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=        # only if using temporary/STS credentials
AWS_REGION=us-east-1

BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
JUDGE_MODEL_ID=us.anthropic.claude-opus-5

KIMI_MODEL=moonshotai.kimi-k2.5
GLM_MODEL=zai.glm-5
```

The IAM principal needs `bedrock:InvokeModel` permission, and you need model
access enabled for each of these three models in the Bedrock console for
your account/region — **third-party model access on Bedrock varies by
account**, so don't assume the defaults above are enabled for you. If
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` are left blank, it falls back to
boto3's default credential chain.

**Model ID gotchas, verified live against this setup — don't guess, check
your own account:**

```bash
# What models does this account/region actually have access to?
aws bedrock list-foundation-models --region us-east-1
# Filter for a provider, e.g.:
aws bedrock list-foundation-models --by-provider Anthropic --region us-east-1

# Current Claude models are often INFERENCE_PROFILE-only — the plain
# "anthropic.claude-haiku-4-5" ID 400s with "on-demand throughput isn't
# supported"; you need the region-prefixed inference profile ID instead:
aws bedrock list-inference-profiles --type-equals SYSTEM_DEFINED --region us-east-1
```

(boto3 equivalents: `client.list_foundation_models(byProvider="Anthropic")`
and `client.list_inference_profiles(typeEquals="SYSTEM_DEFINED")`.) Kimi and
GLM were `ON_DEMAND` on this account (no inference profile needed) — but
that's account-specific too, and their model catalog on Bedrock moves fast;
re-check if `KIMI_MODEL`/`GLM_MODEL` start 404ing.

## Running

```bash
# Full run: all 3 providers x 10 items x 2 conditions = 60 API calls
python experiment.py

# Just Claude
python experiment.py --providers claude_haiku

# Just Kimi and GLM
python experiment.py --providers kimi,glm

# Quick smoke test on the first 3 items, all providers
python experiment.py --limit 3

# Also run the faithfulness/groundedness LLM-judge (extra API calls; judge is always Claude/Bedrock)
python experiment.py --with-judge

# Cap concurrency (default 6 = 3 providers x 2 conditions in flight at once)
python experiment.py --max-workers 3
```

Any provider missing credentials in `.env` is skipped with a warning — the
run continues with whatever's ready. A per-item, per-provider API failure is
also caught and recorded (`stop_reason: "error"`) rather than crashing the
whole run, so one flaky provider doesn't lose the other two.

This writes `results/run_<timestamp>.json` with every raw response, stop
reason, token usage, computed metric, and (if `--with-judge`) judge scores.

### Parallel execution

For each test item, every (provider, condition) pair is dispatched
**concurrently** through a `ThreadPoolExecutor` — so Claude Haiku's English
call, Claude Haiku's Yoruba call, Kimi's English call, Kimi's Yoruba call,
GLM's English call, and GLM's Yoruba call (up to 6 calls) are all in flight
at once instead of run one after another. Each provider has its own cached
client instance (thread-safe, lazily built), so there's no shared connection
state across the parallel calls. When `--with-judge` is on, judge calls for
an item run the same way — concurrently, once that item's generations are
done. This cuts wall-clock time roughly proportional to the number of
providers/conditions instead of summing them. Tune concurrency with
`--max-workers` (default 6) if a provider's rate limit starts rejecting
bursts.

## Analyzing

```bash
python analyze.py                          # analyzes the most recent run
python analyze.py results/run_XXXXXXXX.json # or a specific run
```

No provider SDKs or credentials are needed to run this — it only reads the
saved JSON. It prints:

1. A per-provider **A vs B** table (English prompt vs Yoruba prompt)
2. If more than one provider ran, a **cross-provider comparison** table for
   condition A and another for condition B, so you can see e.g. "Kimi's
   code-switch rate under the Yoruba prompt vs GLM's vs Claude Haiku's" side
   by side.
3. A per-item breakdown line showing every provider's code-switch rate.

And writes two files:

- `results/summary_run_<ts>.json` — aggregate metrics, nested by provider
- `results/human_rating_run_<ts>.csv` — **hand this to a native Yoruba
  speaker.** One row per (item, provider, condition). Columns:
  `fluency_1to5`, `naturalness_1to5`, `task_accuracy_1to5`,
  `faithfulness_1to5`, `groundedness_1to5`, `code_switch_observed_yes_no`,
  `notes` (blank, for the rater), plus `llm_judge_*` columns for comparison.

### Recommended rating protocol

For a cleaner result, do a **blind rating pass**: shuffle the CSV rows and
hide the `provider`, `condition`, and `llm_judge_*` columns before handing it
to the rater, then re-join by `item_id` + response text afterward. This
avoids the rater's expectations (which model "should" do better, or what the
judge already said) biasing the score.

## Interpreting results

- **Within a provider**, if A ≈ B → that model's Yoruba output quality
  doesn't depend on system-prompt language. If B clearly beats A → that
  model benefits from native-language prompting.
- **Across providers**, this experiment can also surface whether the
  "prompt in the target language" effect is universal or model-specific —
  e.g. it might matter a lot for one model and not at all for another,
  which is itself an interesting finding for choosing a backend.
- Watch **refusals**, **errors**, **non-Latin stray chars**, and
  **hallucination count** — non-zero values flag a bug or model failure mode
  worth investigating before trusting the rest of the numbers for that
  provider/run.

## Files

| File | Purpose |
|---|---|
| `config.py` | The two system prompts + the 10 test items (+ references for groundable ones) |
| `metrics.py` | Heuristic automatic metrics (code-switch rate, diacritic density, etc.) |
| `providers.py` | Unified `generate()` across Claude Haiku, Kimi, and GLM — all via Bedrock |
| `llm_judge.py` | LLM-as-judge faithfulness/groundedness scoring (always via Claude/Bedrock) |
| `experiment.py` | Runs every ready provider through both conditions, saves raw results |
| `analyze.py` | Aggregates metrics, prints per-provider + cross-provider report, exports CSV |
| `.env` / `.env.example` | AWS credentials — the only credential set needed for all 3 providers |
| `results/` | Output directory (gitignored except `.gitkeep`) |
| `RESULTS.md` | Write-up of the actual experiment findings from a completed run |

## Extending

- Add more test items in `config.py` (`TEST_ITEMS`) to cover more task types
  or more groundable (reference-bearing) tasks.
- Add another Bedrock-hosted provider by adding a `Provider(...)` entry to
  `PROVIDERS` in `providers.py` with `kind="bedrock_converse"` — check
  `aws bedrock list-foundation-models` for what else your account has access
  to (DeepSeek, Qwen, Mistral, and others are commonly available). A model
  not on Bedrock at all would need a new `kind` and its own client/call
  function, following the `bedrock_claude` vs `bedrock_converse` split.
- Swap `BEDROCK_MODEL_ID` to compare Claude tiers too (e.g.
  `us.anthropic.claude-opus-5` vs `us.anthropic.claude-haiku-4-5-...`) — note
  current-tier models (Opus 5, Sonnet 5) and older-tier models (Haiku 4.5,
  Sonnet 4.5) take different request parameters; `providers.py` already
  handles this distinction automatically.
- Have a second, independent judge model score faithfulness/groundedness and
  check agreement with the first, before trusting the automated scores at
  larger scale.
