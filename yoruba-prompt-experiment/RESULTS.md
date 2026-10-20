# Results: Yoruba vs. English System Prompt Experiment

**Run:** `results/run_20260728_125345.json` · **Items:** 10 · **Judge:** Claude Opus 5 (`us.anthropic.claude-opus-5`, Bedrock) · **Date:** 2026-07-28

**Systems compared** (all three hosted directly on Amazon Bedrock, one AWS credential set):

| Provider key | System | Bedrock model ID |
|---|---|---|
| `claude_haiku` | Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `kimi` | Kimi K2.5 (Moonshot AI) | `moonshotai.kimi-k2.5` |
| `glm` | GLM 5 (Z.AI) | `zai.glm-5` |

## TL;DR

Across all three models, **the system-prompt language (English vs. Yoruba) made no consistent, reliable difference** in faithfulness, groundedness, or code-switching. Per-item scores swing in both directions depending on the model and the task — sometimes the Yoruba prompt clearly helps (GLM's summarization task: 1/1 → 4/4), sometimes it clearly hurts (Kimi's arithmetic task: 5/4 → 1/1) — and those swings roughly cancel out in the aggregate. The bigger, more consistent signal is **which model you pick**: Kimi was the strongest and most faithful performer on this set, GLM the weakest and the only one to leak actual Chinese/Japanese script into its Yoruba output, and Claude Haiku sat in between.

## Headline numbers (10 items each, English → Yoruba system prompt)

| Metric | Claude Haiku 4.5 | Kimi K2.5 | GLM 5 |
|---|---:|---:|---:|
| Mean code-switch rate | 0.0 → 0.0015 | 0.0017 → 0.0021 | 0.0047 → 0.0094 |
| Mean diacritic density /100 chars | 39.9 → 41.1 | 44.2 → 44.0 | 41.8 → 41.3 |
| Non-Latin stray characters (total) | 0 → 0 | 0 → 0 | **3 → 7** |
| Mean response length (words) | 154.9 → 192.4 | 187.5 → **244.7** | 292.7 → 259.6 |
| LLM-judge mean faithfulness (1–5) | 3.4 → 3.6 | **4.3** → 3.9 | 3.1 → 3.6 |
| LLM-judge mean groundedness (1–5) | 2.8 → 2.9 | **3.7** → 3.6 | 2.5 → 3.0 |
| Hallucinations flagged (of 10) | 3 → 2 | 3 → 3 | 4 → 4 |

**Kimi is the strongest model here on every judge metric, in both conditions.** GLM is the weakest on faithfulness/groundedness in condition A, though it closes most of that gap under the Yoruba prompt. None of the within-model A→B shifts are large enough, on n=10, to call a real effect rather than noise — see the per-item table below for why.

## Per-item faithfulness/groundedness (format: `A-faith/A-ground → B-faith/B-ground`)

| Item | Category | Claude Haiku | Kimi | GLM |
|---|---|---|---|---|
| qa_factual_1 | factual_qa | 4/4 → 4/4 | 4/4 → 4/4 | 4/4 → 4/4 |
| advice_1 | advice | 5/4 → 4/2 | 5/4 → 5/4 | 4/4 → 5/4 |
| explain_1 | explanation | 2/2 → 2/2 | 3/2 → 4/3 | 1/1 → 1/1 |
| story_1 | creative | 4/2 → 4/2 | 4/3 → 4/3 | 5/3 → 4/2 |
| opinion_1 | reasoning | 4/3 → 4/3 | 5/4 → 5/4 | 4/3 → 5/4 |
| instruction_1 | task_instruction | 4/4 → 4/3 | 4/3 → 3/3 | 3/2 → 3/2 |
| summarize_1 | summarization | 3/3 → 3/3 | 5/5 → 5/5 | **1/1 → 4/4** |
| conversation_1 | casual_chat | 2/2 → 4/4 | 4/4 → 4/5 | 4/2 → 5/4 |
| reasoning_1 | reasoning (arithmetic) | 2/1 → 3/3 | **5/4 → 1/1** | 1/1 → 2/2 |
| task_translation_1 | task_instruction | 4/3 → 4/3 | 4/4 → 4/4 | 4/4 → 3/3 |

The two biggest single swings in the whole dataset point in **opposite directions**: GLM's summarization jumps from broken (1/1, invented content about "government, prisoners, and year-end celebrations" that has nothing to do with the source) to solid (4/4) under the Yoruba prompt, while Kimi's arithmetic answer goes from perfect (5/4) to badly wrong (1/1, computed 450 km instead of 330) under the *same* switch. If prompt language systematically mattered, we'd expect swings to cluster in one direction — they don't.

## The arithmetic question: a clean cross-model stress test

All three models were asked the same question — a train at 60 km/h for 5.5 hours, correct answer 330 km — under both conditions. This is useful because there's exactly one right answer, so "faithful" isn't a judgment call:

| Model | Condition A (English prompt) | Condition B (Yoruba prompt) |
|---|---|---|
| Claude Haiku 4.5 | **Wrong** — computed 300 km (dropped the half-hour) | Correct number (330) but contradicts itself, states 300 km first |
| Kimi K2.5 | **Correct** — 60 × 5.5 = 330, clean steps | **Wrong** — 450 km, confused reasoning, invents ambiguity about "àbọ̀" |
| GLM 5 | **Wrong** — misread 5.5 as 1.5, got 90 km | Wrong first (390 km via bogus 180+120+30), self-corrects to 330 at the end |

Five of six attempts had *some* problem — only Kimi/A was clean. This isn't a language-of-prompt effect; it's plain arithmetic unreliability that shows up regardless of which language you're steering the model in, and it's worth knowing before trusting any of these models with numeric tasks in production, in any language.

## Model-specific failure modes (the real finding)

Averaging across A and B, each model has its own characteristic problem — more informative than the prompt-language question:

**Claude Haiku 4.5** — generally coherent Yoruba, occasional garbled/invented vocabulary (see the earlier Haiku-only run's `RESULTS.md` history for the "Malaridi" invented-ingredient example), no script leakage.

**Kimi K2.5** — strongest and most consistent on faithfulness/groundedness across both conditions; when it fails, it fails on numeric/reasoning tasks specifically (see the arithmetic swing above) rather than on fluency.

**GLM 5** — the only model that leaked real non-Yoruba script into its output, in *both* conditions:

- `qa_factual_1` (A): `不`, `被` (Chinese ideographs)
- `explain_1` (A): `爵` (Chinese ideograph)
- `instruction_1` (B): `撞`, `为`, `保`, `证` (Chinese) + `し`, `な`, `い` (Japanese Hiragana)

This is a materially different and more severe failure mode than code-switching into English — it's leaking a third, unrelated language family into Yoruba text, consistent with a Chinese-origin model's training distribution surfacing under pressure. Automatic diacritic-density and code-switch-rate metrics don't catch this at all; only the non-Latin-character scan does, and only because I built it to check for exactly this class of bug.

## What the automatic metrics can't tell you

- **Code-switch rate only catches English *function* words** (the, and, however...) — it will miss content-word/loanword switches (e.g. a recipe using "Onion" verbatim) and, as GLM shows, it says nothing about non-English/non-Yoruba script leakage. Use the non-Latin-character count for that, and even that only catches script-level intrusions, not semantic garbage written in Latin-looking Yoruba characters.
- **Response length varies a lot by model and condition** (Kimi nearly balloons from 187 to 245 words average under the Yoruba prompt) — this affects judge scores indirectly (more text = more surface area for both good grounding and hallucination) and isn't something the current design controls for.
- **These are all single-run scores.** The earlier Haiku-only run (10 items, same tasks) scored Haiku's `reasoning_1` as wrong (300 km) in *both* conditions; this run's Haiku got condition B right. Same model, same prompt, different generation — a reminder that n=1 per (model, condition, item) is noisy, and any of these numbers could look different on a re-run.

## What's still missing

- **Native-speaker rating.** `results/human_rating_run_20260728_125345.csv` has one row per (item, provider, condition) — 60 rows — with `fluency_1to5`, `naturalness_1to5`, `task_accuracy_1to5`, `faithfulness_1to5`, `groundedness_1to5` columns blank, waiting for a native Yoruba speaker. This is still the decisive step; everything above is LLM-judge and regex-heuristic only. Recommended: blind the `provider`/`condition`/`llm_judge_*` columns before handing it off (see `README.md` → "Recommended rating protocol").
- **n=10 items, 1 run.** Enough to see the shape of the result and catch real bugs (like GLM's script leakage), not enough to report a precise effect size or run significance tests on.
- **Judge is a single model (Opus 5), not cross-validated.** A second independent judge, or spot-checking a sample of judge verdicts against the human ratings once those come in, would tell you how much to trust the automated scores above.

## Infrastructure notes (what it took to get real numbers)

- **Kimi and GLM are hosted directly on Bedrock** (`moonshotai.kimi-k2.5`, `zai.glm-5`, both `ON_DEMAND`), verified live via `bedrock:ListFoundationModels` — no separate Moonshot/Zhipu API keys needed, just the same AWS credentials as Claude. They're called through Bedrock's provider-agnostic Converse API (`bedrock-runtime.converse()`) rather than the Anthropic-specific Messages API client, since they aren't Anthropic-shaped models. This is a meaningfully simpler setup than the original plan (three separate API keys/SDKs) and worth keeping if you extend this further — check `providers.py` before assuming a new model needs its own client type.
- **Claude models on this account are `INFERENCE_PROFILE`-only** — bare IDs like `anthropic.claude-haiku-4-5` 400 with "on-demand throughput isn't supported"; the region-prefixed `us.anthropic.claude-haiku-4-5-20251001-v1:0` works. Doesn't apply to Kimi/GLM, which are plain `ON_DEMAND`.
- **The judge's `output_config.format` (structured JSON output) 400s on this Bedrock account/SDK combo** despite being documented as supported — worked around by prompting for plain JSON and parsing defensively (`llm_judge.py`).
- **Concurrent judge calls against the single Opus 5 endpoint hit `503`s** under 6-way concurrency with synchronized retries — fixed by capping judge concurrency at 2 and adding jitter to retry backoff so parallel retries desynchronize (`experiment.py`).
- **The code-switch-rate heuristic had a false-positive bug**: single-character tokens ("a", "i") are common standalone Yoruba words, not just English articles. Fixed in `metrics.py` — this is why the code-switch numbers above are near-zero rather than the ~0.01–0.04 an earlier, buggier pass reported.
- This run (60 generations + 60 judge calls, full concurrency) completed in 490s with zero errors — a large improvement over the first Haiku-only attempt, which lost 15/20 calls to connection drops before retry logic was added.

## Bottom line

**On the original question — does system-prompt language matter for Yoruba output — the answer across three different model families is still no, not detectably**, with the same caveat as before: n=10, one run, no human rating yet. What *did* come out of running all three side by side is more useful for actually building something: pick Kimi over GLM if faithfulness/groundedness on Yoruba tasks matters, budget for arithmetic/numeric verification regardless of model, and if you use GLM specifically, add a script-leakage check (non-Yoruba, non-Latin characters) to your output pipeline — it's a real, repeatable failure mode on this evidence, not a one-off.
