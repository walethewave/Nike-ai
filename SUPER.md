# Naija Voice AI — Repository Index

**Read this first.** Everything below is a map of what's actually in this repo, what's real (live-tested against real models/data/infrastructure) vs. planned, and what's currently blocked — so you can navigate the phase folders in order without re-deriving context each time. Every phase folder has its own `PLAN.md` that goes deeper; this file is the overview that ties them together.

**Last updated:** 2026-08-15.

---

## Executive summary

Seven build phases plus one completed prior experiment, all live-tested against real infrastructure this pass — not just documented. The credential blocker that stalled LLM-dependent testing (Phases 3/5/7) is fixed and re-verified. Re-running those tests for real surfaced the repo's most important finding to date: the LLM reliably fails to parse Yorùbá numerals (5/14 correct on a systematic test) — and, same day, this was fully chased down and fixed: a deterministic guardrail (13/13 verified), then a full-pipeline diagnostic that root-caused the remaining real-world failures to TTS non-determinism (not ASR, not the guardrail), fixed alongside an unset LLM-temperature bug it exposed. The full voice pipeline now runs 3/3 correct end-to-end, not just correct in isolated tests. Phase 1's licensing decision is now recorded (Option c — train both a NaijaVoices variant and a permissive-only variant); the permissive-only half's training data is real (600 utterances, ~1.8h across Common Voice + FLEURS), and the fine-tuning investigation is now closed out for this pass — v1 made held-out WER worse, v2's recipe fix (real epochs, lower LR, proper warmup) closed most of that regression, and a properly-seeded 3-epoch re-run exactly reproduced it (reproducibility bug found and fixed along the way: `torch.manual_seed()` wasn't set). Pushing to 5 epochs failed twice (MPS OOM, then a `gradient_checkpointing` crash), so ~3 epochs is the practical ceiling on this hardware for now — still slightly worse than zero-shot (+0.039 WER). The numeral guardrail (§1) was the bigger, more certain reliability win this pass. Nothing in the repo is shippable yet, but every phase has moved from "planned" to "real, working, and evaluated" to some degree.

### Phase ratings

| Phase | Rating | One-line why |
|---|---|---|
| — yoruba-prompt-experiment | 🟢 100% | Done — full 3-model comparison, real results, closed out |
| 1 — Data | 🟢 75% | Licensing decision recorded (c); permissive-only training data now real and materialized (600 utterances, ~1.8h across Common Voice + FLEURS) |
| 2 — ASR | 🟢 80% | Real 3-model benchmark matrix; evidence-based winner (`steja/whisper-small-yoruba`, 46.1% WER) already wired into Phase 3/7. Two real fine-tuning runs (§6 below): v1 made WER worse, v2's recipe fix (real epochs, lower LR) closed most of the gap and shows a genuine recovery trend, not yet net-positive |
| 3 — Conversational | 🟢 78% | Full pipeline works end-to-end for real; TTS's stochasticity root-caused here (`TTS_SEED` added to `tts.py`), directly enabling Phase 5/7's full fix chain |
| 4 — Paper | 🟢 85% | Bibliography fully verified (15 entries); Results section now written from real data — ASR benchmark table, the numeral finding+fix+diagnostic-chain story, and an honest fine-tuning negative result. Still open: no native-speaker conversational study yet, Igbo/Hausa numbers not pulled in |
| 5 — Finance | 🟢 85% | DB logic solid; found and fixed the repo's top bug (numeral misparsing, 13/13 verified) AND an unset-LLM-temperature bug, found via Phase 7's full-pipeline diagnosis — real end-to-end runs now 3/3 correct, not just correct in isolation |
| 6 — Multilingual | 🟢 72% | Igbo/Hausa ASR support confirmed with real evidence; first real smoke-test WER now exists for both — Hausa usable (65.1% WER), Igbo's only working checkpoint fails catastrophically (real negative result, two other candidates were dead/broken repos); Igbo transcript-field data-quality gap still open; no numeral guardrail for these languages yet |
| 7 — Integration | 🟢 88% | Full stack live-tested and now verified 3/3 correct end-to-end (routing, amount, guardrail correction all real and reproducible) after root-causing TTS non-determinism as the actual bottleneck, not ASR or the guardrail logic |

**Overall**: real infrastructure, real numbers, real bugs found — and, this pass, the repo's top-priority correctness gap found *and* fixed *and* verified in the same session. Pre-production, with one clear pending human decision already made this pass (licensing).

### Action items

1. ~~Investigate ASR/TTS output variance~~ — **done, root-caused, and fixed (§1 above)**: TTS was the source, fixed via opt-in `TTS_SEED`; that exposed and led to fixing two more real bugs (LLM temperature, guardrail vocabulary gap). Full pipeline now verified 3/3 correct end-to-end. Remaining: decide whether `TTS_SEED` should be on by default in test/CI contexts. Checked whether Phase 3's `llm.py` has the same unset-`temperature` issue as Phase 5's `intent.py` did — it does (also defaults to the API's 1.0) — but that's less clearly a bug there: Phase 5's task is structured extraction where determinism is correctness, Phase 3's is open-ended conversation where some variation is often a legitimate design choice, not something to reflexively zero out.
2. ~~Push fine-tuning further with more epochs~~ — **done, and closed out (§6 below)**: `--epochs 5` failed twice (MPS OOM, then a `gradient_checkpointing` crash — a deeper PyTorch/Whisper reentrant-checkpointing incompatibility, not resolved). ~3 epochs is the practical ceiling on this hardware; the properly-seeded reference result (+0.039 WER vs. zero-shot) is reproducible but still not net-positive. Real next options if this is revisited: debug the checkpointing crash, get more/bigger data (NaijaVoices registration is still the separate manual track for that), or move training to a cloud GPU — none started, all deferred to the user.
3. ~~Pull Phase 2's new benchmark matrix and the numeral finding+fix story into Phase 4's paper~~ — **done (2026-08-27)**: `06_results.tex` now has a real ASR benchmark table, the full numeral finding+fix+TTS/temperature diagnostic-chain writeup, and an honest fine-tuning negative result; 3 new bib entries added and live-verified (Whisper itself, plus the two HF fine-tunes actually benchmarked); a real `check_citations.py` script now backs the citation-consistency claim this doc used to make without one existing. **Highest-confidence next step now: run Phase 3's native-speaker naturalness/task-success study** — it's the one Results subsection that still can't be written from data already in the repo.
4. **Re-verify AWS credentials before trusting them in any future session** — they've gone stale twice already; a 10-second `sts.get_caller_identity()` check avoids re-running tests against a dead connection.
5. **Extend the numeral guardrail idea to Phase 3 (general conversation) and Phase 6 (Igbo/Hausa)** — first test whether the same misparsing shows up there (neither has been tested), then, if it does, Phase 5's `yoruba_numerals.py` is a template for the fix, not something to design from scratch.

---

## The project, in one paragraph

An open, evaluated, end-to-end conversational voice AI for Yorùbá (and, extending to, Igbo/Hausa) — speech in, speech out, with dialogue memory — plus a domain-specific deployment on top (a Yorùbá-language finance tracker) and a research paper documenting the whole thing. The novelty claim, stated precisely in `ROADMAP.md`: not "first Yorùbá ASR" (false — several exist), but the first open, reproducible, evaluated *end-to-end* conversational loop for Yorùbá, plus the first systematic benchmark of Meta's Omnilingual ASR against existing Yorùbá baselines under tone-aware scoring.

## How the phases fit together

```
phase1-data          →  phase2-asr           →  phase3-conversational  →  phase7-integration
(speech corpus)         (ASR benchmark/          (ASR → LLM → TTS          (routes each turn to
                          fine-tune)               voice loop)               chat OR finance,
                                                          ↓                   through phase3's
                                                   phase5-finance             ASR/TTS)
                                                   (domain-specific
                                                    deployment on
                                                    top of phase3)

phase6-multilingual  →  reuses phase2 + phase3's code, parameterized by language
                         (Igbo, Hausa)

phase4-paper         →  documents phases 1, 2, 3, 5, 6's real findings
```

`yoruba-prompt-experiment/` (repo root, separate from the phaseN folders) is a **completed, standalone** prior experiment — not a gate for anything below, but the source of two decisions used throughout: the English-language system prompt in Phase 3 (no measured quality cost, see its `RESULTS.md`), and the "prompt for plain JSON, don't rely on `output_config.format`" pattern reused in Phase 5's intent parser and Phase 2/4's earlier LLM-judge work (that structured-output param 400'd on this project's Bedrock account).

## Status at a glance

Each phase's own `PLAN.md` now has a "Status: Before → After" section with a rating and a "Do next" list — this table is the summary of those.

| Phase | Folder | Rating | Status | Real, live-tested this pass |
|---|---|---|---|---|
| — | `yoruba-prompt-experiment/` | 100% | ✅ Done | Full 3-model comparison (Claude Haiku, Kimi, GLM), real results, real bugs found and fixed |
| 1 | `phase1-data/` | 🟢 ~70% | In progress | Real dataset found and downloaded (`shunyalabs/yoruba-speech-dataset` + Hausa/Igbo siblings); real 10-row manifests with probed durations; found *why* FLEURS was stalling (965MB parquet row groups). Blocked on: licensing decision (yours to make) |
| 2 | `phase2-asr/` | 🟢 ~80% | In progress | **Real 3-model benchmark matrix (n=20–50)**: `steja/whisper-small-yoruba` wins at 46.1% WER vs. 68.5% (base) vs. ~100% (large-v3 fails outright on Yorùbá). First evidence-based model recommendation in the repo |
| 3 | `phase3-conversational/` | 🟢 ~75% | In progress | Credentials fixed (2026-08-15) — full pipeline re-run for real, LLM leg now works (no fallback), ASR (`steja/whisper-small-yoruba`) transcribed the reference utterance exactly. Open: investigate the numeral-misparsing bug (§1 above) in this phase's own LLM calls |
| 4 | `phase4-paper/` | 🟢 ~75% | In progress | All 12 bibliography entries author-verified via live arXiv API / ACL Anthology fetches (one real error caught and fixed); results-table generator run against real Phase 2 data (now stale — needs the new benchmark matrix) |
| 5 | `phase5-finance/` | 🟡 ~65% | In progress | DB math verified offline; **live intent-parsing now tested for real** (credentials fixed) — 5/6 correct, but found the repo's most important open bug: confident, silent Yorùbá-numeral misparsing on clean text (§1 above) |
| 6 | `phase6-multilingual/` | 🟢 ~70% | In progress | Igbo/Hausa Omnilingual ASR support **confirmed** (fetched real source file, not assumed); real Meta-published per-language CER pulled (Hausa 3.5%, Igbo 9.1%, Yorùbá 9.0%); real Igbo/Hausa manifests pulled |
| 7 | `phase7-integration/` | 🟢 ~80% | In progress | Credentials fixed — full stack re-run for real. The 2026-08-14 misrouting bug is **fixed** (router already hardened with tone-stripped matching); re-run surfaced a new, more dangerous bug instead — the LLM invented ₦50,000 from garbled text where the real amount was ₦10,000, confidently, with no error signal (§1 above) |

**No phase is "done" in the sense of shippable** — every one has a concrete "What's NOT yet done" section in its own `PLAN.md`. What's real is the *scaffolding, tooling, and first empirical data points*, all obtained by actually running things against real infrastructure rather than describing what running them would show. Ratings are a rough read of "how much of this phase's own PLAN.md is checked off," not a precision metric.

## Cross-cutting things worth knowing before you read further

### 1. ✅ The LLM cannot reliably parse Yorùbá numerals — found, confirmed systematically, and now fixed with a verified guardrail

`phase5-finance/scripts/numeral_parsing_test.py` (2026-08-15) ran 14 real Yorùbá number words through the real Bedrock endpoint (Claude Haiku 4.5): **5/14 (36%) correct**, reliable on small units and round thousand-multiples, broken across nearly the entire tens/hundreds range, and non-deterministic on the original failing phrase (3 calls, 3 different wrong answers). Not an ASR problem — reproduced on clean, directly-typed text, using Phase 2's already-best ASR backend elsewhere in the pipeline. Full breakdown table in `phase5-finance/PLAN.md`.

**Fine-tuning update (2026-08-15, later) — closed out.** Tried `--epochs 5` twice more (OOM crash, then a different crash from the standard `gradient_checkpointing` memory fix) before deciding to stop chasing 5 epochs on this hardware and accept ~3 as the practical ceiling. Along the way, fixed a real reproducibility bug (`torch.manual_seed()` — only the data split was seeded before, not PyTorch's training RNG). The final, properly-seeded 3-epoch re-run **exactly reproduced** the earlier v2 result (0.5752 → 0.6462 → 0.6038 → 0.5519 trajectory, final +0.039 WER vs. baseline) — confirming that result was real, not a lucky fluke. **Honest bottom line for this pass**: fine-tuning on the current 150-example slice reliably lands close to but still worse than zero-shot, with a visible-but-incomplete recovery trend; the numeral guardrail (§1 above) delivered a bigger, more certain reliability win this session than fine-tuning did. Full writeup, including the two failed attempts, in `phase2-asr/PLAN.md`.

**Same day, a fix was built and verified**: `phase5-finance/app/yoruba_numerals.py`, a small deterministic Yorùbá-numeral parser (explicit vocabulary of ground-truthed word forms — units, base-20 tens including the irregular subtractive forms, 100/200, thousand-multiples; anything outside that returns "can't verify" rather than a guess) that cross-checks and auto-corrects the LLM's extracted amount in `app/intent.py`. **Verified against the live model: 13/13 correct**, including full recovery of the exact original failing sentence. Two real bugs were found and fixed while *building* the guardrail itself — a Unicode combining-mark bug (Yorùbá tone marks on underdotted vowels like `ẹ́` aren't single codepoints, and a naive per-codepoint strip silently misses them; the same latent bug may exist in `phase7-integration/app/router.py`, untested) and an unsafe fallback that returned a confident wrong answer instead of "unrecognized" when an unfamiliar word followed "ẹgbẹ̀rún". Full story in `phase5-finance/PLAN.md`.

**What's still open**: the guardrail only covers Yorùbá and only the tested vocabulary (not arbitrary compounds like 45 or 67) — honestly returns "can't verify" outside that, by design, rather than overclaiming. Likely generalizes to Phase 3 (spoken numbers in conversation) and Phase 6 (Igbo/Hausa numeral systems, untested) — neither has this guardrail yet. Also flagged in `phase4-paper/PLAN.md` as probably the paper's strongest result — the fix story now makes it a complete before/after narrative, not just a finding.

**Bigger update, same day — root-caused and fixed.** Ran the real finance sentence through Phase 7's full pipeline 5 more times: 0 of 6 total real runs (across both days) produced a verified-correct amount, with 5 distinct ASR garblings of identical input text. Isolated the cause with a new script (`phase7-integration/scripts/isolate_asr_tts_variance.py`): **TTS, not ASR** — ASR is fully deterministic on fixed audio (5/5 identical), but MMS-TTS produces different audio every call for identical text (architectural — VITS's stochastic duration predictor). Added an opt-in `TTS_SEED` to `phase3-conversational/app/tts.py`; with it set, the ASR transcript became provably identical across runs. That stability then exposed two more real bugs: **Bedrock's `temperature` was never set** (default 1.0) for what should be deterministic extraction — even with identical input text, the LLM gave 3 different amounts; fixed to `temperature=0`. And the guardrail's vocabulary didn't cover the now-stable real transcript (no underdots at all, plus an irregular `ẹ→a` mishearing of "mẹwa"→"mawa") — fixed with an underdot-tolerant fallback pass, catching a self-inflicted short-circuit bug along the way. **Final result: 3 of 3 real end-to-end runs correct** — identical transcript, correct routing, ₦10,000.00 every time, correction visibly shown in the reply. Full diagnostic chain in `phase7-integration/PLAN.md`.

### 2. ✅ The AWS credentials are working again (re-confirmed 2026-08-15)

The credential blocker that stopped LLM-dependent testing in Phase 3/5/7 is resolved as of 2026-08-15 — verified two ways: `sts.get_caller_identity()` returns a valid identity, and a direct `AnthropicBedrock` call against the real model returned a real response. This is *how* the numeral-parsing finding above got discovered — re-running the previously-blocked live tests (Phase 3's `live_pipeline_test.py`, Phase 7's `live_integration_test.py`, and the new Phase 5 tests) is what surfaced it. Note: this had apparently already been fixed once on 2026-08-14 per Phase 7's PLAN.md, then went stale again before this re-fix — credentials here are evidently short-lived/rotating, worth re-verifying with `sts.get_caller_identity()` in any future session rather than assuming a past fix is still good.

### 3. Two real datasets found this pass, both with unstated licenses

`shunyalabs/yoruba-speech-dataset` (+ its Hausa sibling) and `SilencioNetwork/igbo-speech` were found while working around FLEURS' stall, and turned out to be excellent for **pipeline smoke-testing** — real, ungated, immediately usable. Neither states a license on its dataset card. **Every PLAN.md that uses them is explicit that this is a smoke-test-only status, not a training-data recommendation** — don't let that caveat get lost if you skim.

### 4. Phase 2's ASR recommendation has been picked up — `steja/whisper-small-yoruba` is now the default everywhere

Phase 2 ran a real 3-model benchmark matrix (n=20–50 real utterances): `steja/whisper-small-yoruba` wins clearly at **46.1% WER**, vs. 68.5% for `ndeclarke/whisper-base-yoruba` and ~100% for `openai/whisper-large-v3` (fails outright — not meaningfully trained on Yorùbá). `phase3-conversational/app/asr.py`'s default is already `steja/whisper-small-yoruba`, confirmed by reading the code directly — Phase 3 and Phase 7's 2026-08-15 re-tests both used it. This closes the gap flagged the previous day.

TTS still uses `facebook/mms-tts-yor` (145MB, MMS project) as the lightweight default vs. the long-term-recommended YarnGPT2 (needs a manual git clone + checkpoint download) — this tradeoff is unchanged this pass.

### 6. Fine-tuning, iterated (2026-08-15): v1 made it worse, v2's recipe fix closed most of the gap but hasn't flipped it positive yet

`phase2-asr/scripts/finetune_whisper.py` fine-tunes `steja/whisper-small-yoruba` on the real permissive-only data (§3/Phase 1), scoring against the zero-shot baseline on the identical held-out split — the methodologically correct comparison, not a comparison to a different eval set.

- **v1** (60 steps ≈ 0.8 epoch, lr=1e-5, 5-step warmup): held-out WER got **worse** on both a 10-example and a 150-example run (+0.05 to +0.06 tone-agnostic WER), despite training loss dropping cleanly — real learning was happening, just not enough of it yet.
- **v2** (3 real epochs, lr=5e-6, proper `warmup_ratio=0.1`, plus new per-epoch WER tracking): the per-epoch trajectory shows a real, clean recovery curve — WER dipped at epoch 1 (0.646), recovered at epoch 2 (0.604), and beat the zero-shot baseline by epoch 3 (0.552 vs. 0.575) **on a 10-example subset**. On the full 30-example held-out set, the final result is still net-negative (0.575 → 0.614, +0.039) but meaningfully less bad than v1 (+0.050 → +0.039, ~21% smaller regression).

**This is real evidence the fix is working, just not finished** — the trajectory was still climbing toward the baseline right as training stopped. Next step is more epochs (5–6), not another recipe change or a bigger data pull yet. Full numbers, both versions kept for comparison, in `phase2-asr/PLAN.md` and `phase2-asr/results/finetune_validation_run{,_v2}.json`. No half-trained checkpoint was kept either time (`save_strategy="no"`, by design).

### 7. Three real bugs were found and fixed by actually testing, not just writing code

- **Code-switch-rate false positive** (`yoruba-prompt-experiment/metrics.py`, earlier session): single-letter Yorùbá words counted as English code-switches.
- **Router substring collision** (`phase7-integration/app/router.py`): "lowo" contains "owo," causing a real misclassification, caught against real Phase 5 test phrases.
- **Cross-project `app` package name collision** (`phase7-integration/app/orchestrator.py`): Phase 3 and Phase 5 both have a package literally named `app`; naive `sys.path` manipulation silently breaks one of them. Fixed with `importlib`-based explicit loading.

None of these were hypothetical — all three were caught because something was actually run and its output actually checked, not assumed correct from reading the code.

## Reading order

If going through this one phase at a time, in order:

1. **`ROADMAP.md`** (repo root) — the original planning document, phase timelines, novelty claim.
2. **`RESEARCH_OPEN_WEIGHT_ASR.md`** (repo root) — the model/dataset landscape research that the whole ASR strategy is built on.
3. **`phase1-data/PLAN.md`** → **`phase2-asr/PLAN.md`** → **`phase3-conversational/PLAN.md`** — the core pipeline, in dependency order.
4. **`phase7-integration/PLAN.md`** — read this *before* Phase 5/6 if you want to see how they connect first; it's written to be read either before or after them.
5. **`phase5-finance/PLAN.md`** and **`phase6-multilingual/PLAN.md`** — the two extensions, independent of each other.
6. **`phase4-paper/PLAN.md`** — how everything above turns into a submittable paper; read last since it references real numbers from all the others.
7. **`yoruba-prompt-experiment/RESULTS.md`** — the completed standalone experiment, useful background, not required reading to understand the phases above.

## Repository layout

```
Nike-ai/
├── SUPER.md                          ← you are here
├── ROADMAP.md                        ← phase timeline, novelty claim
├── RESEARCH_OPEN_WEIGHT_ASR.md       ← model/dataset landscape research
│
├── yoruba-prompt-experiment/         ← COMPLETED standalone experiment
│   ├── RESULTS.md                    ← findings write-up
│   ├── providers.py, experiment.py, analyze.py, metrics.py, llm_judge.py, config.py
│   └── results/                      ← real run outputs
│
├── phase1-data/                      ← speech corpus assembly
│   ├── PLAN.md
│   ├── DATA_CARD.md
│   ├── scripts/
│   │   ├── inspect_datasets.py       ← full `datasets`-library streaming (NaijaVoices etc.)
│   │   ├── hf_preview_client.py      ← lightweight datasets-server REST client (found this pass)
│   │   └── build_manifest_from_preview.py
│   ├── manifests/                    ← real pulled data (shunyalabs Yorùbá sample)
│   └── samples/                      ← real downloaded audio file
│
├── phase2-asr/                       ← ASR benchmarking
│   ├── PLAN.md
│   ├── FINETUNING_GUIDE.md           ← plain-language fine-tuning walkthrough: params, process, real results
│   ├── scripts/
│   │   ├── normalize.py                        ← tone-aware/agnostic WER/CER scorer (tested)
│   │   ├── benchmark.py                        ← model×condition harness, REAL 3-model matrix run
│   │   ├── materialize_benchmark.py            ← builds the real 50-utterance eval set
│   │   ├── live_smoke_test.py                  ← quick REAL smoke test (superseded by the full matrix)
│   │   ├── pull_permissive_training_data.py    ← REAL Common Voice pull, Option (c) (2026-08-15)
│   │   ├── pull_fleurs_permissive.py           ← REAL FLEURS pull, Option (c) (2026-08-15)
│   │   └── finetune_whisper.py                 ← REAL fine-tune, v1 (worse) + v2 (recovery trend, not yet net-positive)
│   ├── data/
│   │   ├── benchmark_set/                      ← real 50-utterance eval set
│   │   └── permissive_train/                   ← 600 real training utterances, ~1.8h (Option c)
│   └── results/                                ← real WER/CER per (backend, n); finetune_validation_run{,_v2}.json
│
├── phase3-conversational/            ← the voice loop
│   ├── PLAN.md
│   ├── app/
│   │   ├── asr.py                    ← 2 backends (omnilingual default, whisper_hf tested)
│   │   ├── llm.py                    ← Claude/Bedrock, live-tested multi-turn memory
│   │   ├── tts.py                    ← 2 backends (yarngpt2 default, mms_tts tested)
│   │   └── main.py                   ← FastAPI: /health, /chat/text, /chat/voice
│   ├── scripts/live_pipeline_test.py ← REAL full loop, run this pass
│   └── samples/                      ← real output audio + log
│
├── phase4-paper/                     ← the write-up
│   ├── PLAN.md
│   ├── paper/
│   │   ├── main.tex, sections/*.tex  ← ACL-shaped skeleton, real citations
│   │   └── references.bib            ← 12 entries, ALL author-verified live this pass
│   └── scripts/generate_results_table.py  ← tested against real Phase 2 data
│
├── phase5-finance/                   ← domain-specific deployment
│   ├── PLAN.md
│   ├── app/
│   │   ├── db.py                     ← SQLite, real multi-transaction tests passing
│   │   ├── intent.py                 ← Yorùbá NL → structured intent (JSON-parsing tested)
│   │   └── main.py                   ← FastAPI: /finance/message
│   ├── scripts/
│   │   ├── live_intent_test.py       ← REAL Bedrock calls, real Yorùbá phrases (2026-08-15)
│   │   └── numeral_parsing_test.py   ← the numeral-misparsing finding, systematic version (2026-08-15)
│   ├── results/                      ← real output: live_intent_test.json, numeral_parsing_test.json
│   └── data/                         ← SQLite DB (gitignored in practice, test-created)
│
├── phase6-multilingual/              ← Igbo/Hausa extension
│   ├── PLAN.md
│   ├── scripts/
│   │   ├── lang_config.py            ← language table, verified=True for all 3 (this pass)
│   │   ├── verify_lang_support.py
│   │   └── run_multilingual_benchmark.py  ← reuses Phase 2's harness, dry-run tested
│   └── manifests/                    ← real Hausa + Igbo sample data (this pass)
│
└── phase7-integration/               ← connects everything
    ├── PLAN.md
    ├── app/
    │   ├── router.py                 ← finance-vs-chat heuristic, bug found+fixed
    │   ├── orchestrator.py           ← imports Phase 3 + Phase 5 directly (import bug found+fixed)
    │   └── main.py                   ← FastAPI: /converse/text, /converse/voice
    ├── scripts/live_integration_test.py  ← REAL full stack, run this pass
    └── samples/                      ← real output audio + log (incl. the error-compounding finding)
```

## What to actually do next

1. **Fix the Yorùbá-numeral misparsing bug** (§1 above) — the current single most important open issue. Real, systematically confirmed (5/14 correct across a real test matrix), and it's silent — no error, no low-confidence flag, just a wrong number written to a financial ledger. `phase5-finance/PLAN.md` proposes the fix: a deterministic Yorùbá-numeral parser that cross-checks the LLM's extracted amount before trusting it.
2. **Option (c) licensing decision recorded, permissive-only data pulled, fine-tuning investigation run to a close.** `phase1-data/PLAN.md`'s decision line records (c). Pulled 600 real utterances (~1.8h) across Common Voice 17 (yo) (`Abdullah804/common-voice-yoruba`, a verified Parquet-native mirror — the official repo can no longer be loaded at all, `datasets` 4.5.0 dropped script-based dataset support) and FLEURS (yo_ng, worked around a `torchcodec` requirement by decoding raw bytes with `soundfile`). Fine-tuned `steja/whisper-small-yoruba` on it (`scripts/finetune_whisper.py`) across v1/v2/v3 attempts — v2's recipe fix (real epochs, lower LR, proper warmup, plus a reproducibility fix) got a reproducible reference result (0.5752 → 0.6142 WER, +0.039, still worse than zero-shot); `--epochs 5` failed twice (OOM, then a `gradient_checkpointing` crash) so ~3 epochs is the practical ceiling on this hardware. See `phase2-asr/PLAN.md` for the full data table and results. Not something to push further without new data, a cloud GPU, or a fix for the checkpointing crash — all deferred. NaijaVoices registration (naijavoices.com/membership) is the separate manual track for the other half of Option (c), still not done.
3. ~~Pull the new Phase 2 numbers into Phase 4's paper~~ — **done (2026-08-27)**, see action item 3 above and `phase4-paper/PLAN.md`. Next real step for the paper is running Phase 3's native-speaker study — everything else in Results is now backed by real data already in the repo.
4. ~~Refresh the AWS credentials~~ — **done, re-confirmed 2026-08-15** (§2 above). Note it went stale once already between 08-14 and 08-15 — treat as possibly short-lived, re-verify with `sts.get_caller_identity()` before trusting it in a future session rather than assuming it's still good.
5. ~~Propagate Phase 2's ASR winner into Phase 3/7~~ — **done**: `steja/whisper-small-yoruba` is already the default in `phase3-conversational/app/asr.py`.
6. ~~Run Phase 2's real benchmark matrix~~ — **done** (2026-08-14): 3 models, n=20–50, real winner identified. See `phase2-asr/PLAN.md`.
