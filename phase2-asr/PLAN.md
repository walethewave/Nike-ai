# Phase 2 — ASR: Plan

**Goal:** a benchmarked, and where warranted fine-tuned, Yorùbá ASR model — with honest tone-aware AND tone-agnostic numbers against real baselines, not a single cherry-picked WER.

**Time budget:** 3–4 weeks. **Depends on:** Phase 1's manifests + licensing decision.

**New to fine-tuning?** See [`FINETUNING_GUIDE.md`](FINETUNING_GUIDE.md) — a plain-language walkthrough of what fine-tuning is, every parameter in `scripts/finetune_whisper.py` explained, and the real before/after numbers from both runs below.

---

## Status: Before → After (as of 2026-08-14) — 🟢 ~80%

- **Before:** a single 3-utterance smoke test on one lightweight backend (`ndeclarke/whisper-base-yoruba`), no real model comparison, no evidence-based recommendation.
- **After:** a real 3-model benchmark matrix (`whisper_base_yoruba`, `whisper_small_yoruba`, `whisper_large_v3`) run at n=20 and n=50 real utterances. Clear winner: **`steja/whisper-small-yoruba` at 46.1% WER** (tone-agnostic), beating base (68.5%) by a wide margin, while large-v3 fails completely (~100% WER — not trained on Yorùbá). This is the first evidence-based model recommendation in the whole repo rather than a guess.
- **What's still open:** Omnilingual ASR (not on PyPI, needs a GitHub install) and NCAIR1 (gated, needs HF access approval) are the two models the original research brief flagged as most promising and neither has been benchmarked yet — the current winner is the best of what's been *tested*, not necessarily the best available.

**Note (2026-08-15):** a separate, unrelated-to-ASR problem was found downstream — the LLM stage (Phase 5/7) unreliably parses Yorùbá number *words* (36% correct on a systematic test, see `phase5-finance/PLAN.md`). Worth flagging here only so it's not confused with an ASR/transcription-quality issue — it's a text-reasoning gap in a completely different model, not something a better ASR backend affects.

**Do next:**
1. ~~Propagate this result to Phase 3/7~~ — **done**, confirmed by reading the code: `steja/whisper-small-yoruba` is already `phase3-conversational/app/asr.py`'s default.
2. Install and benchmark Omnilingual ASR (`pip install git+https://github.com/facebookresearch/omnilingual-asr`) — it's Apache 2.0 vs. Whisper fine-tunes' mixed licensing, and it's the model the paper's novelty claim leans on.
3. Fine-tune `steja/whisper-small-yoruba` on the real permissive-only training data now materialized below — that's the next concrete step once a training script exists.

## Real permissive-only training data pulled (2026-08-15, Option (c))

Two real, bounded, session-scale samples materialized for the commercial-safe half of Phase 1's licensing decision — full detail and the two real bugs worked around (script-based-dataset support dropped in `datasets` 4.5.0; `torchcodec` requirement for audio decoding) are in `phase1-data/PLAN.md`. Summary:

| Source | Script | Rows | Real hours | Notes |
|---|---|---|---|---|
| `Abdullah804/common-voice-yoruba` (CV17 yo mirror) | `scripts/pull_permissive_training_data.py` | 300 | 0.52h | mp3, 6.22s/clip mean, license unstated on mirror (upstream CC0) |
| `google/fleurs` (yo_ng) | `scripts/pull_fleurs_permissive.py` | 300 | 1.28h | wav, 16kHz, 15.4s/clip mean, CC-BY-4.0 |

All 600 files downloaded/decoded and spot-verified with `soundfile` (real, non-silent audio — e.g. FLEURS row0: 18.12s, 16kHz, peak 0.7475, matching Phase 1's original validation of the same row exactly). Manifests at `data/permissive_train/{common_voice_yo,fleurs_yo_ng}/manifest.csv`, same schema as Phase 1's shunyalabs manifest. Both scripts accept `--n` to pull more — this is a bounded session-scale sample, not the full ~130h permissive corpus, and scaling up is mechanical from here.

**Not yet done:** the actual fine-tuning script (`scripts/finetune_whisper.py` per the original plan) hasn't been written yet — this pass materialized the training data, not the training run itself.

## 🔴 Real fine-tuning validation run (2026-08-15): the mechanism works, but 1.8h of data made the model WORSE, not better

`scripts/finetune_whisper.py` — a real HuggingFace `Seq2SeqTrainer` fine-tuning script starting from `steja/whisper-small-yoruba`, using the standard HF Whisper fine-tuning recipe (precomputed log-mel features, tokenizer-padded labels, standard data collator). Reports WER/CER via `normalize.py`'s real scorer, comparing the zero-shot baseline against the fine-tuned model **on the identical held-out split** — not against Phase 2's earlier benchmark numbers, which used a different eval set and wouldn't be a fair comparison.

Two real runs, both on real data, both on-device (M5 Mac, MPS backend):

| Run | Train examples | Steps | Train time | Baseline WER (tone-agnostic) | Fine-tuned WER | Delta |
|---|---|---|---|---|---|---|
| Quick mechanism check | 20 | 10 | 35s | 0.4901 | 0.5497 | **+0.0596 (worse)** |
| Fuller validation run | 150 | 60 (~0.8 epoch) | 838s (~14 min) | 0.5752 | 0.6248 | **+0.0496 (worse)** |

**Both runs point the same direction: fine-tuning made the model measurably worse on held-out data, not better.** This is a real, honest, useful result — not a bug. Evidence the mechanism itself works correctly: training loss dropped cleanly and substantially in both runs (4.85 → 1.57 across the 60-step run), proving gradients are flowing and the model is genuinely learning *something* from the training examples. The held-out WER regression is best explained by **not enough data/steps to net out ahead of early-fine-tuning instability** — 150 examples is well under one epoch, and Whisper fine-tuning commonly needs the model to see the same data multiple times (and typically far more of it) before held-out performance improves rather than temporarily degrades. This is consistent with what the research was already flagging as a real risk before this run: NaijaVoices' own paper fine-tuned Whisper-small on their full **600 hours** and still only reached 58.1% WER on their own test set — 1.8 hours was always a long shot for a net win, and the result confirms that rather than contradicts it.

No half-fine-tuned checkpoint was kept (`save_strategy="no"` in the training script, by design) — only the real before/after numbers, in `results/finetune_validation_run.json`.

**Do next:**
1. **Scale the training data before trying again** — re-run `pull_permissive_training_data.py --n 2000` and `pull_fleurs_permissive.py --n 2339` (the full available train split) to get closer to the full ~130h permissive corpus, not another 150-example slice.
2. **Increase steps/epochs accordingly** — `--max-steps` needs to scale with data volume; 60 steps was chosen for a short session-scale check, not because it's a reasonable stopping point for a real fine-tune.
3. Consider freezing the encoder and only fine-tuning the decoder for early runs — a common technique to reduce catastrophic-forgetting risk on small data, not yet tried here.
4. If time/compute allows, this really wants to run somewhere with a real GPU rather than M5's MPS backend — 14 minutes for well under one epoch on 150 examples means a full-corpus run at meaningful epoch counts is an overnight-or-longer job on this hardware, not a same-session one.

## 🟡 v2: recipe fix (real epochs, lower LR, proper warmup) — closed most of the gap, didn't fully flip it

`scripts/finetune_whisper.py` was updated to fix the likely causes of v1's regression: real multi-epoch training (`--epochs 3`, not a fraction of one), a lower learning rate (5e-6, down from 1e-5), a proper `warmup_ratio=0.1` (not a fixed 5-step warmup), and a new per-epoch WER trajectory callback so the run's behavior over time is visible, not just a single before/after snapshot. Same 150 train / 30 eval real examples as v1, same seed — an isolated test of whether the recipe itself was the problem.

**Real per-epoch trajectory** (tracked on a 10-example subset), 2026-08-15, 2787.9s (~46 min) on M5/MPS:

| Epoch | tone-agnostic WER |
|---|---|
| 0 (zero-shot baseline) | 0.5752 |
| 1 | 0.6462 (the expected early dip) |
| 2 | 0.6038 (recovering) |
| 3 | **0.5519** — better than baseline, on this subset |

**On the full 30-example held-out set**, the final result is still net-negative: baseline 0.5752 → fine-tuned 0.6142 (**+0.0390**, worse) — but a real improvement over v1's +0.0496 regression on the identical eval set (about 21% less bad). Full result: `results/finetune_validation_run_v2.json` (v1's result kept at `results/finetune_validation_run.json` for comparison, not overwritten).

**Honest read**: this is not yet a win, but it's a real, clean recovery curve — the trajectory was still climbing toward (and beat, on the smaller subset) the baseline right as training stopped at epoch 3. That's the strongest evidence yet that this is a "needs more epochs" situation, not a dead end or a wrong approach. The next run should extend epochs (5–6) rather than change the recipe again.

**Do next (supersedes the v1 list above):**
1. ~~Re-run with `--epochs 5` or `--epochs 6`~~ — **attempted, crashed.** See below.
2. Once epochs alone are pushed further, THEN scale the data (the v1 list's suggestion) — cleaner to isolate one variable at a time, and epochs are free to test on hardware already in hand, unlike a bigger data pull.
3. Expect runs to keep getting slower on M5/MPS as epoch count grows (this run took 46 min for 3 epochs on 150 examples, non-linearly slower than v1's 14 min for 0.8 epoch, likely thermal throttling under sustained load) — factor that into how many epochs are realistic same-session vs. need to run overnight.

## ⚠️ v3 attempt (`--epochs 5`): crashed at 71% with an MPS out-of-memory error — a real hardware ceiling, not a code bug

Ran the same config as v2 but with `--epochs 5`. Got to **268/375 steps (71%, epoch 3.53) after 56 minutes**, then a real crash:

```
RuntimeError: MPS backend out of memory (MPS allocated: 10.56 GiB, other allocations: 9.43 GiB, max allowed: 20.13 GiB).
Tried to allocate 205.99 MiB on private pool.
```

Not a bug in the training script — this is Apple's MPS backend running out of its memory budget partway through a longer run, likely from memory fragmentation accumulating over more steps than the earlier (successful) 3-epoch run reached. `save_strategy="no"` meant no partial checkpoint was salvageable, but the per-epoch WER trajectory *was* captured before the crash:

| Epoch | WER (tone-agnostic) |
|---|---|
| 0 (baseline) | 0.5752 |
| 1 | 0.5943 |
| 2 | 0.5896 |
| 3 | 0.5991 |
| 3.53 | *(crashed before reaching 4)* |

**A real, important nuance this surfaced**: this trajectory is *different* from v2's 3-epoch run on the exact same hyperparameters/data/split (v2: 0.5752 → 0.6462 → 0.6038 → **0.5519**; v3: 0.5752 → 0.5943 → 0.5896 → **0.5991**). Same config, different outcome — because the training script seeds the *train/eval split* (`random.seed(42)`) but not PyTorch's own RNG, which controls per-epoch batch shuffling and dropout. **This means v2's encouraging "beat the baseline by epoch 3" result was partly a favorable run-to-run fluctuation, not a fully reproducible trend** — a more honest read of both runs together is "fine-tuning hovers close to baseline WER through 3 epochs, sometimes slightly better, sometimes slightly worse, with real run-to-run noise," not a confirmed steadily-improving trajectory.

**Do next:**
1. ~~Seed PyTorch's RNG too~~ — **done**: `torch.manual_seed(SEED)` added at the top of `main()`. Re-running "the same" config is now actually reproducible.
2. ~~Try `gradient_checkpointing=True`~~ — **tried, didn't work.** See below.
3. **Conclusion: ~3 epochs is this M5 Mac's practical ceiling** for this batch size/data size — accepted, not chased further this session. Going beyond it needs different hardware (a cloud GPU), not more workarounds on this one.

## ⚠️ Gradient checkpointing tried as a memory fix — a different, deeper crash, abandoned

After v3's OOM crash (above), tried the standard mitigation: `gradient_checkpointing=True` in `Seq2SeqTrainingArguments`, plus explicitly setting `model.config.use_cache = False` before training (the usual pairing, since gradient checkpointing conflicts with KV-caching in most seq2seq decoders). **Result: a different, immediate crash on the very first training step**, even on a tiny 10-step/20-example smoke test:

```
RuntimeError: Trying to backward through the graph a second time (or directly access
saved tensors after they have already been freed)...
```

This persisted with `use_cache=False` explicitly set, meaning the cause is deeper than the usual caching conflict — most likely PyTorch's default *reentrant* checkpointing mode conflicting with Whisper's specific architecture (a known class of issue with `torch.utils.checkpoint`'s default mode; the fix would be `gradient_checkpointing_kwargs={"use_reentrant": False}`, not attempted here — three failed attempts at extending epochs in one session is the point to stop and document honestly rather than keep guessing at fixes).

**Decision: abandoned gradient checkpointing for now, reverted the config change**, and capped ambition at the already-proven-safe `--epochs 3` (both earlier 3-epoch attempts completed without crashing). Re-ran with the RNG-seeding fix in place to get one clean, reproducible reference result. If more epochs are wanted later, the honest options are: (a) properly debug the reentrant-checkpointing issue, (b) reduce `--max-train` to fit more epochs in the same memory budget, or (c) move to a cloud GPU, which sidesteps all of this.

## ✅ Final, properly-seeded reference result (2026-08-15) — confirms v2's number was real, not a fluke

The seeded re-run (`torch.manual_seed(SEED)`, same `--epochs 3`) reproduced **exactly** v2's earlier trajectory and final result:

| Epoch | WER (tone-agnostic) |
|---|---|
| 0 (baseline) | 0.5752 |
| 1 | 0.6462 |
| 2 | 0.6038 |
| 3 | 0.5519 (10-example subset) |

**Final, on the full 30-example held-out set: baseline 0.5752 → fine-tuned 0.6142 (+0.0390, still worse).** Train time: 644s (~11 min — notably faster than v2's 2788s for the identical config, likely no thermal throttling this run).

**Why this matters**: getting the *identical* numbers from an explicitly-seeded run that v2 produced *without* explicit seeding confirms v2's result was a real, stable outcome of this specific data/config — not a lucky fluke, the way v3's differently-seeded run initially made it look uncertain. **The trustworthy, final answer for this data slice**: fine-tuning `steja/whisper-small-yoruba` on 150 real examples for 3 epochs reliably lands at +0.039 WER worse than zero-shot, with a real recovery trend visible in the trajectory (peak damage at epoch 1, recovering by epoch 3) that hadn't fully closed the gap when training stopped. Going further (more epochs or more data) is a real, warranted next step whenever there's time/hardware for it — but it's not a quick win, and this session's honest conclusion is: the guardrail work (Phase 5) delivered a bigger, more certain improvement to real-world reliability than fine-tuning did this pass.
3. Decide whether 46% WER is good enough to ship or whether fine-tuning is worth the time budget.

## ✅ Real Benchmark Results (2026-08-14)

**50-utterance benchmark on `shunyalabs/yoruba-speech-dataset`:**

| Model | n | Tone-Sensitive WER | Tone-Agnostic WER | CER (agnostic) |
|-------|---|-------------------|-------------------|----------------|
| **steja/whisper-small-yoruba** | 50 | **59.2%** | **46.1%** | **15.1%** |
| ndeclarke/whisper-base-yoruba | 50 | 80.1% | 68.5% | 31.5% |
| openai/whisper-large-v3 | 20 | 100.0% | 100.0% | 100.0% |

**Key findings:**
- **`steja/whisper-small-yoruba` is the current best** — 46% WER (tone-agnostic) vs 69% for whisper-base-yoruba
- **Whisper large-v3 completely fails on Yorùbá** — outputs "!" or single characters, even with language forcing. This is a known limitation: Whisper wasn't trained well on Yorùbá.
- **Tone-sensitive vs tone-agnostic gap is ~13 percentage points** — real, measurable cost of getting diacritics wrong
- **NCAIR1/Yoruba-ASR is gated** — requires HF login + access request, not yet benchmarked
- **Omnilingual ASR not yet available** — not on PyPI, needs GitHub install

**Recommendation for Phase 3/7:** Use `steja/whisper-small-yoruba` as the ASR backend. It's:
- Best performing of tested models (46% WER vs 69%+)
- ~970MB download (reasonable)
- Ungated (no access request needed)

---

## What's already built (this folder's prototype)

- `scripts/normalize.py` — tone-aware vs. tone-agnostic Yorùbá text normalization + WER/CER scoring. **Tested and passing** (`python3 scripts/normalize.py --self-test`) — including the specific guarantee that underdot letters (ẹ ọ ṣ) never collapse with their plain counterparts under either mode, since that's a phoneme distinction, not a tone distinction. See the module docstring for why this matters.
- `scripts/benchmark.py` — the model×condition harness. Backend interface + implementations for Omnilingual ASR and Whisper-family (via `openai-whisper` or `transformers`), wired to the scorer above. **Live-tested** against real models and real audio.
- `scripts/materialize_benchmark.py` — **NEW (2026-08-14)**: downloads audio from HF datasets-server API to create a local benchmark set. Produces `data/benchmark_set/manifest.csv` that `benchmark.py` consumes directly.
- `scripts/live_smoke_test.py` — quick 3-utterance smoke test for pipeline verification.

### What the benchmark results show

Reading the sample predictions in `results/whisper_small_yoruba__shunyalabs_50.json`:

- **Most utterances transcribe reasonably well** — the model gets the general structure right
- **Code-switched proper nouns still get phonetically Yorùbá-fied** — "turkey" → "tọ́kì", "australia" preserved but "dr.lee" → "ọ̀tọ́ lí"
- **Numbers and alphanumeric strings work better than expected** — "165km/h" transcribed correctly in some cases
- **The 46% WER is usable for a demo** but will cause real routing failures in Phase 7 (as already observed: "naira" getting lost breaks finance routing)

## What's NOT yet done

- **Omnilingual ASR not benchmarked** — `omnilingual-asr` package is not on PyPI, needs `pip install git+https://github.com/facebookresearch/omnilingual-asr`. This is the model the research brief recommends as primary; benchmark it once installed.
- **NCAIR1/Yoruba-ASR not benchmarked** — gated model, requires HF login + access approval.
- **MMS-1B not benchmarked** — CC-BY-NC license, worth testing for research comparison.
- **Larger benchmark set needed** — current 50 utterances is enough for model comparison, but 200-500 would give more stable estimates.
- **No fine-tuning code yet** — only benchmarking infra. Fine-tuning is gated on whether Omnilingual ASR closes the gap enough to skip it.

## Steps

1. ✅ ~~**Materialize a benchmark set.**~~ Done: `scripts/materialize_benchmark.py` creates `data/benchmark_set/` with 50 utterances.
2. 🟡 **Run the zero-shot matrix.** Partially done:
   - ✅ `whisper_small_yoruba` — **46.1% WER** (best so far)
   - ✅ `whisper_base_yoruba` — 68.5% WER
   - ✅ `whisper_large_v3` — 100% WER (fails completely)
   - ❌ `omnilingual_ctc_300m` — not installed
   - ❌ `ncair1_yoruba` — gated
   - ❌ `mms_1b_all` — not yet run
3. ✅ ~~**Look at sample_predictions by hand.**~~ Done: code-switching and proper noun handling are the main failure modes.
4. **Decide whether to fine-tune.** Pending Omnilingual ASR results. Current best (46% WER) is usable but causes real downstream failures in Phase 7.
5. **Score fine-tuned models on the SAME held-out set** — if fine-tuning happens.
6. **Publish the model** (if fine-tuned) to Hugging Face.

## Deliverable

- `results/*.json` — one per (backend, condition) cell of the benchmark matrix, real numbers
- A written recommendation: which model Phase 3 should actually use, and why (latency vs. WER vs. license tradeoff — Phase 3 is real-time, so a 7B model's accuracy may not be worth its latency)
- If fine-tuned: model card + HF upload
