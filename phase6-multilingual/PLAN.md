# Phase 6 — Igbo & Hausa Extension: Plan

**Goal:** extend Phase 2's ASR benchmark and Phase 3's conversational pipeline to Igbo and Hausa. **Depends on:** Phase 2 + Phase 3 being solid for Yorùbá first — this phase is deliberately cheap *because* it reuses their code rather than rebuilding.

**Time budget:** 12–16 weeks (mostly data/eval work, not new engineering — see below).

---

## Status: Before → After (as of 2026-08-27) — 🟢 ~72%

- **Before:** Igbo/Hausa support in Omnilingual ASR was an open question (`verified=False` in `scripts/lang_config.py`), no real data pulled for either language.
- **After:** confirmed Igbo/Hausa support by fetching Omnilingual ASR's actual source file (no package install needed) — `lang_config.py` now marked `verified=True` for both. Pulled real manifests for both (`manifests/hausa_sample.csv`, `manifests/igbo_sample.csv`). Found Meta's own published per-language CER numbers: Hausa 3.5%, Igbo 9.1%, Yorùbá 9.0% — real evidence that tonal languages (Yorùbá, Igbo) are harder than non-tonal ones (Hausa), which is this phase's core hypothesis.
- **What's still open:** the Igbo dataset's transcript field is literally the string `"unknown"` for the `transcript` column (the usable text is in a different field, `script`) — a real data-quality caveat, not yet resolved. Keyword lists (reused by Phase 7's router) are not native-speaker-verified.

**Update (2026-08-27) — Step 3 done: first real WER/CER data points for both languages, and a real, honest negative result for Igbo.** New script `scripts/live_smoke_test_multilingual.py` mirrors Phase 2's `live_smoke_test.py` pattern exactly (same REST preview client, same scorer, same output schema), parameterized per language:

- **Hausa: real, usable result.** `CLEAR-Global/whisper-small-clearglobal-hausa-asr-1.0.0` (ungated, cc-by-sa-4.0, a real complete checkpoint fine-tuned on the TWB Voice 1.0 Hausa subset) against 5 fresh `shunyalabs/hausa-speech-dataset` utterances: **65.1% WER, 36.0% CER** (tone-sensitive and tone-agnostic identical — expected, standard Hausa orthography in this data has no acute/grave tone marks for the scorer to strip, consistent with Hausa being non-tonal in writing). Sample outputs show the model getting the topic/register right but missing specific words and proper nouns — a real, if weak, working baseline. `results/live_smoke_test_hausa.json`.
- **Igbo: the working model catastrophically fails; two other candidates were dead ends.** `benjaminogbonna/whisper-tiny-igbo` (ungated, apache-2.0, real checkpoint) against 5 fresh `SilencioNetwork/igbo-speech` utterances produced degenerate, repetition-loop output ("...ọkpụtọrọrụọrụọrụọrụ...", "mamamama...", repeated dozens of times) — **WER 2412%, CER 4645%** (both >100% because insertions from the repetition loops dwarf the reference length). This is a real, reproducible failure, not a scoring artifact: read the sample predictions in `results/live_smoke_test_igbo.json` directly. Two other Igbo candidates that looked promising in an HF search turned out to be genuine dead ends, worth flagging for anyone repeating this search: `anthonysoronnadi/whisper-fine-tuned-on-igbo` and `isurut/whisper_base_finetune_cv_igbo` are both **empty repositories** (confirmed via the HF API — only a `.gitattributes` file, 0 bytes of actual model weights, despite looking like real entries in search results), and `Danieljava/whisper-small-finetuned-asr-igbo-yoruba-data-...` has a malformed `tokenizer_config.json` (`extra_special_tokens` stored as a list instead of a dict) that crashes `transformers` 4.57.6's tokenizer loading regardless of the `protobuf` dependency it appears to need — not usable without hand-patching the checkpoint's config, which wasn't attempted. `NCAIR1/Igbo-ASR` exists and is a real, complete whisper-small checkpoint, but is gated (`"gated": "auto"` on the HF API), same pattern as `NCAIR1/Yoruba-ASR` in Phase 2.
- **Root-cause note, not yet disentangled:** Igbo's failure could be model quality (whisper-tiny is small and this fine-tune's own training quality is unverified), or the data itself (these are `free_speech`-type recordings 52-68s long — much longer than Hausa's ~15s read-sentence clips — and Whisper's pipeline needed an experimental `chunk_length_s=30` long-form path just to avoid crashing outright, which the library itself warns is "very experimental" for seq2seq models), or the reference-text mismatch already flagged (the `script` field is a discussion prompt, not a verified transcript). Given the empty/broken state of the other two Igbo candidates, a same-size, cleanly-packaged comparison model to isolate which cause dominates isn't available yet without more search.
- **Honest bottom line:** Hausa now has one real, weak, usable baseline number; Igbo does not — its only working checkpoint fails completely on this data, and this pass didn't produce a second Igbo data point to compare against because the alternatives were broken, not because the investigation stopped early.

**Note (2026-08-15):** Phase 5 found the Bedrock LLM (Claude Haiku 4.5) only correctly parses 5/14 (36%) of real Yorùbá number words in a systematic test, then built and verified a deterministic guardrail against it (`app/yoruba_numerals.py`, 13/13 against the live model) — see `phase5-finance/PLAN.md`. Igbo and Hausa almost certainly have their own equally complex numeral systems (Igbo is also base-20-ish with its own irregularities); worth assuming the same class of bug exists here too until tested. If it does, Phase 5's module is a template for the fix (small explicit vocabulary + auto-correct on disagreement), not something to design from scratch — though the actual Igbo/Hausa number words would need separately verifying, the same way Yorùbá's were.

**Do next:**
1. ~~Get a first real WER/CER data point per language~~ — **done for Hausa (65.1% WER); Igbo blocked on finding a working, non-tiny checkpoint** (see above) — worth one more targeted search pass (or waiting on NaijaVoices gated access) before concluding Igbo community tooling is simply behind Yorùbá's.
2. Resolve the Igbo dataset's transcript-field caveat — either find a cleaner source or confirm `script` is usable as ground truth. Now compounded by the free-speech-length issue found this pass.
3. Get native-speaker review of the Igbo/Hausa keyword lists before Phase 7 trusts them for real routing.
4. Once Phase 2's benchmark infra is pointed at Igbo/Hausa data, run the same 3-model matrix here for real (not just published numbers) to confirm Meta's numbers hold on this repo's actual data.

## Why this is cheap — and what's now PROVEN, not just claimed (updated 2026-08-13)

The original roadmap treated Igbo/Hausa as "repeat the fine-tuning process... collect data... fine-tune Whisper for each" — implying comparable effort to Phase 1+2 combined, twice. That's not true, and this pass closed the two open questions from the previous version of this doc:

1. **NaijaVoices already has Igbo and Hausa data at the same scale as Yorùbá** — `igbo-batch-{0,1,2}` and `hausa-batch-{0,1,2}`, same schema, same ~600h-per-language scale (verified live against the dataset's HF metadata alongside the Yorùbá configs — see `phase1-data/PLAN.md`).
2. **✅ Omnilingual ASR's Igbo/Hausa support is now CONFIRMED, not assumed.** Fetched the real source file directly from GitHub (`src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py`) without installing the multi-GB package — `yor_Latn`, `ibo_Latn`, and `hau_Latn` are all present in `supported_langs`. `lang_config.py`'s `verified` flags are now `True` for all three languages, reflecting this.

**Bonus finding, not originally planned for**: Omnilingual ASR's own repo ships `per_language_results_table_7B_llm_asr.csv` — Meta's own reported CER per language for their 7B LLM-ASR model:

| Language | Training hours (Meta's corpus) | Reported CER |
|---|---:|---:|
| Hausa (`hau_Latn`) | 836.4 | **3.5%** |
| Igbo (`ibo_Latn`) | 729.6 | **9.1%** |
| Yorùbá (`yor_Latn`) | 828.1 | **9.0%** |

This is a real, useful early anchor for the comparative-analysis step (Step 4 below): **Hausa is non-tonal; Yorùbá and Igbo are both tonal — and Hausa's reported CER is ~2.5× lower**, directly consistent with (though not proof of, different eval sets, not this project's own held-out data) the tonal-languages-are-harder-for-ASR hypothesis this phase exists to test. Cite Meta's own numbers as prior evidence in the paper; still run the real comparative benchmark on your own shared eval set before treating this as settled.

This folder's prototype (`scripts/run_multilingual_benchmark.py`) **proves the reuse claim concretely**: it imports `phase2-asr/scripts/benchmark.py` and `normalize.py` directly and runs the identical harness across all three configured languages. Dry-run tested — confirmed all three languages produce correct, independent results through the shared code path, with **zero language-specific benchmarking logic written**.

## Real data pulled for all three languages, 2026-08-13 (not just Yorùbá anymore)

Using the same `hf_preview_client.py` technique from Phase 1 (proven more reliable than full `datasets`-library streaming — see that phase's PLAN.md for why), found and pulled real, working preview data for Hausa and Igbo too:

- **Hausa**: `shunyalabs/hausa-speech-dataset` (same publisher, same schema as the Yorùbá one already in use) — 5 real rows pulled, mean duration **15.9s**, real Hausa transcripts (e.g. *"nasarorin da vautier ta samu marasa alaka da bada umarni ba..."*). Manifest: `manifests/hausa_sample.csv`.
- **Igbo**: `SilencioNetwork/igbo-speech` (config `igbo_nigeria`, **not** `default` — different schema shape, discovered by checking `/splits` first) — 5 real rows pulled, mean duration **62.0s** (much longer than the Hausa/Yorùbá read-sentence clips — this dataset is *free speech* recordings, richer metadata too: gender, ethnicity, dialect, recording location, even background noise source tags). Manifest: `manifests/igbo_sample.csv`. **Caveat worth flagging honestly**: the `transcript` field for this dataset is literally `"unknown"` — what's usable as reference text is the `script` field (the prompt the speaker was asked to read/discuss), which is not necessarily what they actually said, especially for a *free_speech* type entry. Treat this dataset as eval-adjacent, not train-ready, until that's resolved.
- **License status for both, same as Yorùbá's `shunyalabs` source**: unstated on the dataset card. Smoke-test only, same caveat as Phase 1.

## What's NOT yet done

- **Whisper language codes** (`ig`, `ha` in `lang_config.py`) are standard ISO 639-1 codes and should work with `openai-whisper`'s `language=` parameter the same way `yo` does in Phase 2 — not independently re-verified here, low risk given ISO 639-1 codes are what Whisper's own documentation uses, but worth a spot check.
- ~~No real ASR inference has been run on the Igbo/Hausa samples yet~~ — **done for both (2026-08-27)**, see the Step 3 update above. Hausa got a usable number; Igbo's only working checkpoint failed catastrophically, so Igbo still effectively needs this.
- **Phase 3's conversational pipeline is Yorùbá-only** (`SYSTEM_PROMPT` in `phase3-conversational/app/llm.py` is hardcoded Yorùbá). Extending to Igbo/Hausa needs a per-language system prompt and a language-selection mechanism in the API — not built here, since it depends on Phase 3 being stable first.
- **No NaijaVoices Igbo/Hausa manifest yet** — same gating requirement as Yorùbá (see `phase1-data/PLAN.md`), needs the registration step done once, covers all three languages.

## Steps

1. ~~Run `scripts/verify_lang_support.py`~~ — done via direct source-file fetch instead (no install needed); `lang_config.py` updated with real `verified=True` flags.
2. Pull Igbo and Hausa manifests from NaijaVoices the same way Phase 1 did for Yorùbá, once registered/gated access is set up (the `shunyalabs`/`SilencioNetwork` manifests above are smoke-test-only, unstated license — not a substitute for licensed training data).
3. ~~Run a Phase-2-style `live_smoke_test.py` for Igbo and Hausa~~ — **done (2026-08-27)**, `scripts/live_smoke_test_multilingual.py`. Hausa: real 65.1% WER baseline. Igbo: the one working checkpoint found catastrophically fails (see Step-3 update above); worth a second search pass for a working non-tiny Igbo model before treating this as settled.
4. Run `scripts/run_multilingual_benchmark.py --manifests-dir ... --backend omnilingual_ctc_300m` for real, producing `results/<backend>__igbo.json` and `results/<backend>__hausa.json` alongside Phase 2's Yorùbá results — directly comparable since it's the identical scorer.
5. **Write the comparative analysis** — this is the actual research contribution of this phase, per `ROADMAP.md`: does the tonal-vs-non-tonal gap seen in Meta's own numbers (above) replicate on your own held-out set? Does the same base-model ranking from Phase 2's Yorùbá results hold across languages, or flip?
6. Extend Phase 3's `SYSTEM_PROMPT` to be per-language (a small dict keyed by `lang_config.LANGUAGES`, not a rewrite) and add a language parameter to `/chat/text`/`/chat/voice`.
7. Repeat Phase 3's native-speaker evaluation with Igbo and Hausa speakers.

## Deliverable

- ✅ Verified language support table — **done**, all three confirmed
- ✅ Real preview data for all three languages — **done**, manifests in `manifests/`
- Benchmark results for Igbo and Hausa, same schema as Phase 2's Yorùbá results — 🟡 Hausa done (one real smoke-test number, 65.1% WER); Igbo blocked on finding a working checkpoint (smoke-tested one, it failed catastrophically). Full Omnilingual ASR matrix (Step 4) still not started for either.
- Comparative analysis (tonal vs. non-tonal, ranking stability across languages) — Meta's own numbers give a head start (see above), your own held-out benchmark is still the thing to publish
- Phase 3's pipeline working in 3 languages, not 1
