# Phase 1 — Data: Plan

**Goal:** a validated, licensed, documented Yorùbá speech corpus (audio + transcript pairs) ready to feed Phase 2's ASR benchmark/fine-tuning, with a manifest format Phase 2 can consume without re-deriving anything.

**Time budget:** 1–2 weeks (down from the original 3–4 — see why below).

---

## Status: Before → After (as of 2026-08-14) — 🟢 ~70%

- **Before:** plan only. FLEURS was the assumed data source but stalled with no root cause found; no real audio had been pulled or validated; licensing question unopened.
- **After:** FLEURS's stall root-caused (parquet row groups ~965MB, over the datasets-server's 300MB scan limit — not fixable, not worth waiting on). Found and validated a real working alternative (`shunyalabs/yoruba-speech-dataset`), built `hf_preview_client.py` (now reused by Phase 6 too) to preview/download HF data without the slow full `datasets` library. Pulled and validated a real manifest (10 rows) and a real audio sample (18.12s, 16kHz, verified with soundfile).
- **What's still open:** the licensing decision (research-only vs. commercial-clean vs. both) — this is a call only you can make, not something to derive from code. NaijaVoices registration is a manual step at naijavoices.com/membership.

**Update (2026-08-15): decision made — Option (c), both. The permissive-only half is now real, materialized data, not just a plan.** Hugging Face auth confirmed working (`HF_TOKEN` in `yoruba-prompt-experiment/.env`, verified via `HfApi().whoami()`). Pulled real bounded samples from both permissive sources:

- **Common Voice 17 (yo)**: the official `fsicoli/common_voice_17_0` repo turned out to be unusable — `datasets` 4.5.0 has fully dropped script-based dataset support (confirmed live: `RuntimeError: Dataset scripts are no longer supported`). Used `Abdullah804/common-voice-yoruba` instead, an ungated Parquet-native community re-export — verified as a faithful mirror by cross-checking its first row against a second, independent mirror (`Binaryy/yoruba-common-voice`): identical sentence, `client_id`, and Mozilla's own `common_voice_yo_<id>.mp3` filename convention. Pulled 300 real rows, all downloaded and duration-probed: **0.52 real hours** (mean 6.22s/clip). Neither mirror carries its own license tag — same "verify before non-smoke-test use" caveat already applied to shunyalabs.
- **FLEURS (yo_ng)**: streaming still fails the same way documented above (965MB row groups), confirmed again even through the full `datasets` library, not just the REST preview API. Worked around with a full non-streaming download of the whole train split (2,339 rows, ~15 min, ~3.3GB cached) — that succeeded, but hit a second real snag: `datasets` 4.5.0 wants `torchcodec` to auto-decode audio, not installed here. Worked around by loading the audio column with `decode=False` and decoding raw bytes with `soundfile` directly instead of adding a new heavy dependency. Materialized 300 real rows: **1.28 real hours** (mean 15.4s/clip).

**Combined permissive-only real sample so far: 600 real utterances, ~1.8 hours**, both sources downloaded, decoded, and verified (see `phase2-asr/data/permissive_train/`). This is a bounded session-scale sample, not the full ~130h permissive corpus — scaling further is mechanical from here (`--n` flag on both pull scripts), not blocked on anything new.

NaijaVoices registration remains the separate manual track for the other half of (c), not yet done.

**Do next:**
1. Make the licensing call — it gates whether Phase 5 (finance, likely commercial) can use this data path.
2. If commercial-clean is chosen, register for NaijaVoices access.
3. Scale the manifest past 10 rows using `scripts/build_manifest_from_preview.py` once the licensing decision is made (no point scaling data you might not be allowed to use commercially).

## What changed from the original plan

The original roadmap assumed you'd scrape Common Voice + BBC Yorùbá + YouTube from scratch to reach 10 hours. That's no longer the bottleneck: **NaijaVoices already has ~622,000 Yorùbá utterances across 3 batches** (verified live against the dataset's HF metadata, 2026-07-31). The work in this phase is **validation, licensing decisions, and manifest-building** — not collection.

## Confirmed facts about the candidate datasets (verified live, not from memory)

### NaijaVoices — primary candidate, but gated

- Repo: `naijavoices/naijavoices-dataset` on Hugging Face.
- **Gated.** You must register at [naijavoices.com/membership](https://naijavoices.com/membership) *and* accept the HF gate (name, email, confirmation checkboxes) before `load_dataset` will work — the earlier assumption that this streams freely was wrong. Do this first; it's a manual step nothing here can automate.
- **Yorùbá configs:** `yoruba-batch-0` (207,386 examples, ~47.6 GB), `yoruba-batch-1` (207,386 examples, ~47.6 GB), `yoruba-batch-2` (207,384 examples, ~47.6 GB). **~622,156 utterances, ~142 GB total for Yorùbá alone** if you pull all three batches. Same schema and similar scale exists for `igbo-batch-{0,1,2}` and `hausa-batch-{0,1,2}` (useful directly for Phase 6).
- **Schema:** `audio` (HF Audio feature), `speaker_id`, `text`, `language`, `gender`, `age_range`, `phase` (string — appears to be a data-collection-phase label, not a train/val/test split; there is no pre-made split, you'll need to carve your own held-out test set).
- **License: CC BY-NC-SA 4.0.** Non-commercial, share-alike. See the licensing decision below — **do not skip this.**
- Given the size, **do not download all three batches to a laptop.** Stream (`streaming=True`) and pull a bounded sample, or download one batch at most for local fine-tuning experiments.

### Alternatives that are NOT gated (verified live)

- `google/fleurs` config `yo_ng` — Google FLEURS, CC-BY-4.0, not gated at the dataset-info level (confirmed via `HfApi().dataset_info()`). Read-only test/dev set style corpus (~10 h). Good for **evaluation**, too small to be a primary training set. **Streaming a row from it stalled/timed out repeatedly in this project's dev sandbox** (multiple attempts, one over 10 minutes with no result — network-environment-specific, not necessarily a problem on your machine) — the actual parquet file resolves fine via `HfApi`, so this looks like a large-file-transfer issue in that specific sandbox rather than anything wrong with the dataset. **Re-verify the exact field names** (`inspect_datasets.py` currently guesses `raw_transcription` for the text field, based on FLEURS' typical schema, not a confirmed live read) before relying on it.
- `mozilla-foundation/common_voice_17_0` config `yo` — not gated at the dataset-info level, but Mozilla Common Voice repos on HF typically still require accepting Mozilla's terms via the HF UI once per account before streaming works (this is normal for all Common Voice languages, not Yorùbá-specific). CC0. ~120 h.

*(ÌròyìnSpeech and BibleTTS are not yet confirmed accessible via `datasets.load_dataset` — check their own release pages; treat as secondary until verified.)*

**Update (2026-08-15): FLEURS's `raw_transcription` field question is resolved.** Earlier notes flagged `raw_transcription` as "a guess from FLEURS' typical schema, not a confirmed live read." Now confirmed live by reading actual rows: `raw_transcription` is correct — it preserves capitalization and punctuation ("Dr.Lee ... Turkey ... A(H5N1)"), while the sibling `transcription` field is a lowercased/simplified variant. Both preserve diacritics. See `phase2-asr/scripts/pull_fleurs_permissive.py`.

### A third option, found while working around the FLEURS stall — verified live and actually working

While debugging why FLEURS wouldn't stream, a lighter-weight approach turned up a real, ungated, immediately-usable dataset: **`shunyalabs/yoruba-speech-dataset`** — 3,548 utterances (2,339 train / 378 validation / 831 test), 16kHz audio, real diacritic-preserved Yorùbá transcripts, **not gated**. Confirmed by actually downloading and validating a real audio file from it (`samples/shunyalabs_train_row0.wav`, 18.12s, verified with `soundfile`) and pulling a 10-row real manifest (`manifests/shunyalabs_train_sample.csv`, mean duration 14.19s/row — extrapolating that rate across all 2,339 train rows gives **~9.2 hours** for the train split alone, though that's an extrapolation from 10 rows, not a full-corpus measurement).

**⚠️ License: unstated.** The dataset's HF card has no `license` field and no descriptive README beyond the schema — this is a real gap, not an oversight in this doc. Do not use it beyond smoke-testing (which is what's been done so far) until you've contacted the dataset maintainer or found a license elsewhere. This is a sharper version of the Phase 1 licensing decision below: NaijaVoices' restriction is *known* (CC BY-NC-SA); this one's is *unknown*, which for a paper/product is arguably worse.

**The tool that found this, `scripts/hf_preview_client.py`, is worth using generally.** It hits `datasets-server.huggingface.co`'s REST API directly instead of going through the full `datasets` Python library's streaming machinery. Two concrete advantages observed: (1) it reveals *why* a dataset won't stream — FLEURS' failure mode turned out to be ~965MB parquet row groups exceeding the API's 300MB scan limit, an exact, actionable error rather than an indefinite hang; (2) for datasets that do work, it's a single HTTP request instead of spinning up the `datasets` library's full resolution machinery. Use `inspect_datasets.py` (the original prototype, using the `datasets` library) for datasets you already know work and want full streaming access to (like NaijaVoices, once gated); use `hf_preview_client.py` first when *evaluating* an unfamiliar dataset, since it fails fast and clearly instead of hanging.

## The licensing decision — make this now, in writing

Re-read `RESEARCH_OPEN_WEIGHT_ASR.md` §2. Pick one and record it below before pulling more than a smoke-test sample of NaijaVoices:

- **(a) Research-only** — use NaijaVoices freely, publish under CC BY-NC-SA, Phase 5 stays a non-commercial demo.
- **(b) Commercial-clean** — train only on FLEURS + Common Voice (~130 h total), skip NaijaVoices entirely, weaker model but Phase 5 can be a real product.
- **(c) Both** — train two model variants, report the WER cost of going permissive-only. This is itself a result worth putting in the paper.

> **Decision (recorded 2026-08-15):** **(c) — train both variants.** Chosen because the WER-cost-of-licensing comparison is itself a publishable result (see Phase 4), and Phase 5 can launch on the permissive-only model first without waiting on NaijaVoices registration. The permissive-only half (Common Voice 17 + FLEURS) is being set up now; NaijaVoices registration is the parallel manual track for the other half, not yet done.

### Follow-up research pass (2026-08-15) — real numbers behind each option, and proof (b)/(c) are viable, not just a fallback

A closer pass over the four real candidate datasets, with actual hour counts and license terms:

| Dataset | Hours (yo) | License | Commercial? | Role |
|---|---|---|---|---|
| NaijaVoices | ~600 | CC BY-NC-SA 4.0 | No | Primary training (option a/c) |
| Common Voice 17 | ~120 | CC0-1.0 | Yes | Commercial training (option b/c) |
| FLEURS (yo_ng) | ~10 | CC-BY-4.0 | Yes | Benchmark / eval only — too small to train on |
| shunyalabs/yoruba-speech-dataset | ~9 (est., extrapolated from a 10-row sample — see above) | Unstated | Unclear | Smoke-test only, as already documented above |

ÌròyìnSpeech and BibleTTS remain unverified secondary sources (not yet confirmed loadable via `datasets.load_dataset` — check their release pages before counting on them).

**NaijaVoices specifics** (from its HF dataset card): 1,800 hours total across Igbo/Hausa/Yorùbá (~600h Yorùbá), 5,000+ speakers, balanced gender/age. Split into three ~167h batches (`yoruba-batch-0/1/2`) — same schema for Igbo/Hausa, directly reusable by Phase 6. Schema: `audio, speaker_id, text, language, gender, age_range` — no pre-made train/val/test split, you carve your own. Full raw version is 500+ GB; a compressed **84GB version has identical data** — use that, not the raw batches, for local work. Still gated behind manual registration at naijavoices.com/membership plus an HF gate accept — nothing here changes that requirement.

**The important new fact: option (b)/(c) is not hypothetical.** A public reference fine-tune, NaijaVox-2.0 (a LoRA fine-tune of Whisper-large-v3), reached **22.3% WER on the FLEURS test set** training on FLEURS + ÌròyìnSpeech + Common Voice + WaxalNLP — **zero NaijaVoices data**. That's a huge gap below our current best (`steja/whisper-small-yoruba` at 46.1% WER, see `phase2-asr/PLAN.md`), and it's evidence the permissive-only ceiling is respectable, not a consolation prize. Their V1→V2 gain (28.8% → 22.3%) came mostly from adding more of exactly these permissively-licensed datasets plus augmentation — data volume is the lever, and it doesn't have to come from NaijaVoices. It also confirms FLEURS as the field's de-facto shared benchmark: whatever gets trained here should report WER on FLEURS test specifically, for comparability to this and other published numbers.

**This strengthens the case for (c)** — a permissive-only model is now a credible target with a known achievable floor (~22% WER, likely better once fine-tuned on this project's own pipeline and evaluation), not just an asterisk next to the "real" NaijaVoices-trained model.

**Do next, in order:**
1. Register at naijavoices.com/membership, accept the HF gate, set `HF_TOKEN`.
2. Pull the 84GB compressed NaijaVoices version, not the 500GB raw batches, for local/laptop work.
3. Carve the held-out test set by whole `speaker_id` before any fine-tuning — this is what makes WER claims trustworthy (see Steps §4 above, unchanged).
4. Verify ÌròyìnSpeech and BibleTTS's actual license and loadability before relying on them for the permissive-only variant.
5. Record the actual decision above once 1–2 are far enough along to know what's realistic on your hardware/timeline.

## What's real and done as of this pass (2026-08-13)

- `scripts/hf_preview_client.py` — built, live-tested against `shunyalabs/yoruba-speech-dataset` and against FLEURS (where it correctly surfaced the real error instead of hanging).
- `scripts/build_manifest_from_preview.py` — built, produced a real 10-row manifest with real probed durations: `manifests/shunyalabs_train_sample.csv`.
- `samples/shunyalabs_train_row0.wav` — a real downloaded, validated 18.12s audio file, used directly in Phase 2's live ASR smoke test (see `phase2-asr/PLAN.md` — this exact file produced a real 60% WER result against a real model).
- **Not yet done:** NaijaVoices registration (manual step, requires you personally), the full held-out test-set speaker carving, and the final licensing decision write-up below.

## Steps

1. **Register for NaijaVoices access** (naijavoices.com/membership) and accept the HF gate on `naijavoices/naijavoices-dataset`. Set `HF_TOKEN` in your environment (`huggingface-cli login` or export directly).
2. Run `scripts/inspect_datasets.py` — pulls a small streamed sample (default 20 rows) from whichever of {NaijaVoices, FLEURS, Common Voice} you have access to, reports duration/speaker/gender stats, and writes one manifest CSV per source into `manifests/`. This is the prototype; see below.
3. Decide sample size for Phase 2. You do not need all 622k Yorùbá utterances to get a meaningful fine-tuning result — start with one batch (~207k utterances, ~50 h if avg. clip is ~1s... **verify actual average clip duration from the inspection script, don't assume**), and scale up only if compute/time allows.
4. **Carve a held-out test set** now, before any fine-tuning happens, and never touch it again. Recommended: hold out entire `speaker_id`s (not random utterances) so the model can't leak speaker identity information — standard ASR practice, and directly relevant to your WER claims being trustworthy.
5. Write the **data card** (`DATA_CARD.md`) — sources, hours, license per source, speaker/gender/age distribution, known gaps (dialect coverage, domain — NaijaVoices `phase` field may indicate scripted vs. spontaneous speech; check before writing this).
6. Hand off `manifests/*.csv` to Phase 2 — the manifest schema is the contract between phases (see prototype output columns).

## Manifest schema (the Phase 1 → Phase 2 contract)

| Column | Meaning |
|---|---|
| `source` | `naijavoices` \| `fleurs` \| `common_voice` |
| `source_config` | e.g. `yoruba-batch-0`, `yo_ng`, `yo` |
| `row_index` | index within the streamed source, for re-fetching |
| `text` | ground-truth transcript, **NFC-normalized** (see Phase 2's `normalize.py` — do this once here, not repeatedly in Phase 2) |
| `duration_sec` | audio duration |
| `speaker_id` | if available, else empty |
| `gender` | if available, else empty |
| `license` | license tag for this row's source, carried through so downstream code can filter by it (important for the (c) track) |

## Deliverable

- `manifests/*.csv` — one per accessible source, real data (not placeholders)
- `DATA_CARD.md` — filled in, not a template
- A written licensing decision (above)
- A held-out test-set speaker list (`manifests/test_speakers_yoruba.txt` or similar) that Phase 2 must never train on
