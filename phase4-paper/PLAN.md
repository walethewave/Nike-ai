# Phase 4 — Paper & Publication: Plan

**Goal:** a submitted paper (AfricaNLP workshop + arXiv preprint) with the narrowed, defensible novelty claim from `ROADMAP.md`.

**Time budget:** 5–6 weeks, overlapping with the tail of Phase 3.

---

## Status: Before → After (as of 2026-08-27) — 🟢 ~85%

- **Before:** bibliography entries were placeholders (`Unknown, VERIFY AUTHORS`) — not submittable.
- **After:** all 12 entries verified against live arXiv/ACL Anthology sources; one real factual error caught and fixed (an MMS-paper author's name was wrong from memory: "Maxime" → "Maryam" Fazel-Zarandi), one truncated title fixed. Two new entries added for datasets/models actually used elsewhere in the repo (`shunyalabs-yoruba`, `mms-tts`). `scripts/generate_results_table.py` confirmed working against real (not synthetic) Phase 2 output.
- **Update (2026-08-27) — Results section written from real data, no longer placeholders.** `06_results.tex` now has five real subsections: ASR benchmark (real 3-system table, `steja/whisper-small-yoruba` winning at 46.1% tone-agnostic WER vs. 68.5%/100%), a fine-tuning negative result (§ below), the numeral-parsing finding + fix + the TTS/temperature diagnostic chain that followed it (the paper's strongest result, written up in full), an honest placeholder for the still-not-run native-speaker naturalness study (explicitly *not* conflated with the numeral guardrail's 3/3 code-path check), and a baselines subsection. `generate_results_table.py` was broken by a schema collision (fine-tuning run logs now live in the same `phase2-asr/results/` directory as benchmark JSONs, and the script crashed trying to read both as one schema) — fixed by skipping non-benchmark files rather than crashing. Added `scripts/check_citations.py` (PLAN.md had claimed a "citation-consistency script" existed; it didn't — this is a real one now) — run before any submission, confirms every `\cite*` key resolves and every bib entry is used. Added 3 new live-verified bib entries needed for the results table (`whisper` — the original Radford et al. 2022 paper, cited by name throughout but never actually given an entry until now; `whisper-small-yoruba` and `whisper-base-yoruba-colab`, the two HF community fine-tunes directly benchmarked). `07_limitations.tex`'s single-run-evaluation and domain-mismatch paragraphs are also now real, not placeholders. **Still open, unchanged from before:** the native-speaker conversational evaluation (Phase 3 Steps 5–6) hasn't been run, so `06_results.tex`'s "End-to-End Conversational Results" subsection is still a `[FILL IN]`, and Phase 6's Igbo/Hausa numbers aren't in the paper yet either.

**Do next:**
1. ~~Write up the Yorùbá-numeral misparsing finding AND its fix~~ — **done (2026-08-27)**, `06_results.tex` §"Yorùbá-Numeral Parsing," including the downstream TTS-determinism/LLM-temperature diagnostic chain it led to.
2. ~~Re-run `generate_results_table.py` against Phase 2's real benchmark JSONs~~ — **done**, real 3-system table now in `06_results.tex` via `sections/_asr_table.tex`.
3. **Run Phase 3's native-speaker naturalness/task-success study** — this is now the one piece of the Results section that can't be written from data already in the repo; everything else that was blocked on "real numbers" now has them.
4. **Pull Phase 6's Igbo/Hausa numbers in once they exist** — the paper currently only covers Yorùbá numerically, even though the system claims multilingual scope.
5. Draft/tighten the Introduction and Related Work prose against the now-complete Results section — check the novelty claim (§ below) still holds against anything published since 2026-08-13's last literature check.

## Venue and timing — verified live, with an important correction

The original roadmap assumed "AfricaNLP typically runs alongside EACL/NAACL in early [year], deadline October–November." **That's not quite right for the cycle you're actually targeting.** Verified live (2026-07-31): the 6th AfricaNLP Workshop's submission deadline was **March 7, 2025** — i.e. AfricaNLP runs on a **spring** cycle (workshop co-located with ACL-family spring/summer conferences), not a fall→winter one. A 7th edition's exact 2026/2027 dates weren't published as of this check.

**Action, not assumption:** check `https://africanlp-workshop.github.io/cfp.html` (or the current year's AfricaNLP site) yourself before finalizing the Phase 4 timeline — don't plan against a November deadline that may not exist. If the spring pattern holds, a Phase 4 finishing in Nov–Dec 2026 would be aiming at a **~March 2027** deadline, not the roadmap's original November 2026 one. That's actually *more* runway, not less — good news if true, but confirm it.

**In the meantime:** arXiv has no deadline. Post the preprint as soon as Phase 2+3 results are solid, regardless of workshop timing — visibility doesn't need to wait on a CFP.

## The novelty claim (restated from ROADMAP.md — this is the thing the whole paper argues)

> First open, reproducible, evaluated end-to-end conversational voice agent (speech-in → speech-out, with dialogue state) for Yorùbá, plus the first systematic benchmark of Omnilingual ASR on Yorùbá against Whisper/MMS/NCAIR1 under tone-aware and tone-agnostic scoring.

Both halves of this claim are falsifiable by a reviewer with 10 minutes and Google Scholar — so before submitting, re-search for anything published between now and submission that might have closed either gap. Prior art moves fast in this space (ÒWE-Voice and the tone-in-ASR paper were both 2026 publications found during this project's own research pass).

## Structure (seeded, not placeholder text — see `paper/sections/`)

Standard ACL-workshop shape: Abstract, Introduction, Related Work, Data, Methodology, Experiments, Results, Limitations, Ethics/Broader Impact, Conclusion. Each section file has real citations and real framing already, not "TODO" stubs — but the actual *findings* (Results section) can't be written until Phase 2/3 produce them.

## LaTeX template — get the real one, don't trust a hand-rolled class file

`paper/main.tex` in this repo uses plain `article` as a **structural placeholder** so the section files compile and you can see page count/flow. **Before submission**, download the actual ACL template from the ACL Anthology's official template repo (search "ACL style files" / "acl-style-files" on GitHub — the exact repo path changes by year, don't hardcode one here) and swap `\documentclass{article}` for the real `\documentclass[11pt]{acl}` (or workshop-specific variant). Do not submit with the placeholder class.

## Bibliography — now fully author-verified, not placeholder

`paper/references.bib` has **12 entries, all with real, live-verified author lists** as of 2026-08-13 — not the earlier "Unknown, VERIFY AUTHORS" placeholders. Verification method: the 6 arXiv papers (Omnilingual ASR, NaijaVoices, Voices Unheard, the African-ASR literature review, MMS, ÌròyìnSpeech) were checked against `export.arxiv.org/api/query` directly; the 2 ACL Anthology papers (ÒWE-Voice, Tone-in-Yorùbá-ASR) were checked against their live Anthology pages — **which also corrected the Tone-in-Yorùbá-ASR title**, previously truncated to "Tone in Yoruba ASR: Evaluating the Impact of..." in earlier research notes, now the full "...Tone Recognition on Transformer-Based ASR Models." One real error caught this way: the MMS paper's author list (recalled from training knowledge, not re-checked at the time) had "Maxime Fazel-Zarandi" where the real name is "Maryam Fazel-Zarandi" — fixed. The two HF-model-card entries (NCAIR1/Yoruba-ASR, YarnGPT2) aren't academic papers, so "author" means the organizational/individual attribution read directly off each model card, not a guess. Two new entries were added for artifacts used directly in this project's live testing: `shunyalabs-yoruba` (the dataset used across Phase 1–3's smoke tests) and `mms-tts` (the TTS backend used in Phase 3's live pipeline test) — both cited in `03_data.tex`/`04_methodology.tex`, not just parked in the bib file unused.

**Update (2026-08-27):** three more entries added, needed once the real ASR benchmark table went into `06_results.tex` — `whisper` (Radford et al. 2022, the original paper, verified live against the arXiv API; had been referenced by name in `02_related_work.tex` and `01_introduction.tex` but never actually given a bib entry) and `whisper-small-yoruba` / `whisper-base-yoruba-colab` (the two HF community fine-tunes directly benchmarked — verified live by reading each model card, same "not an academic paper" caveat as `ncair-yoruba-asr`/`yarngpt2`). **15 entries total, all live-verified.** The "citation-consistency script" this doc referenced before turned out not to actually exist in the repo — `scripts/check_citations.py` is now a real script (run `python3 scripts/check_citations.py`); as of 2026-08-27 it confirms all 15 defined bib entries are cited and all cited keys resolve, zero missing, zero unused.

**Remaining before actual submission:** exact publication years for four HF-model-card entries weren't stated on their cards (marked `% approximate` in the `.bib` comments) — low-stakes but still worth pinning down by asking the maintainers or checking repo creation dates.

## Results table generator (this folder's prototype)

`scripts/generate_results_table.py` reads Phase 2's benchmark JSON output (the exact schema `phase2-asr/scripts/benchmark.py` produces) and emits both a LaTeX table and a Markdown table. Originally tested against synthetic sample data (`scripts/sample_results/`); first run against real data on 2026-08-13 against a 3-utterance smoke test only (not a publishable result).

**Update (2026-08-27):** `phase2-asr/results/` now also contains fine-tuning run logs (`finetune_validation_run*.json`, a different schema with no `"scores"` key) alongside the real benchmark JSONs — running the script against the whole directory crashed with `KeyError: 'scores'`. Fixed: `load_results()` now skips any file missing the `scores` key and prints which files it skipped, instead of crashing. Re-run against the real benchmark matrix (3 systems, 6 result files after the 20/50/3-utterance re-runs) and confirmed correct; output hand-curated (one row per system, its largest-$n$ run) into `paper/sections/_asr_table.tex`, now `\input`ed directly by `06_results.tex` rather than hand-copied numbers.

## Steps

1. Confirm the actual current AfricaNLP CFP + deadline (don't trust the date above without re-checking closer to submission time).
2. As Phase 2/3 results land, run `generate_results_table.py` and paste output into `06_results.tex`.
3. Write Results, Limitations sections from real data — this is the only part of the paper that can't be pre-written.
4. Verify every bibliography entry's author list against the source before submission (see the incomplete-author flags in `references.bib`).
5. Swap in the real ACL class file.
6. Submit to arXiv first (no deadline pressure), then the workshop.

## Deliverable

- Submitted (or arXiv-posted, workshop-pending) paper
- `references.bib` with verified authors
- Results section populated from real Phase 2/3 numbers, not placeholders
