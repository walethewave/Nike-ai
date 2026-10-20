# Naija Voice AI — Roadmap

**Status as of 2026-07-31.** The Yorùbá system-prompt experiment (originally "Phase 0") is **done** — see `yoruba-prompt-experiment/RESULTS.md`. It's a completed, standalone research artifact now, not a gate for anything below. This roadmap starts from Phase 1.

Every phase below has its own folder with a `PLAN.md` (exact research + steps) and a working prototype. Read `RESEARCH_OPEN_WEIGHT_ASR.md` before touching Phase 1 or 2 — it changes two assumptions from the original plan:

1. **You don't need to scrape audio for a month.** `naijavoices/naijavoices-dataset` on Hugging Face already has ~600 hours of Yorùbá speech, freely streamable.
2. **Whisper is not the obvious base model anymore.** Meta's Omnilingual ASR (Nov 2025, Apache 2.0, 1,600+ languages including `yor_Latn`) is newer, more permissively licensed, and purpose-built for exactly this problem.

Also see the licensing note in that doc — NaijaVoices is CC BY-NC-SA (non-commercial), which is a real decision point for Phase 5, not a footnote.

## Revised novelty claim

Not "first Yorùbá ASR" — that's false, and reviewers will say so (NCAIR1/Yoruba-ASR, ÒWE-Voice, and others already exist). The defensible claim:

> First open, reproducible, evaluated **end-to-end conversational voice agent** (speech-in → speech-out, with dialogue state) for Yorùbá, plus the first systematic benchmark of Meta's Omnilingual ASR on Yorùbá against Whisper/MMS/NCAIR1 under both tone-aware and tone-agnostic scoring.

## Phase overview

| Phase | Folder | What | Depends on |
|---|---|---|---|
| 1 | `phase1-data/` | Assemble + validate the Yorùbá speech training/eval corpus | — |
| 2 | `phase2-asr/` | Benchmark + fine-tune ASR (Omnilingual ASR primary, Whisper/MMS/NCAIR1 comparison) | Phase 1 |
| 3 | `phase3-conversational/` | End-to-end voice loop: ASR → LLM (Claude) → TTS (YarnGPT2) | Phase 2 |
| 4 | `phase4-paper/` | Write up + submit to AfricaNLP / arXiv | Phase 2 + 3 results |
| 5 | `phase5-finance/` | Domain-specific deployment on top of Phase 3 | Phase 3 |
| 6 | `phase6-multilingual/` | Extend Phase 2 benchmark + Phase 3 pipeline to Igbo/Hausa | Phase 2 + 3 |

## Timeline (unchanged cadence: ~2–3 weekday hours + weekend deep work)

| Phase | When | Time | Change from original plan |
|---|---|---|---|
| 1 — Data | Weeks 1–2 | **1–2 weeks** (was 3–4) | NaijaVoices removes most collection work; time now goes to validation/cleaning + a data card, not scraping |
| 2 — ASR | Weeks 3–6 | 3–4 weeks | Add Omnilingual ASR to the comparison; decide the NC-license track (a/b/c) before starting |
| 3 — Conversational | Weeks 7–11 | 4–5 weeks | Unchanged in shape; TTS pick is YarnGPT2 (not YarnGPT-local — superseded) |
| 4 — Paper | Weeks 12–17 | 5–6 weeks | Novelty claim narrowed (see above); related-work section now has real citations to work from (§3 of research brief) |
| 5 — Finance | Months 6–9 | 8–12 weeks | Gate on the Phase 2 license decision — NC data blocks a commercial deployment |
| 6 — Igbo/Hausa | Months 9–12 | 12–16 weeks | Cheaper than planned: Omnilingual ASR + NaijaVoices both already cover `ibo_Latn`/`hau_Latn`, so this is mostly re-running Phase 2/3 with different language codes, not new infrastructure |

## What to do this week

1. Read `RESEARCH_OPEN_WEIGHT_ASR.md` in full.
2. Go to `phase1-data/`, read its `PLAN.md`, run `prototype.py` to pull a small streamed sample of NaijaVoices Yorùbá and confirm the pipeline works end-to-end on your machine.
3. Make the licensing call (§2 of the research brief) — write your decision into `phase1-data/PLAN.md`'s "Decision" section before scaling up data pulls.
