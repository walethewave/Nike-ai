# Phase 3 — Generic Conversational System: Plan

**Goal:** speech-in → speech-out Yorùbá conversation with memory, as a small FastAPI service. **Depends on:** Phase 2's model recommendation for the ASR layer.

**Time budget:** 4–5 weeks.

---

## Status: Before → After (as of 2026-08-15) — 🟢 ~75%

- **Before (2026-08-14):** architecture prototype run end-to-end for the first time; ASR/TTS worked but the LLM leg failed with `403 PermissionDeniedError` — stale AWS credentials, confirmed via `sts.get_caller_identity()`.
- **After (2026-08-15): credentials fixed, full pipeline re-run for real.** Verified via two independent checks before re-testing: `sts.get_caller_identity()` now returns a valid identity (Account `672965015248`, IAM user `Dev`), and a direct `AnthropicBedrock` call against `us.anthropic.claude-haiku-4-5-20251001-v1:0` returned a real response — not just auth working, the actual Bedrock call Phase 3 needs works. Re-ran `scripts/live_pipeline_test.py` end-to-end: ASR transcribed the reference utterance **exactly** correctly this run, and the LLM generated a real, coherent, multi-sentence Yorùbá reply (no fallback text) in 8.65s, synthesized to a real 39.81s reply audio file. Full trace: `samples/live_pipeline_test_log.json`.
- **Correction:** `app/asr.py`'s `WHISPER_HF_MODEL_ID` default is already `steja/whisper-small-yoruba` (Phase 2's proven winner) — the switch already happened before this pass, not something still to do. Today's re-run used it: the ASR leg transcribed the reference utterance **exactly** correctly, consistent with Phase 2's 46.1% WER number being a real, usable floor.

**Do next:**
1. **Numeral-parsing reliability in the LLM leg is a confirmed, severe issue — and Phase 5 has since built and verified a fix pattern worth reusing here.** Phase 5's systematic test (`phase5-finance/scripts/numeral_parsing_test.py`, 2026-08-15) found the same Bedrock model (Claude Haiku 4.5, what this phase's `llm.py` also calls) gets only **5/14 (36%) correct** on real Yorùbá number words. **Update, same day**: Phase 5 built `app/yoruba_numerals.py`, a small deterministic Yorùbá-numeral parser that cross-checks and auto-corrects the LLM's output — verified 13/13 against the live model (see `phase5-finance/PLAN.md`). This phase's `llm.py` doesn't currently extract structured numeric fields the way Phase 5's does, so there's nothing to plug the guardrail into yet — but if this phase ever needs to reason over user-stated numbers (dates, quantities, prices), Phase 5's module is directly reusable rather than something to build from scratch.

## What's already built and LIVE-TESTED (this folder's prototype)

- `app/llm.py` — Claude via Amazon Bedrock, Yorùbá system prompt, multi-turn conversation memory. **Live-tested against real Bedrock credentials, 2026-07-31**: a 2-turn conversation ("my name is Wale" → "what's my name?") correctly round-tripped, with the model recalling "Wale" from session history. See the transcript in the build log — this isn't a mock.
- `app/main.py` — FastAPI app: `/health`, `/chat/text` (real, working), `/chat/voice` (full pipeline, needs ASR+TTS installed).
- `app/asr.py`, `app/tts.py` — **now each support two backends** (env-var-selected): the heavier, long-term-recommended one (Omnilingual ASR / YarnGPT2, per Phase 2's research — not yet live-tested here, multi-GB / manual setup), and a **lightweight, real, live-tested-in-this-repo alternative** (`ASR_BACKEND=whisper_hf`, `TTS_BACKEND=mms_tts`) — see the full pipeline test below.
- `app/tts.py`'s `mms_tts` backend also supports an opt-in `TTS_SEED` env var (2026-08-15) — MMS-TTS (VITS) was found to be genuinely stochastic (confirmed live: 5 identical-text synthesis calls produced 5 different audio outputs, root-caused in `phase7-integration/PLAN.md`). Setting `TTS_SEED` fixes PyTorch's RNG before each call, making output reproducible — proven 3/3 identical audio, and this is what let Phase 7 track a downstream bug (an unset LLM `temperature`) to its actual cause instead of blaming ASR/TTS variance. Unset by default so normal use keeps natural variation; set it for reproducible test runs.

### ⚠️ Full pipeline live-tested end-to-end, 2026-08-13 — with one real, currently-blocking finding

`scripts/live_pipeline_test.py` runs the **entire loop for real**: pulls a real Yorùbá utterance, transcribes it (real ASR), generates a reply (real LLM call), synthesizes it (real TTS), saves the output audio. Actually run — here's exactly what happened, unedited:

| Stage | Result | Time |
|---|---|---|
| Fetch input audio | Real utterance pulled from `shunyalabs/yoruba-speech-dataset` | 3.25s |
| ASR (`whisper_hf`) | `"àní ìrí àkókò gẹ́gẹ́ bí ìṣẹ̀lẹ̀ tó ń ṣẹ̀ lẹ́sẹ̀sẹ̀ láti ọjẹ́ wájú wọ́n á sí àkókò yìí lọ sẹ́yìn"` — recognizably close to the reference (`"a ní ìrírí àkókò gẹ́gẹ́ bí ìṣẹ̀lẹ̀ tó ń ṣẹ̀ lẹ́sẹsẹ láti ọjọ́ iwájú wá sí àkókò yìí lọ sẹ́yìn"`), several word-level errors but the sentence is legible — a noticeably better result than the earlier 70%-WER smoke test items in Phase 2, worth remembering as a reminder that WER varies a lot by utterance, not just by model | 11.3s |
| **LLM (Bedrock)** | **FAILED**: `PermissionDeniedError: 403 — "The security token included in the request is invalid."` The AWS credentials that worked on 2026-07-28/31 (see `yoruba-prompt-experiment/RESULTS.md` and this file's own multi-turn-memory test above) are **no longer valid** as of 2026-08-13 — confirmed independently via `sts.get_caller_identity()`, same `InvalidClientTokenId` error, not a bug in this code. **Action needed: refresh `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in `.env`** (rotated, expired, or revoked — check the AWS IAM console) before any further live LLM testing in this project, not just this phase. | 1.94s (to fail) |
| TTS (`mms_tts`) | Given a fallback Yorùbá string (since the LLM stage failed) — real synthesis, real 3.22s 16kHz audio, non-silent (peak amplitude 0.155). Saved to `samples/live_pipeline_test_reply.wav`. | 2.68s |

**Total wall-clock for the full loop (minus the failed LLM call): ~19s**, most of it ASR (11.3s, on CPU/MPS — GPU would cut this substantially) and audio fetch (3.25s, dataset-server round trip, not representative of a real deployment's local mic input). This is the first real latency data point for Phase 3's "measure latency per layer" step (Step 4 below) — rough, small n=1, but real.

**The LLM stage's success is not in doubt** — it's proven working with these exact credentials 2 days prior (this file's own test above, and the finance tracker's live tests in Phase 5). This is a credential-freshness issue, not an architecture or code problem. Re-run `scripts/live_pipeline_test.py` once `.env` has valid credentials to get a complete, uninterrupted real trace.

### Run it yourself

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in AWS credentials (same pattern as yoruba-prompt-experiment)
python3 -m uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/chat/text -H "Content-Type: application/json" \
     -d '{"text": "Bawo ni?"}'
```

**Gotcha hit while building this:** on this machine, the `uvicorn` binary on `$PATH` resolves to a *different* Python installation (Homebrew's) than the one `pip install` targets, so `import anthropic` silently failed inside the server even though it was installed. Fix: run `python3 -m uvicorn ...` instead of bare `uvicorn ...`, so the module resolves against the same interpreter you installed into. Worth checking for yourself with `which uvicorn` / `which python3` before assuming a "ModuleNotFoundError" means a package didn't install.

## Why English system prompt, and why Claude Haiku by default

Both decisions cite `yoruba-prompt-experiment/RESULTS.md` directly:
- System prompt is English (see `SYSTEM_PROMPT` in `llm.py`) because the experiment found no measurable quality difference from writing it in Yorùbá, across three model families. One less translation to maintain.
- Default model is Claude Haiku 4.5, not the strongest model tested (Kimi scored higher on faithfulness/groundedness in that experiment) — **this is a placeholder, not a final decision.** For a real-time conversational loop, latency matters as much as the offline-batch metrics that experiment measured. Before locking this in, re-run a *latency*-focused comparison (Haiku vs. Kimi vs. GLM, wall-clock per turn) — the accuracy-vs-speed tradeoff wasn't part of the original experiment's scope.

## Setup (ASR + TTS layers — not yet done)

### ASR (Omnilingual ASR)
```bash
pip install omnilingual-asr
```
Then set `ASR_MODEL_CARD=omniASR_CTC_300M_v2` (or whatever Phase 2 recommends) in `.env`. See `phase2-asr/PLAN.md` for the model comparison this choice should be based on — don't just take the smallest default forever.

### TTS (YarnGPT2)
Not pip-installable as of this writing — needs a git clone plus a separately-hosted checkpoint:
```bash
git clone https://github.com/saheedniyi02/yarngpt.git
pip install outetts uroman torch torchaudio
# Download the WavTokenizer checkpoint + config (see the yarngpt repo's README
# for current wget/gdown commands — these move, don't hardcode a URL here)
```
Then set `WAV_TOKENIZER_MODEL_PATH` / `WAV_TOKENIZER_CONFIG_PATH` in `.env` to wherever you downloaded them, and make sure the cloned `yarngpt` directory is on `PYTHONPATH` (or `pip install -e .` it if it has a `setup.py`).

## Steps

1. Install the ASR layer, confirm `/health` reports `"asr": "importable"`.
2. Install the TTS layer the same way.
3. Record yourself saying something in Yorùbá, hit `/chat/voice` with the file, confirm you get audio back that sounds right. This is the actual "does the demo work" test — do it before anything else in this phase.
4. **Latency measurement.** Time each layer separately (ASR ms, LLM ms, TTS ms) for a typical utterance. If total round-trip is uncomfortably slow for a "conversation," that's the thing to optimize next — likely candidates: smaller ASR model, streaming TTS instead of generate-then-play, or a faster LLM tier.
5. **Real-user testing.** Get 3–4 native Yorùbá speakers to actually use it. Record what breaks — a garbled ASR transcript, a TTS mispronunciation, a conversationally-weird LLM reply. This qualitative record is direct input to Phase 4's paper (limitations section) and Phase 2 (does the ASR error pattern point at a fine-tuning need you didn't see in the benchmark?).
6. **Build the 50-question evaluation set** mentioned in the original roadmap — but now with a concrete rubric: did ASR get the words right (tie to Phase 2's WER work), did the LLM answer in Yorùbá with no code-switching (tie to the code-switch heuristic in `yoruba-prompt-experiment/metrics.py` — genuinely reusable here), did a native speaker rate the spoken response as natural.
7. Record a short demo video once steps 3–6 are solid enough to not embarrass you live.

## Known limitations to fix before calling this "done"

- **In-memory session store.** Restarting the server loses every conversation. Fine for a demo, not for anything real — swap for Redis or a DB-backed store before Phase 5 builds on top of this.
- **No streaming.** The current `/chat/voice` waits for the full LLM response before starting TTS, and returns the full audio file at once. A genuinely conversational feel needs streaming at least on the TTS side (start speaking before the whole reply is synthesized).
- **No barge-in / interruption handling.** Real conversation lets you interrupt. Out of scope for the prototype, worth flagging as a limitation in the paper rather than pretending it's solved.
- **`/chat/voice` writes temp files to disk per request** — fine for a prototype, would need cleanup/streaming-in-memory for any real load.

## Deliverable

- Working `/chat/voice` end-to-end (audio in, audio out), demonstrated with real recordings
- Latency breakdown per layer
- Notes from real native-speaker testing sessions
- 50-question evaluation results
- Short demo video
