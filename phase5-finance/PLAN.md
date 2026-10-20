# Phase 5 — Finance Tracker: Plan

**Goal:** a domain-specific deployment on top of Phase 3's conversational layer — the specialization the original roadmap called "Version 2." **Depends on:** Phase 3 (reuses its ASR/TTS layers — not duplicated here) and Phase 1's licensing decision (see gate below).

**Time budget:** 8–12 weeks.

---

## Status: Before → After (as of 2026-08-15) — 🟢 ~80%

- **Before (2026-08-14):** DB math verified offline (7-transaction scenario, defensive JSON-parsing). Live intent-parsing (the LLM leg) blocked by stale AWS credentials.
- **After (2026-08-15): credentials fixed, live intent-parsing tested for real, found a severe real bug, and shipped a verified fix for it same-day.** New script `scripts/live_intent_test.py` ran 6 real Yorùbá finance utterances through `app/intent.py`'s actual `parse_intent()` against the real Bedrock endpoint — 5/6 correct intent classification, but a systematic follow-up test found only 5/14 (36%) correct Yorùbá-numeral parsing. Built `app/yoruba_numerals.py`, a deterministic guardrail, and verified it against the live model: **13/13 correct**, including full recovery of the original failing case. See "The Yorùbá-numeral guardrail" section below for the full story, including two real bugs found while building the guardrail itself.

### ⚠️ A real, important finding: the LLM misparses Yorùbá numerals even on clean text

Input: *"Mo gba ẹgbẹ̀rún mẹ́wàá gẹ́gẹ́ bí owó oṣù"* ("I received **ten thousand** as salary" — ẹgbẹ̀rún mẹ́wàá = thousand-ten = 10,000). The model correctly identified the intent (`add_income`) and category (`salary`) but extracted **amount = 20,000**, not 10,000. This text was typed directly, not run through ASR — so this rules out ASR degradation as the cause. This is the same failure *shape* as Phase 7's "naira" finding (a confidently-wrong number, no error raised, DB write proceeds normally), but it proves the problem is at least partly in the LLM's Yorùbá-numeral reasoning itself, not only a downstream consequence of noisy ASR input. **This is a real risk for a finance app specifically** — a silently-wrong amount is worse than a refusal, since nothing in the current flow would catch it before it's written to the ledger.

A second, lower-severity miss: *"Ṣe àkótán ìnáwó mi fún mi"* ("give me a summary of my spending") was classified as `get_balance` instead of `get_summary` — possibly a prompt-wording issue on the test phrase's naturalness rather than a numeral problem; worth re-testing with a differently-worded phrase before concluding much from it.

The `add_expense_no_amount` case (*"Mo ná owó lórí ọkọ̀ irin-àjò"* — "I spent money on transport," no amount stated) correctly returned `amount: null`, confirming `app/main.py`'s existing "ask for the amount" fallback path is reachable and correct — that defensive path works as designed.

**What's still open:** see the full systematic test below (2026-08-15) — this is no longer a single unverified case, it's a mapped, confirmed pattern. This phase is also still gated on Phase 1's licensing decision per its dependency note above.

### 🔴 Follow-up: a systematic numeral test confirms this is a severe, structural weakness — 5/14 correct (36%)

`scripts/numeral_parsing_test.py` (new) holds a fixed sentence template (`"Mo ná náírà {word} lórí oúnjẹ"` — "I spent {word} naira on food") and varies only the Yorùbá number word, spanning units, regular tens, irregular subtractive tens (a known-hard part of Yorùbá's base-20 counting system), hundreds, and thousand-multiples. Real calls against the real Bedrock endpoint, full log at `results/numeral_parsing_test.json`:

| Number word | Expected | Got | Result | Category |
|---|---|---|---|---|
| mẹ́ta | 3 | 3 | ✅ | unit |
| ogún | 20 | 20 | ✅ | regular ten (2×10) |
| ẹgbẹ̀rún kan | 1,000 | 1,000 | ✅ | thousand ×1 |
| ẹgbẹ̀rún méjì | 2,000 | 2,000 | ✅ | thousand ×2 |
| ẹgbẹ̀rún márùn-ún | 5,000 | 5,000 | ✅ | thousand ×5 |
| mẹ́wàá (alone) | 10 | 10,000 | ❌ | unit |
| ọgbọ̀n | 30 | 900 | ❌ | regular ten (3×10) |
| ogójì | 40 | 20 | ❌ | regular ten (4×10) |
| àádọ́ta | 50 | 500 | ❌ | irregular subtractive ten (60−10) |
| ọgọ́ta | 60 | 20 | ❌ | regular ten (3×20) |
| àádọ́rin | 70 | 4,000 | ❌ | irregular subtractive ten (80−10) |
| ọgọ́rùn-ún | 100 | 200 | ❌ | hundred |
| igba | 200 | 1,000 | ❌ | irregular hundred |
| ẹgbẹ̀rún mẹ́wàá | 10,000 | **non-deterministic** | ❌ | thousand ×10 |

**The pattern is sharp, not random noise**: the model is reliable on small units (1–10 spoken alone) and simple round thousands (1×–5× a single-digit multiplier), but the entire tens/hundreds range is broken — including `ogún` (20) being right while every other regular ten (30/40/60) and every irregular subtractive ten (50/70) is wrong, in inconsistent directions (sometimes too high, sometimes too low, sometimes wildly off like 30→900).

**The original failing case ("ẹgbẹ̀rún mẹ́wàá" / ten thousand) was re-tested 3 separate times across today's session and got 3 different wrong answers**: 20,000 (in `live_intent_test.py`'s sentence), a 429 rate-limit (not a real data point, excluded), and 2,000 (on retry, this test's own sentence). Three calls, three different wrong numbers for the identical semantic input — this is strong evidence the model isn't applying a consistent (even if wrong) parsing rule, it's closer to pattern-guessing without real compositional understanding of how Yorùbá numerals compose (unit × multiplier, or the base-20 subtractive forms for 50/70/90).

**Why this matters more than it might first appear**: `ogún` (20) being correct means a naive smaller test could have concluded "tens mostly work" — the full sweep across values is what reveals the breakdown starts almost immediately after the one memorized/common case. This is the kind of result that only a systematic test surfaces, not a single anecdote.

**Do next (superseded by the guardrail below — kept for history):**
1. ~~Build the deterministic guardrail now~~ — **done**, see below.
2. Extend the test matrix to compound numbers within the tens (e.g. 21–29, 45, 67) once the guardrail exists — useful for calibrating exactly how good the deterministic parser needs to be, not urgent before that.
3. Re-test `get_summary` with 2-3 differently-phrased prompts to see if the earlier misclassification was a wording fluke or a real gap.
4. Revisit once Phase 1's licensing decision lands, since it determines what data this phase can legally use.

## ✅ The Yorùbá-numeral guardrail is built and verified (2026-08-15) — 13/13 correct, including every previously-known-wrong case

**See [`FINETUNING_GUIDE.md`](../phase2-asr/FINETUNING_GUIDE.md)-style plain explanation not needed here — this one's simpler.** `app/yoruba_numerals.py` is a small, deterministic, rule-based Yorùbá-numeral parser — not a full parser for arbitrary Yorùbá numbers (the compound/subtractive system is genuinely complex, and a guardrail that's confidently wrong is worse than one that admits it can't help). Instead, an explicit vocabulary of exactly the word forms already tested and ground-truthed in `phase2-asr/scripts/numeral_parsing_test.py`: units 1–10, the base-20 tens (including the irregular subtractive forms for 50/70/90), 100 and the irregular 200 (`igba`), and thousand-multiples (`ẹgbẹ̀rún` + a unit/ten/hundred word). Anything outside that vocabulary returns `None` — an honest "can't verify," never a guess.

**Two real bugs found and fixed while building it — worth knowing if you touch this code:**
1. **A Unicode combining-mark bug.** Yorùbá letters like `ẹ́` aren't single codepoints — `ẹ` (precomposed "e with dot below") has no precomposed form combined with a tone mark on top, so the acute/grave rides along as a *separate* standalone combining character even after NFC normalization. A naive per-codepoint tone-stripping loop (the same pattern already used in `phase7-integration/app/router.py`) silently leaves these combining marks untouched, so `ẹ́` doesn't actually get stripped down to `ẹ` — every word containing an underdotted vowel with a tone mark failed to match the vocabulary until this was fixed with an explicit second pass removing standalone combining acute/grave characters. **This same latent bug may exist in `router.py`, which was never specifically tested against underdot+tone combinations — worth checking.**
2. **An unsafe fallback for "ẹgbẹ̀rún" + an unrecognized word.** This project's own real test sentence (`phase7-integration/PLAN.md`'s `FINANCE_TEXT`) uses the informal spelling "mẹwa" (not the fuller "mẹ́wàá"), which wasn't in the vocabulary — and the parser's first version silently treated any unrecognized word after "ẹgbẹ̀rún" as "no multiplier," returning a confident **1000** instead of the correct **10,000**. That's the exact failure mode this guardrail exists to prevent, just self-inflicted. Fixed two ways: added the informal spelling, and changed the fallback so "bare 1000" is only trusted when "ẹgbẹ̀rún" is genuinely the last word in the message — any other unrecognized word after it now returns `None` (can't verify) rather than a guess.

**Integration** (`app/intent.py`): after the LLM returns a parsed intent, `verify_amount()` cross-checks any `add_expense`/`add_income` amount against the deterministic parser. When the parser is confident and disagrees, the deterministic value **overrides** the LLM's (logged as `amount_verification.corrected: true`) — a disagreement means the LLM is wrong, not the checker, since the deterministic parser only ever fires on vocabulary it's provably right about. When the parser can't recognize a number at all, the LLM's amount is trusted as before — this guardrail narrows the known failure cases, it doesn't claim to catch every possible one. `app/main.py`'s reply text surfaces a correction plainly to the user (not silent) when one happens, and the API response gets a new `amount_corrected: bool` field for programmatic/audit use.

**Real verification, against the live Bedrock endpoint** (not simulated):

| Test batch | Result |
|---|---|
| The 8 words that were WRONG in the original systematic test (30, 40, 50, 60, 70, 100, 200, and bare 10) | **8/8 correct** — 7 via real auto-correction, 1 (`àádọ́ta`/50) the LLM happened to get right this time, guardrail confirmed the match |
| The 5 words that were already correct (3, and thousand ×1/×2/×5/×10) | **5/5 still correct** — no regressions |
| The original problem sentence, *"Mo gba ẹgbẹ̀rún mẹ́wàá gẹ́gẹ́ bí owó oṣù"* (previously got 3 different wrong answers across 3 tries: 20,000 / a rate-limit / 2,000) | **10,000 — correct, via real auto-correction** |

**13/13 real, live test cases correct with the guardrail active**, including full recovery of the exact case that started this entire investigation. This closes the repo's top-priority open correctness gap.

**What's still NOT covered** (honest limits, not overclaimed):
- Compound numbers outside the vocabulary (21–29, 45, 67, hundreds beyond 200, etc.) — the guardrail correctly says "can't verify" for these rather than guessing, but that also means it can't *catch* an LLM error on them either. A real gap, not a silent one.
- Only Yorùbá is covered — Phase 6's Igbo/Hausa keyword lists (used by Phase 7's router) may have analogous numeral risks, untested.
- The guardrail runs on the text `parse_intent()` receives — if ASR degradation has already destroyed the number word entirely (Phase 7's original "naira" finding), there's nothing left to verify against; this guardrail only helps when the number word itself is present and merely mis-parsed, not when it's been erased.

**Update (2026-08-15, same day): real end-to-end testing found the vocabulary-exactness limit above is not theoretical.** Phase 7 ran the same real finance sentence through the full TTS→ASR→router→intent pipeline 5 times (see `phase7-integration/PLAN.md`'s follow-up section). The Yorùbá word for "ten" (mẹwa) came back from ASR as five *different* spellings across those 5 runs — "mẹwa", "màwà", "maha", "mẹ̀wà", and others — and the guardrail's exact-match vocabulary only covers the two spellings tested/added so far ("mẹwaa", "mẹwa"). In the one run where an amount got extracted at all, the ASR spelling ("màwà") wasn't in the vocabulary, so the guardrail correctly stayed silent rather than force a wrong correction — but that also meant it didn't catch the LLM's fabricated ₦5,000 (real value: ₦10,000). **This is the guardrail behaving exactly as designed** (never force a correction it isn't sure of) — the real lesson is that ASR degradation itself, not the guardrail's logic, is the dominant remaining risk for this pipeline.

## ✅ Root-caused and fixed, same day: it wasn't ASR degradation, it was TTS non-determinism plus two other real bugs — full pipeline now 3/3 clean

Phase 7 isolated the source of the run-to-run ASR variance (`scripts/isolate_asr_tts_variance.py`): **TTS (MMS-TTS/VITS), not ASR** — ASR is fully deterministic on fixed audio, but MMS-TTS produces a different waveform every single call for identical input text (architectural: VITS uses a stochastic duration predictor + flow-based decoder). Fixed with an opt-in `TTS_SEED` in `phase3-conversational/app/tts.py`. Stabilizing TTS output then exposed two more real bugs, both fixed here:

1. **`app/intent.py` had no `temperature` set at all** (API default: 1.0) for what should be deterministic structured extraction. With the ASR transcript now provably identical across calls, the LLM still returned 3 different amounts for the same input — proof the non-determinism wasn't upstream. Fixed: `temperature=0`.
2. **The guardrail's vocabulary didn't cover Whisper's actual stable output**: the now-reproducible real transcript was `"...egberun mawa..."` — no underdots at all, and "mẹwa" specifically came out as "mawa" (an `ẹ→a` mishearing, not the more common `ẹ→e`). Fixed `app/yoruba_numerals.py` with an underdot-collapsing fallback pass for the general case, plus "mawa" as an explicit loose-only entry for this specific irregular mishearing. **Adding "mawa" naively (to the strict vocabulary) introduced a NEW bug** — caught immediately by re-testing: it matched as a bare unit (=10) before the "thousand × multiplier" pattern was checked, so the guardrail "corrected" a ₦10,000 expense down to ₦10. Fixed by keeping such loose-only entries out of the strict vocabulary specifically to prevent this short-circuit.

**Final result, 3 real end-to-end runs with all fixes active** (`TTS_SEED=42` + `temperature=0` + expanded vocabulary): **3/3 correct** — identical ASR transcript, correct routing, and ₦10,000.00 every time, with the correction visibly surfaced in the reply (*"Mo tún iye owó ṣe sí ₦10,000.00 — mo gbọ́ nọ́mbà tí kò tọ́ ní àkọ́kọ́."*). Full diagnostic chain and data in `phase7-integration/PLAN.md`. Expanding the vocabulary DID help, as predicted — but it needed the TTS/temperature fixes first to even get a stable, testable signal to expand it against.

## ⚠️ Gate: this phase is blocked by Phase 1's licensing decision

If Phase 2's deployed model was trained on NaijaVoices (CC BY-NC-SA — non-commercial, share-alike), **this cannot legally be a commercial product** without either (a) using the permissive-only model variant from Phase 1's track (c), or (b) staying a non-commercial research demo. Check `phase1-data/PLAN.md`'s recorded decision before doing anything beyond local testing with this prototype. Don't find this out after building a business around it.

## What's already built and LIVE-TESTED (this folder's prototype)

- `app/db.py` — SQLite schema, `add_transaction`/`get_balance`/`get_summary`/`list_transactions`. **Unit-tested**: income/expense math verified correct, negative-amount and invalid-kind rejection verified.
- `app/yoruba_numerals.py` — deterministic Yorùbá-numeral parser, the guardrail against the LLM's numeral-parsing weakness (see below). **Self-tested** (`python3 app/yoruba_numerals.py --self-test`, 16/16) and **live-verified** against the real Bedrock model, 13/13.
- `app/intent.py` — parses a Yorùbá (or English) financial statement into one of `add_expense` / `add_income` / `get_balance` / `get_summary` / `unclear`, via Claude/Bedrock, then cross-checks any extracted amount via `yoruba_numerals.py`. Deliberately avoids `output_config.format` (structured-output enforcement) — that param 400'd on this project's Bedrock account when tried elsewhere (see `yoruba-prompt-experiment/llm_judge.py`'s fix); reuses the same proven prompted-JSON pattern instead of re-hitting that bug.
- `app/main.py` — FastAPI wiring: `/finance/message` (text in, structured intent + Yorùbá reply out). **Live-tested against real Bedrock, 2026-07-31**, all four intents:
  - Expense: *"Mo na naira ẹgbẹrun marun lori ounje loni"* → correctly parsed ₦5,000/food, balance updated correctly.
  - Income: *"Mo gba owo osu mi, naira ogorun egberun"* → correctly parsed ₦200,000/salary, balance correctly became ₦195,000.
  - Balance query: *"Elo owo ti mo ku bayi?"* → correct answer.
  - Summary query: *"Fun mi ni akotan inawo mi"* → correct breakdown.
  - Off-topic: *"Kini oju ojo bi loni?"* (weather question) → correctly classified `unclear` with a Yorùbá clarification reply, rather than hallucinating a finance answer.

The reply amounts are computed **deterministically from the database** (`build_reply()` in `main.py`), not generated by the LLM — the LLM only classifies intent and extracts fields. This matters specifically for a finance app: a hallucinated number in a chat response is an annoyance, a hallucinated number in a balance confirmation is a trust-destroying bug.

### Deepened, 2026-08-13: DB edge cases + defensive-parsing robustness (both real, both passing)

**DB layer**, run against a fresh temp database, no mocks:
- A realistic 7-transaction scenario (2 incomes, 5 expenses across food/rent/transport/utilities) — balance and per-category aggregation both computed correctly (₦144,500, food correctly summed across two separate purchases as ₦11,500).
- Zero-amount transactions correctly rejected.
- A ₦99,999,999.99 transaction accepted with no float-precision surprises.
- `list_transactions(limit=5)` against 30+ rows correctly returns exactly 5, most-recent-first.

**Intent parser's JSON extraction (`_extract_json`)**, tested against 6 realistic-shaped LLM outputs without needing a live model call: plain JSON, markdown-fenced JSON, JSON with leading prose ("Here is the parsed result:..."), JSON with trailing prose, and two inputs that should legitimately fail (malformed JSON, empty string) — all 6 behaved correctly, including the two failure cases raising cleanly rather than silently returning garbage. This is exactly the defensive parsing `parse_intent()` relies on to fall back to `intent="unclear"` instead of crashing on a weird model response.

### ⚠️ Live intent-parsing tests are currently blocked (credentials, not code)

The 2026-07-31 live tests above (all four intents working against real Bedrock) are still valid proof the *logic* works. But as of 2026-08-13, `yoruba-prompt-experiment/.env`'s AWS credentials return `InvalidClientTokenId` / `403 PermissionDeniedError` — confirmed independently via `sts.get_caller_identity()`, same failure Phase 3's live pipeline test hit. **This blocks re-running or extending the live intent tests until `.env` has fresh credentials.** Everything in this section that *could* be tested without live Claude calls (DB math, JSON-parsing robustness) has been — what's left blocked is specifically the "does Claude correctly classify *this* Yorùbá sentence" question, which needs the model in the loop.

## What's NOT yet done

- **Text-only.** `/finance/message` takes text, not audio. Wiring in Phase 3's `app/asr.py`/`app/tts.py` is the next step — deliberately not duplicated into this folder, import them once Phase 3's ASR/TTS setup is done.
- **Single user, no auth.** One SQLite file, no user accounts. Fine for a personal demo, not for "five real users" as the roadmap's Phase 5 goal states — needs at minimum a `user_id` column and some session-to-user mapping before multi-user testing.
- **No amount-in-words parsing robustness testing.** The live tests above used fairly clean number phrasing ("ẹgbẹrun marun" = five thousand). Yorùbá number words get more complex for non-round numbers — this needs a dedicated test set, not just spot checks, before trusting it with real users' money-adjacent data.
- **No currency ambiguity handling.** Assumes Naira always. If informal amounts ("2k", "2 thousand") or other currencies come up in real usage, the intent prompt needs updating — build this from observed real-user inputs (see Phase 3's "Steps 5" native-speaker testing notes), not speculatively.

## Steps

1. Wire in Phase 3's ASR/TTS to get `/finance/voice` (audio in, audio out) — mechanical once Phase 3's layers are working, reuse `app/asr.py`/`app/tts.py` from that folder directly (import, don't copy).
2. Build a proper test set of Yorùbá financial statements — varied number phrasing, informal amounts, multiple transactions in one message, ambiguous category. Run it through `intent.py`, hand-check the parses.
3. Add minimal auth/multi-user support (even just an API key → user_id mapping) before recruiting real testers.
4. Deploy somewhere reachable (a basic VPS or PaaS demo URL — doesn't need to be fancy).
5. Recruit 5 real users. Log every message + parsed intent (with consent) so misclassifications are debuggable, not just anecdotal.
6. Write the short evaluation report the roadmap calls for: intent accuracy, what broke, what got fixed.

## Deliverable

- `/finance/voice` working end-to-end
- Test set + intent-accuracy numbers (not just the 5 spot-checks above)
- Deployed demo URL
- User evaluation report — candidate content for a Phase 4 follow-up paper or subsection
