# Naija Voice AI — Simple Project Summary

*A plain-English recap: why we're building this, what we did, what worked, what didn't, and what's left.*

---

## 1. The Problem

Voice AI (Siri, Alexa, ChatGPT voice mode, etc.) barely works for Nigerian languages — Yorùbá, Igbo, Hausa. Existing speech-to-text models either don't support these languages at all, or support them badly:

- Big general models (like Whisper large-v3) basically **fail completely** on Yorùbá — they weren't trained on enough of it.
- Smaller, community fine-tuned models do better, but are still wrong roughly **half the time**.
- Even when speech-to-text works, the AI "brain" (the LLM) that's supposed to understand what was said **can't reliably understand Yorùbá numbers** — a dangerous bug if you're building something like a finance app.

So the real problem isn't one bug — it's that **every stage of the pipeline is a little bit broken**, and those small breaks stack up into something unusable for real users.

## 2. The Solution (our approach)

Instead of chasing one magic model, we built a **cascaded pipeline** — three stages, each one doing one job:

```
Speech in → ASR (speech-to-text) → LLM (understands + responds) → TTS (text-to-speech) → Speech out
```

Why this way, instead of one giant "speech-to-speech" model (like GPT-4o voice mode)?

- A true end-to-end speech-to-speech model needs **way more training data and compute** than exists for Yorùbá/Igbo/Hausa right now — companies like Meta needed huge budgets just for speech-to-*text* in these languages.
- The cascaded approach lets us **see and fix each stage separately** — if the LLM gets a number wrong, we can catch and correct it with a simple rule, without retraining the whole system.
- This is also the approach a real Nigerian company (Intron Health, makers of "Sahara") used successfully — see `SUPER.md` for that research.

## 3. What We Actually Did (in order)

### Step 1 — Picked the best existing ASR model (no training yet)
We benchmarked 3 Whisper-based speech-to-text models on real Yorùbá audio:

| Model | Word Error Rate (lower = better) |
|---|---|
| `steja/whisper-small-yoruba` | **46%** ✅ best |
| `whisper-base-yoruba` (community fine-tune) | 68% |
| `whisper-large-v3` (OpenAI, no Yorùbá training) | 100% — total failure |

**Takeaway**: a smaller model *fine-tuned on Yorùbá* beat a much bigger general model. Raw size doesn't help if the model never saw the language.

### Step 2 — Tried to fine-tune it ourselves to do better
We pulled ~600 real Yorùbá audio clips (legally reusable, not copyright-restricted) and tried to further fine-tune `steja/whisper-small-yoruba` — the winner from Step 1 — to push its accuracy higher.

**Result: it didn't work this time.**
- First attempt made things *worse* (46% → 62% error, due to bad training settings).
- A fixed, more careful attempt got close to the original but still landed **slightly worse than before training** (46% → 61%).
- We tried training longer (5 rounds instead of 3) to see if it would eventually win — it **crashed twice** (ran out of memory, then hit a software bug) before finishing.

**Why it's not a failure, just a limit**: we didn't have enough data (only 150 training clips) or a powerful enough computer (a MacBook, not a proper training server) to realistically beat an already-good model. This is an honest, documented dead end — not something to hide.

### Step 3 — Found a much bigger, scarier bug (and fixed it)
While testing a Yorùbá finance-tracking app (type "I spent ₦10,000 on food" in Yorùbá), we discovered the AI "brain" (the LLM) **gets Yorùbá numbers wrong most of the time** — and it does it *confidently*, with no warning sign.

- Tested 14 real Yorùbá number words → only **5 correct (36%)**.
- The same sentence, asked 3 times, gave **3 different wrong answers**.
- This is the kind of bug that would silently write the wrong amount into someone's financial records.

**The fix**: we built a small, simple, 100%-predictable Yorùbá number parser (no AI involved — just rules) that double-checks the AI's answer and corrects it if they disagree.

- Verified it against the real AI: **13 out of 13 correct**, including every case that was previously wrong.

### Step 4 — Found the bug *behind* the bug
Even with the fix in place, real end-to-end tests (speak → app replies) were still getting things wrong sometimes. We dug in and found **three separate causes stacked on top of each other**:

1. **The text-to-speech voice wasn't consistent** — saying the same sentence 5 times produced 5 slightly different audio clips, which sometimes garbled the number.
2. **The AI "brain" had no randomness control turned off** — it should always be 100% consistent for a task like "extract this number," but it wasn't set to be.
3. **Our number-parser's vocabulary had a gap** — a common ASR mishearing of "10" wasn't in its dictionary.

We fixed all three. **Result: 3 out of 3 real end-to-end tests correct**, instead of 0 out of 6 before.

### Step 5 — Tried the same system on Igbo and Hausa
- **Hausa**: got a real, usable (if weak) number — 65% error rate with a community model.
- **Igbo**: the one model we could actually get working **failed completely** (it started babbling repeated nonsense). Two other Igbo models we tried to use turned out to be broken/empty downloads on Hugging Face. Still an open problem.

### Step 6 — Wrote it all up for a research paper
Took the real benchmark numbers, the numeral bug story, and the fine-tuning results and wrote them into the paper's Results section — including the fine-tuning failure, reported honestly as a real research finding, not hidden.

---

## 4. Which Fine-Tuning, Exactly?

To be precise about "the fine-tuning":
- **Base model**: `steja/whisper-small-yoruba` (already a Yorùbá-tuned model — we fine-tuned *further* on top of it, not from scratch)
- **Data**: ~600 real Yorùbá audio clips pulled from Common Voice + FLEURS (legally reusable), 150 used for training, 30 held back to test on
- **Hardware**: a MacBook (Apple Silicon/MPS) — not a cloud GPU
- **Result**: accuracy got slightly *worse*, not better (46.1% → 61.4% word error rate), after multiple honest attempts to fix the training settings
- **Why we stopped**: more training (5+ rounds) kept crashing due to memory limits and a software incompatibility we didn't fully solve

---

## 5. Gaps & Limitations (being honest)

- **Fine-tuning didn't beat the starting point** — would need more data and/or a real GPU to possibly fix this.
- **Igbo has no working ASR model yet** in our setup — the only one we could run fails badly.
- **No numeral guardrail for Igbo/Hausa yet** — only Yorùbá has the number-checking fix so far.
- **No real native-speaker testing yet** — all our "it works" claims are from our own scripted tests, not real Yorùbá/Igbo/Hausa speakers using the app naturally.
- **The licensing situation for training data is still unresolved** for the higher-quality data source (NaijaVoices) — needs manual registration, not done yet.
- **A router bug**: if speech-to-text squishes two words together with no space, our safety checks can miss it. Known, not yet fixed.

---

## 6. What's Next

1. Run a real test with actual Yorùbá/Igbo/Hausa speakers (not just our own scripts).
2. Find or train a working Igbo ASR model — current one is unusable.
3. Extend the number-checking fix to Igbo and Hausa.
4. Finish the NaijaVoices data registration for a bigger, better training set.
5. Decide whether to chase more fine-tuning (needs a real GPU) or focus on growing the data instead.

---

*For the full, detailed version with all data tables and technical specifics, see `SUPER.md` and each phase's own `PLAN.md` file.*
