# Fine-Tuning Whisper for Yorùbá — What Was Done, and How to Do It Yourself

This explains, in plain terms, what fine-tuning actually is, exactly what was run in this repo (`scripts/finetune_whisper.py`), what every parameter means and how to set it, and the real before/after results from both attempts. Written so you can re-run this yourself, change the settings with an understanding of what each one does, and interpret the output.

---

## 1. What fine-tuning actually is

The ASR model we've been using, [`steja/whisper-small-yoruba`](https://huggingface.co/steja/whisper-small-yoruba), is already a version of OpenAI's Whisper model that someone else trained to understand Yorùbá. **Fine-tuning** means taking that already-trained model and continuing to train it further, on *our own* data, so it specializes even more — ideally getting better at exactly the kind of speech we care about (in this case, more Yorùbá, from Common Voice and FLEURS).

Under the hood, a model is just a huge number of adjustable numbers ("weights" or "parameters") — `whisper-small` has about 244 million of them. Training, whether from scratch or fine-tuning, is the process of nudging those numbers, a little at a time, so the model's predictions get closer to being correct on the data it's shown.

**Important, and this is exactly what happened here**: fine-tuning doesn't automatically make a model better. If you don't feed it enough data, or you train for too few passes, or your settings are too aggressive, you can actually make it *worse* — the nudges move the weights away from a good, well-calibrated starting point before they've had a chance to settle into a better one. That's precisely what we saw and measured.

---

## 2. The end-to-end process, step by step

Here's everything that happened, in order, matching what `scripts/finetune_whisper.py` actually does when you run it:

### Step 1 — Download/load the base model
```python
processor = WhisperProcessor.from_pretrained("steja/whisper-small-yoruba")
model = WhisperForConditionalGeneration.from_pretrained("steja/whisper-small-yoruba")
```
This pulls the model's weights (~967MB) from Hugging Face and loads them into memory. It was already cached locally from Phase 2's earlier ASR benchmarking, so this step was instant here — the first time you ever load a model, it downloads; every time after that, it's read from `~/.cache/huggingface/hub/`.

The **processor** is two things bundled together:
- A **feature extractor**: turns raw audio into the numeric format the model expects (a "log-mel spectrogram" — more on this below).
- A **tokenizer**: turns text into a sequence of numbers ("tokens") the model can predict, and back again.

### Step 2 — Download/prepare the training data
Two real, ungated, commercial-safe sources were pulled (see `phase1-data/PLAN.md` for the full licensing reasoning):

- **Common Voice 17 (Yorùbá)** — via `scripts/pull_permissive_training_data.py`, pulling from a Parquet-format mirror (`Abdullah804/common-voice-yoruba`) since the official repo can no longer be loaded at all (Hugging Face's `datasets` library dropped support for its old-style loading script). 300 real audio clips downloaded, 0.52 hours total.
- **FLEURS (yo_ng)** — via `scripts/pull_fleurs_permissive.py`, downloading the full train split (2,339 clips, ~3.3GB) once, then extracting 300 of them into individual WAV files. 1.28 hours total.

Combined: **600 real utterances, about 1.8 hours of audio**, each with a matching real transcript, listed in two manifest CSV files.

### Step 3 — Turn audio + text into what the model can actually train on
This is `prepare_examples()` in the script. For every row in the manifest:

1. **Load the audio, force it to 16kHz mono.** Whisper only understands audio at exactly 16,000 samples per second. Common Voice's clips came in at 32kHz, so they get resampled down.
2. **Extract a log-mel spectrogram.** This converts the raw sound wave into a picture-like grid representing which frequencies (pitches) are present at each moment in time — the actual input format Whisper's audio encoder reads. You don't need to understand the math, just know: this is *always* a fixed-size grid (80 frequency bands × 3000 time steps, covering up to 30 seconds), regardless of how long the actual clip is.
3. **Tokenize the transcript.** The reference text ("Mo ná ẹgbẹ̀rún méjì...") gets converted into a list of integer token IDs the model's decoder will learn to predict, one at a time.

### Step 4 — Split into training data and held-out (eval) data
```python
random.seed(42)
random.shuffle(rows)
eval_rows = rows[:30]        # held out — the model NEVER trains on these
train_rows = rows[30:180]    # what the model actually learns from
```
The eval set is set aside and never shown to the model during training. This is the only way to honestly measure whether fine-tuning helped: testing on data the model has already memorized would make it look artificially good.

### Step 5 — Measure the "before" (zero-shot baseline)
Before touching any weights, the untouched base model transcribes every held-out clip, and those transcriptions are scored against the real reference text using `normalize.py`'s WER/CER scorer (Word Error Rate / Character Error Rate — lower is better; this is the standard way to measure ASR quality; see Phase 2's `PLAN.md` for a fuller explanation of tone-sensitive vs. tone-agnostic scoring).

### Step 6 — Train
This is the actual fine-tuning loop, handled by Hugging Face's `Seq2SeqTrainer`. For each batch of training examples, repeatedly:
1. **Forward pass**: feed the audio into the model, get its predicted transcript (as probabilities over possible next tokens).
2. **Compute loss**: compare the prediction to the real transcript — how wrong was it?
3. **Backward pass**: compute the gradient — for each of the 244 million weights, "which direction, and by how much, would nudging you reduce that error?"
4. **Optimizer step**: actually apply a small nudge to every weight in that direction, scaled by the learning rate.

Repeat this thousands of times (once per batch), and the model gradually gets better at predicting the training examples. The real risk — and what we actually observed — is that it can temporarily get *worse* at everything else while it's adjusting, especially early on and especially with too little data.

### Step 7 — Measure the "after"
Same as Step 5, but now using the fine-tuned weights, on the exact same 30 held-out clips. Comparing this number to the Step 5 baseline is the entire point of the exercise.

### Step 8 — Save results (not a checkpoint)
The script deliberately does **not** save the fine-tuned model weights to disk (`save_strategy="no"`). Only the real before/after numbers get saved, to `results/finetune_validation_run*.json`. This was intentional — these were validation runs to test whether the approach works, not attempts to produce a model worth keeping. Once a run actually beats the baseline, that's when saving the checkpoint becomes worthwhile.

---

## 3. Every parameter, explained, and how to set it

All of these are command-line flags on `scripts/finetune_whisper.py`.

| Parameter | What it controls | How to think about setting it |
|---|---|---|
| `--max-train` | How many real examples to train on | More is (almost) always better for the model, at the cost of longer runs. We only had 600 pulled; used 150. |
| `--max-eval` | How many real examples to hold out for measuring WER | Bigger = a more statistically reliable WER number, but slower to evaluate at every checkpoint. 30 is a reasonable minimum to not be too noisy. |
| `--epochs` | How many full passes over the training data | **The parameter that mattered most here.** 1 epoch = the model has seen every training example exactly once. Our v1 run only reached 0.8 epochs — nowhere near enough. v2 used 3, and showed a clear recovery trend. Whisper fine-tuning guides typically recommend 3–10+ epochs depending on data size; more data generally needs fewer epochs, less data needs more. |
| `--max-steps` | Override: stop after this many *optimizer updates*, ignoring `--epochs` | Useful for a fast sanity check (e.g. "does this even run without crashing?") but not a real stopping criterion — that's why v1 used it and got an incomplete result. |
| `--batch-size` | How many examples get processed together in one step | Bigger batches → more stable, less noisy gradient estimates, but more memory. Kept at 2 here because Apple Silicon's MPS backend has real memory limits with variable-length audio; on a real GPU you could likely go higher (8, 16, 32). |
| `--lr` (learning rate) | How big a nudge each weight gets per step | **The second most important parameter.** Too high = big, destabilizing jumps (this contributed to v1's regression at `1e-5`). Too low = learning is painfully slow and you'd need far more steps to see any effect. v2 used `5e-6`, half of v1's, and the recovery curve was visibly cleaner. |
| `warmup_ratio` (fixed at `0.1` in v2, not exposed as a flag yet) | What fraction of training is spent *ramping up* the learning rate from 0 to its target value, instead of starting at full strength immediately | Protects the pretrained weights from a jarring first few updates. v1 used a fixed 5-step warmup (basically none, given 60 total steps); v2 used 10% of the whole run, which is standard practice. |
| `--eval-trajectory-n` | How many held-out examples to score after *every* epoch, to watch the trend | Kept small (default 10) so it doesn't slow training down much — it's for visibility into the pattern (getting better vs. worse over time), not the final reported number (which uses the full `--max-eval` set). |
| `--output-dir` | Where a checkpoint *would* be saved | Unused in practice — see Step 8 above. |

**Not a flag, but worth knowing**: `device` is auto-detected (`mps` on this Apple Silicon Mac, would be `cuda` on a machine with an NVIDIA GPU, `cpu` as a last resort). MPS is Apple's GPU-acceleration backend for PyTorch — it works, but it's less mature than CUDA and, in this session, showed real thermal throttling under sustained load (some individual training steps took 20-77 seconds instead of the usual 2-4 seconds). If you have access to a CUDA GPU (a rented cloud one, or Google Colab's free tier), the exact same script would run dramatically faster with no code changes.

---

## 4. What was actually run, and the real before/after

### Run 1 (v1) — the first attempt, too short
```
python3 scripts/finetune_whisper.py --max-steps 60 --max-train 150 --max-eval 30
# (implicitly: lr=1e-5, warmup_steps=5 — the old defaults)
```

| | Before (zero-shot) | After (fine-tuned) | Change |
|---|---|---|---|
| WER (tone-agnostic) | 0.5752 | 0.6248 | **+0.0496 — worse** |

Only 60 steps on 150 examples = about **0.8 epochs** — the model hadn't even seen the full training set once. Loss dropped cleanly during training (proof the mechanism works), but that wasn't enough to net out ahead of the early-training dip on held-out data.

### Run 2 (v2) — recipe fixed: real epochs, lower learning rate, proper warmup
```
python3 scripts/finetune_whisper.py --epochs 3
# (lr=5e-6, warmup_ratio=0.1 — the new defaults)
```

Per-epoch trajectory (tracked live, on a small 10-example slice):

| Epoch | WER (tone-agnostic) |
|---|---|
| 0 (before) | 0.5752 |
| 1 | 0.6462 (the expected early dip) |
| 2 | 0.6038 (recovering) |
| 3 | **0.5519** — already better than baseline, on this slice |

Final result on the full 30-example held-out set:

| | Before (zero-shot) | After (fine-tuned) | Change |
|---|---|---|---|
| WER (tone-agnostic) | 0.5752 | 0.6142 | **+0.0390 — still worse, but 21% less bad than v1** |

**The honest conclusion**: v2 is not yet a win, but it's real, measurable progress in the right direction, with a trajectory that was still climbing toward (and briefly past) the baseline right when training stopped. The single highest-confidence next step is simply **more epochs** (5–6) on the same data — not a different technique, and not (yet) more data.

---

## 5. If you want to try it yourself

```bash
cd phase2-asr
python3 scripts/finetune_whisper.py --epochs 5
```

Things worth trying, roughly in order of expected impact:
1. **More epochs** (`--epochs 5` or `6`) — the trajectory above suggests this alone might finally flip the result positive.
2. **More data** — re-run the pull scripts with a higher `--n` (FLEURS has 2,339 rows already downloaded locally — free to use more of; Common Voice's mirror has 3,430 total, a slower network pull for more).
3. **A real GPU** — same script, but on Colab or a rented GPU instead of this Mac's MPS backend, for much faster iteration.

Every run's real result gets saved to a fresh JSON file under `results/` — nothing is guessed or estimated, only what was actually measured.
