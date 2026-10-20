"""
Fine-tunes `steja/whisper-small-yoruba` (Phase 2's proven zero-shot winner,
46.1% WER) on the real permissive-only training data pulled for Option (c)
of Phase 1's licensing decision (`pull_permissive_training_data.py` +
`pull_fleurs_permissive.py` — 600 real utterances, ~1.8h, Common Voice 17
(yo) + FLEURS yo_ng).

This is a SHORT, SESSION-SCALE VALIDATION RUN by design, not a claim of a
finished, production fine-tune — 1.8h is a thin slice next to the ~130h
full permissive corpus or the 600h NaijaVoices corpus. The point is to prove
the training pipeline works end-to-end (data loading, feature extraction,
training loop, generation-based eval) on real data, using a bounded subset
and step count so it finishes in a normal working session. Scaling up later
is a matter of raising --max-train/--max-steps and re-pulling more data with
the two pull scripts above, not rewriting this script.

Training recipe follows HuggingFace's standard Whisper fine-tuning pattern
(precomputed log-mel features + tokenized labels, right-shift handled by the
data collator) — the well-tested, low-risk approach, not a custom loop.

Reports WER/CER (both tone modes, via normalize.py) for BOTH the zero-shot
baseline AND the fine-tuned model, on the IDENTICAL held-out split — a
before/after on the same data, not a comparison against Phase 2's earlier
benchmark numbers (which used a different eval set, shunyalabs) that
wouldn't be a fair apples-to-apples comparison. Also reports a per-epoch
WER trajectory on a small held-out subset (via EpochTrajectoryCallback),
not just a single before/after snapshot.

v1 of this script (max_steps=60, ~0.8 epochs, lr=1e-5, warmup_steps=5) made
held-out WER WORSE on two real runs — see results/finetune_validation_run.json
and PLAN.md. v2 (this version) fixes the likely causes: real multi-epoch
training instead of under one epoch, a lower learning rate, and a
warmup_ratio instead of a fixed tiny warmup_steps count. Saves to
results/finetune_validation_run_v2.json so both real results stay comparable.

Usage:
    python3 scripts/finetune_whisper.py --epochs 3 --max-train 150 --max-eval 30
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from normalize import evaluate_both_modes  # noqa: E402

BASE_MODEL = "steja/whisper-small-yoruba"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "permissive_train"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TARGET_SR = 16000
SEED = 42


def load_manifest_rows() -> list[dict]:
    """Combines both permissive-only manifests into one list of real
    {audio_path (absolute), text} rows."""
    rows = []
    for name in ("common_voice_yo", "fleurs_yo_ng"):
        manifest_path = DATA_DIR / name / "manifest.csv"
        base = DATA_DIR
        with open(manifest_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "audio_path": str(base / r["audio_path"]),
                    "text": r["text"].strip(),
                    "source": r["source"],
                })
    return rows


def load_audio_16k(path: str) -> np.ndarray:
    """Reads real audio (mp3 or wav) and resamples to 16kHz mono — the
    format Whisper's feature extractor expects. Uses soundfile (already
    proven to decode both formats in this repo, see pull scripts) +
    scipy's polyphase resampler, rather than adding librosa/torchcodec as
    new dependencies for something this small."""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        from math import gcd
        g = gcd(sr, TARGET_SR)
        data = scipy.signal.resample_poly(data, TARGET_SR // g, sr // g)
    return data.astype(np.float32)


@dataclass
class Example:
    input_features: np.ndarray
    labels: list[int]
    text: str
    audio_path: str


def prepare_examples(rows: list[dict], processor) -> list[Example]:
    examples = []
    for i, row in enumerate(rows):
        audio = load_audio_16k(row["audio_path"])
        input_features = processor.feature_extractor(
            audio, sampling_rate=TARGET_SR
        ).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        examples.append(Example(input_features, labels, row["text"], row["audio_path"]))
        if (i + 1) % 50 == 0:
            print(f"  Prepared {i + 1}/{len(rows)} examples...")
    return examples


class WhisperTrainDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"input_features": ex.input_features, "labels": ex.labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """The standard HF Whisper fine-tuning collator: pads input_features
    via the feature extractor, pads labels via the tokenizer, masks padded
    label positions with -100 (ignored by the loss), and drops a leading
    BOS token if the tokenizer's padding already introduced one (the model
    adds it back itself during training)."""
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def run_generation_eval(model, processor, examples: list[Example], device, label: str) -> dict:
    """Real generation (not teacher-forced loss) on held-out examples,
    scored with Phase 2's real tone-aware/tone-agnostic scorer."""
    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for ex in examples:
            input_features = torch.tensor(ex.input_features).unsqueeze(0).to(device)
            generated_ids = model.generate(input_features, max_new_tokens=128)
            hyp = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            references.append(ex.text)
            hypotheses.append(hyp)

    scores = evaluate_both_modes(references, hypotheses)
    scores["label"] = label
    scores["sample_predictions"] = [
        {"reference": r, "hypothesis": h} for r, h in zip(references[:5], hypotheses[:5])
    ]
    print(f"[{label}] tone_agnostic WER={scores['tone_agnostic']['wer']:.4f} "
          f"CER={scores['tone_agnostic']['cer']:.4f} (n={len(references)})")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=150, help="max real training examples to use")
    parser.add_argument("--max-eval", type=int, default=30, help="max real held-out examples to use")
    parser.add_argument("--epochs", type=float, default=3.0,
                         help="real training epochs over --max-train examples (primary control — "
                              "the first validation run only reached 0.8 epochs, which is too little "
                              "for held-out WER to net out ahead of early-training instability)")
    parser.add_argument("--max-steps", type=int, default=-1,
                         help="override: if >0, stop at this many optimizer steps instead of --epochs "
                              "(kept for quick mechanism checks, not used by default anymore)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6,
                         help="lowered from the first run's 1e-5 — paired with a shorter warmup, "
                              "1e-5 likely contributed to the early held-out WER regression")
    parser.add_argument("--eval-trajectory-n", type=int, default=10,
                         help="how many held-out examples to score after each epoch, to see the WER "
                              "trajectory (dip-then-recover vs. steadily worse) instead of only a "
                              "single before/after snapshot")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                               WhisperForConditionalGeneration, WhisperProcessor)

    # v2 and v3 used identical hyperparameters/data/split but produced
    # different per-epoch WER trajectories — traced to this: only the
    # train/eval SPLIT was seeded (below), not PyTorch's own RNG, which
    # controls per-epoch batch shuffling and dropout. Seeding it here
    # makes re-runs of "the same" config actually reproducible.
    torch.manual_seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading real manifest rows...")
    rows = load_manifest_rows()
    print(f"  {len(rows)} real rows available across both permissive sources")

    random.seed(SEED)
    random.shuffle(rows)
    eval_rows = rows[:args.max_eval]
    train_rows = rows[args.max_eval:args.max_eval + args.max_train]
    print(f"  Using {len(train_rows)} for training, {len(eval_rows)} held out for eval "
          f"(disjoint, seed={SEED})")

    print(f"Loading base model + processor ({BASE_MODEL})...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    print("Preparing real held-out eval examples (audio decode + feature extraction)...")
    eval_examples = prepare_examples(eval_rows, processor)

    print("\n=== Zero-shot baseline, on the held-out set (before any fine-tuning) ===")
    model.to(device)
    baseline_scores = run_generation_eval(model, processor, eval_examples, device, "zero_shot_baseline")

    print("\nPreparing real training examples (audio decode + feature extraction)...")
    train_examples = prepare_examples(train_rows, processor)
    train_dataset = WhisperTrainDataset(train_examples)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trajectory_examples = eval_examples[:args.eval_trajectory_n]
    trajectory: list[dict] = [{
        "epoch": 0.0, "tone_agnostic_wer": baseline_scores["tone_agnostic"]["wer"],
        "n": len(trajectory_examples), "label": "zero_shot_baseline",
    }]

    from transformers import TrainerCallback

    class EpochTrajectoryCallback(TrainerCallback):
        """Scores a small held-out subset after every epoch so the run
        shows a WER trajectory (dip-then-recover vs. steadily worse)
        instead of only a single before/after snapshot — the first
        validation run only had that snapshot and couldn't distinguish
        the two explanations."""
        def on_epoch_end(self, training_args_, state, control, model=None, **kwargs):
            scores = run_generation_eval(
                model, processor, trajectory_examples, device,
                f"epoch_{state.epoch:.1f}",
            )
            trajectory.append({
                "epoch": round(state.epoch, 2),
                "tone_agnostic_wer": scores["tone_agnostic"]["wer"],
                "n": len(trajectory_examples), "label": "training",
            })
            model.train()

    output_dir = args.output_dir or str(RESULTS_DIR / "finetune_checkpoint")
    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        # gradient_checkpointing=True was tried here as a memory mitigation
        # after --epochs 5 crashed with an MPS out-of-memory error at 71%.
        # It didn't work: even with model.config.use_cache=False set first,
        # it hit an immediate, different crash ("backward through the graph
        # a second time") on the very first training step — a deeper
        # incompatibility between PyTorch's default reentrant checkpointing
        # mode and Whisper's architecture, not resolved here. See PLAN.md's
        # "v3" section — ~3 epochs is the practical ceiling on this M5/MPS
        # setup for this data/batch size without a real GPU.
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
        run_desc = f"{args.max_steps} steps (override)"
    else:
        training_kwargs["num_train_epochs"] = args.epochs
        run_desc = f"{args.epochs} real epochs"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[EpochTrajectoryCallback()],
    )

    print(f"\n=== Training: {run_desc}, lr={args.lr}, warmup_ratio=0.1, batch size {args.batch_size}, "
          f"{len(train_rows)} real examples ===")
    t0 = time.time()
    trainer.train()
    train_sec = round(time.time() - t0, 1)
    print(f"Training finished in {train_sec}s")
    print("\nWER trajectory (tone-agnostic, on a", args.eval_trajectory_n, "example subset):")
    for point in trajectory:
        print(f"  epoch {point['epoch']:>4.1f}: WER={point['tone_agnostic_wer']:.4f}")

    print("\n=== Fine-tuned model, on the SAME held-out set ===")
    finetuned_scores = run_generation_eval(model, processor, eval_examples, device, "finetuned")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "base_model": BASE_MODEL,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "epochs": args.epochs if args.max_steps <= 0 else None,
        "max_steps": args.max_steps if args.max_steps > 0 else None,
        "learning_rate": args.lr,
        "warmup_ratio": 0.1,
        "train_sec": train_sec,
        "device": device,
        "wer_trajectory": trajectory,
        "zero_shot_baseline": baseline_scores,
        "finetuned": finetuned_scores,
        "wer_delta_tone_agnostic": round(
            finetuned_scores["tone_agnostic"]["wer"] - baseline_scores["tone_agnostic"]["wer"], 4
        ),
    }
    out_path = RESULTS_DIR / "finetune_validation_run_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nBaseline  tone_agnostic WER: {baseline_scores['tone_agnostic']['wer']:.4f}")
    print(f"Finetuned tone_agnostic WER: {finetuned_scores['tone_agnostic']['wer']:.4f}")
    print(f"Delta: {result['wer_delta_tone_agnostic']:+.4f} (negative = improvement)")
    print(f"\nFull log: {out_path}")


if __name__ == "__main__":
    main()
