"""
A REAL, live, end-to-end ASR smoke test — not a mock, not a dry-run. Pulls
one real Yorùbá audio file (via Phase 1's hf_preview_client.py), runs a
real Whisper-base Yorùbá fine-tune on it, and scores the result with the
tone-aware/tone-agnostic scorer, saving output in the exact schema
`benchmark.py` produces so it's directly usable by Phase 4's results-table
generator.

Model choice: `ndeclarke/whisper-base-yoruba-colab-CV17.0` (~292MB) rather
than one of the heavier options in BACKEND_REGISTRY (Omnilingual ASR is
1.3GB+ even at the smallest size, NCAIR1/Yoruba-ASR is ~970MB) — chosen
specifically so this smoke test is something you can actually run without
a long wait, not because it's the model Phase 2's real benchmark should
settle on. Extending BACKEND_REGISTRY in benchmark.py with this and running
the full comparison matrix is the real next step (see PLAN.md).

Usage:
    python3 scripts/live_smoke_test.py
    python3 scripts/live_smoke_test.py --n 5   # more than one utterance
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase1-data" / "scripts"))

from normalize import evaluate_both_modes  # noqa: E402
from hf_preview_client import download_audio, preview_rows  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODEL_ID = "ndeclarke/whisper-base-yoruba-colab-CV17.0"
DATASET = "shunyalabs/yoruba-speech-dataset"


def run(n: int) -> dict:
    print(f"Loading {MODEL_ID} (this downloads ~292MB on first run)...")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=MODEL_ID)

    print(f"Fetching {n} real utterance(s) from {DATASET}...")
    rows_info = preview_rows(DATASET, "default", "train", length=n)
    if "error" in rows_info:
        raise RuntimeError(f"Could not fetch preview rows: {rows_info['error']}")

    references, hypotheses, per_utterance = [], [], []
    for i, entry in enumerate(rows_info["rows"]):
        row = entry["row"]
        reference = row["transcript"]
        audio_url = row["audio"][0]["src"]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            print(f"  [{i}] downloading audio...")
            download_audio(audio_url, tmp.name)
            print(f"  [{i}] transcribing...")
            hypothesis = asr(tmp.name)["text"].strip()

        references.append(reference)
        hypotheses.append(hypothesis)
        per_utterance.append({"reference": reference, "hypothesis": hypothesis})
        print(f"  [{i}] REF: {reference[:80]}")
        print(f"  [{i}] HYP: {hypothesis[:80]}")

    scores = evaluate_both_modes(references, hypotheses)
    return {
        "backend": f"whisper/{MODEL_ID}",
        "lang": "yor_Latn",
        "n_utterances": len(references),
        "scores": scores,
        "sample_predictions": per_utterance,
        "note": ("LIVE smoke test against a real downloaded model and real downloaded "
                 "audio from a dataset with an unstated license (see phase1-data/PLAN.md) "
                 "— treat these numbers as a pipeline proof, not a benchmark result to "
                 "cite, until run against a properly-licensed, properly-sized test set."),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    result = run(args.n)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "live_smoke_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nn={result['n_utterances']}")
    print(f"tone_sensitive WER={result['scores']['tone_sensitive']['wer']}  "
          f"CER={result['scores']['tone_sensitive']['cer']}")
    print(f"tone_agnostic  WER={result['scores']['tone_agnostic']['wer']}  "
          f"CER={result['scores']['tone_agnostic']['cer']}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
