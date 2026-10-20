"""
Materializes a real, bounded training sample from FLEURS (yo_ng) for the
Option (c) permissive-only Yorùbá ASR variant — the second of the two
permissive sources (see pull_permissive_training_data.py for Common Voice).

FLEURS could NOT be pulled via this repo's usual REST-preview approach
(hf_preview_client.py) — its ~965MB row groups make even a 1-row preview
fail (see phase1-data/PLAN.md). Worked around by using
`datasets.load_dataset(..., streaming=False)` instead: a full, one-time,
non-streaming download of the train split (2,339 rows, ~889s / ~15 min,
~3.3GB cached under ~/.cache/huggingface/hub). That succeeded, but hit a
SECOND real, separate issue: `datasets` 4.5.0's default Audio decoding
backend requires `torchcodec`, which isn't installed here. Worked around
by loading the audio column with `decode=False` and decoding the raw bytes
with `soundfile` directly instead — consistent with how the rest of this
repo handles audio, and avoids adding a new heavy dependency.

Also resolves a real open question flagged in phase1-data/PLAN.md: which
text field to use. `raw_transcription` (not `transcription`) is the correct
field — confirmed live by reading actual rows: `raw_transcription` preserves
capitalization and punctuation ("Dr.Lee ... Turkey ... A(H5N1)"), while
`transcription` is a lowercased/simplified variant. Diacritics are present
in both.

Usage:
    python3 scripts/pull_fleurs_permissive.py --n 300
"""
from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "permissive_train"
AUDIO_DIR = DATA_DIR / "fleurs_yo_ng" / "audio"
MANIFEST_PATH = DATA_DIR / "fleurs_yo_ng" / "manifest.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of FLEURS rows to materialize")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FLEURS yo_ng train split (from local cache if already downloaded)...")
    t0 = time.time()
    ds = load_dataset("google/fleurs", "yo_ng", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"Loaded {len(ds)} rows in {round(time.time() - t0, 2)}s")

    n = min(args.n, len(ds))
    rows = []
    for i in range(n):
        row = ds[i]
        text = row["raw_transcription"].strip()
        if not text:
            continue

        audio_bytes = row["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(audio_bytes))
        duration_sec = round(len(data) / sr, 2)

        out_path = AUDIO_DIR / f"row{i}.wav"
        sf.write(str(out_path), data, sr)

        rows.append({
            "source": "google/fleurs",
            "source_config": "yo_ng",
            "row_index": i,
            "text": text,
            "duration_sec": duration_sec,
            "speaker_id": "",  # FLEURS doesn't expose a speaker_id field
            "gender": row.get("gender", ""),
            "license": "cc-by-4.0",
            "audio_path": str(out_path.relative_to(DATA_DIR)),
        })

        if (i + 1) % 50 == 0:
            print(f"  Materialized {i + 1}/{n}...")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "source_config", "row_index", "text", "duration_sec",
                  "speaker_id", "gender", "license", "audio_path"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_sec = sum(r["duration_sec"] for r in rows)
    print(f"\nWrote {len(rows)} rows to {MANIFEST_PATH}")
    print(f"Total real audio: {total_sec / 3600:.2f} hours ({total_sec:.0f}s)")


if __name__ == "__main__":
    main()
