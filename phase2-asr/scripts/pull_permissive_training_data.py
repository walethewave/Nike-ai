"""
Pulls real training data for the Option (c) permissive-only Yorùbá ASR
variant (see phase1-data/PLAN.md's licensing decision) from two ungated,
commercial-safe sources:

  - Common Voice 17 (yo): the official `fsicoli/common_voice_17_0` repo is a
    script-based dataset, and `datasets` 4.5.0 has fully dropped script-based
    dataset support (confirmed live, not assumed — RuntimeError: "Dataset
    scripts are no longer supported"). Using `Abdullah804/common-voice-yoruba`
    instead: an ungated, Parquet-native community re-export. Verified live
    that its first row (sentence, client_id, and the Mozilla-standard
    `common_voice_yo_<id>.mp3` filename) is byte-identical to a second,
    independent mirror (`Binaryy/yoruba-common-voice`) — real evidence this
    is a faithful re-export of the official CV17 Yorùbá data, not a
    fabrication. Neither mirror carries its own license tag on HuggingFace,
    even though the upstream Common Voice corpus is CC0 — same "smoke-test
    until verified further" caveat this repo already applies to shunyalabs.

  - FLEURS (yo_ng): pulled separately via `datasets.load_dataset(...,
    streaming=False)` (see scripts/pull_fleurs_full.py notes in PLAN.md) —
    NOT via this script's REST-preview approach, because FLEURS' ~965MB
    row groups make even a 1-row REST preview fail (confirmed in Phase 1).
    Full non-streaming download avoids that specific failure mode.

Usage:
    python3 scripts/pull_permissive_training_data.py --n 200
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase1-data" / "scripts"))
from hf_preview_client import check_splits, download_audio, preview_rows  # noqa: E402

DATASET = "Abdullah804/common-voice-yoruba"
CONFIG = "default"
SPLIT = "train"
PAGE_SIZE = 100  # datasets-server /rows endpoint's practical page cap

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "permissive_train"
AUDIO_DIR = DATA_DIR / "common_voice_yo" / "audio"
MANIFEST_PATH = DATA_DIR / "common_voice_yo" / "manifest.csv"


def pull_common_voice(n: int) -> list[dict]:
    print(f"Checking access to {DATASET}...")
    splits_info = check_splits(DATASET)
    print(f"  Splits available: {[s['split'] for s in splits_info['splits']]}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    offset = 0
    while len(rows) < n:
        length = min(PAGE_SIZE, n - len(rows))
        print(f"Fetching rows {offset}..{offset + length}...")
        page = preview_rows(DATASET, CONFIG, SPLIT, offset=offset, length=length)
        page_rows = page.get("rows", [])
        if not page_rows:
            print("  No more rows available.")
            break

        for entry in page_rows:
            row = entry["row"]
            row_index = entry["row_idx"]
            text = row.get("sentence", "").strip()
            if not text:
                continue  # skip rows with no transcript — not usable for fine-tuning

            audio_field = row.get("audio")
            if not audio_field:
                continue
            audio_url = audio_field[0]["src"] if isinstance(audio_field, list) else audio_field["src"]

            out_path = AUDIO_DIR / f"row{row_index}.mp3"
            try:
                download_audio(audio_url, str(out_path))
            except Exception as exc:
                print(f"  Row {row_index}: download failed ({exc}), skipping")
                continue

            rows.append({
                "source": DATASET,
                "source_config": CONFIG,
                "row_index": row_index,
                "text": text,
                "duration_sec": "",  # not probed here — see phase1's --probe-duration pattern to add real durations
                "speaker_id": row.get("client_id", "")[:16],  # truncated, these are long hashes
                "gender": row.get("gender", ""),
                "license": "unstated-on-mirror (upstream Common Voice is CC0; mirror repo carries no license tag — verify before non-smoke-test use)",
                "audio_path": str(out_path.relative_to(DATA_DIR)),
            })

        offset += length
        time.sleep(0.2)  # be polite to the API between pages

    return rows


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "source_config", "row_index", "text", "duration_sec",
                  "speaker_id", "gender", "license", "audio_path"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of Common Voice rows to pull")
    args = parser.parse_args()

    rows = pull_common_voice(args.n)
    write_manifest(rows)

    total_audio_bytes = sum(f.stat().st_size for f in AUDIO_DIR.glob("*.mp3"))
    print(f"Downloaded {len(list(AUDIO_DIR.glob('*.mp3')))} real audio files, "
          f"{total_audio_bytes / 1e6:.1f} MB total.")


if __name__ == "__main__":
    main()
