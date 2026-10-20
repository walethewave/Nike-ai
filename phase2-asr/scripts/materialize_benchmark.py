"""
Materialize a benchmark set: fetches audio directly from the HF datasets-server
API and downloads to disk, producing a manifest with audio_path that benchmark.py
can consume directly.

This is the Phase 1 → Phase 2 handoff step mentioned in PLAN.md. Run once to
create a fixed, local benchmark set; re-run benchmark.py against it as many
times as needed without re-downloading.

Usage:
    # From shunyalabs dataset (default):
    python3 scripts/materialize_benchmark.py --output-dir data/benchmark_set --limit 50

    # From a specific dataset/config/split:
    python3 scripts/materialize_benchmark.py \
        --dataset shunyalabs/yoruba-speech-dataset \
        --config default --split train \
        --output-dir data/benchmark_set --limit 100

Output:
    data/benchmark_set/
        manifest.csv      # audio_path,text columns for benchmark.py
        audio/
            0.wav
            1.wav
            ...
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reuse Phase 1's preview client for fresh API access
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase1-data" / "scripts"))
from hf_preview_client import download_audio, preview_rows  # noqa: E402


def materialize(
    output_dir: str,
    dataset: str = "shunyalabs/yoruba-speech-dataset",
    config: str = "default",
    split: str = "train",
    limit: int = 50,
    offset: int = 0,
) -> Path:
    out_path = Path(output_dir)
    audio_dir = out_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {limit} rows from {dataset} ({config}/{split}) offset={offset}...")
    result = preview_rows(dataset, config, split, offset=offset, length=limit)
    if "error" in result:
        raise RuntimeError(f"API error: {result['error']}")

    rows = result.get("rows", [])
    print(f"Got {len(rows)} rows from API")

    manifest_rows = []
    for i, entry in enumerate(rows):
        row = entry["row"]
        # shunyalabs uses "transcript", some datasets use "text"
        text = row.get("transcript") or row.get("text")
        audio_info = row.get("audio", [])
        if not audio_info or not text:
            print(f"  [{i}] SKIP — missing audio or text")
            continue

        audio_url = audio_info[0]["src"]
        audio_path = audio_dir / f"{offset + i}.wav"
        print(f"  [{i+1}/{len(rows)}] Downloading...")
        try:
            download_audio(audio_url, str(audio_path))
        except Exception as exc:
            print(f"  [{i}] FAILED: {exc}")
            continue

        manifest_rows.append({"audio_path": str(audio_path), "text": text})

    manifest_path = out_path / "manifest.csv"
    # Append if exists, else create with header
    file_exists = manifest_path.exists()
    with open(manifest_path, "a" if file_exists else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nMaterialized {len(manifest_rows)} utterances to {out_path}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Materialize benchmark audio from HF datasets-server")
    parser.add_argument("--dataset", default="shunyalabs/yoruba-speech-dataset")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True, help="Directory to write audio files + manifest")
    parser.add_argument("--limit", type=int, default=50, help="Number of utterances to download")
    parser.add_argument("--offset", type=int, default=0, help="Starting row index")
    args = parser.parse_args()

    materialize(args.output_dir, args.dataset, args.config, args.split, args.limit, args.offset)


if __name__ == "__main__":
    main()
