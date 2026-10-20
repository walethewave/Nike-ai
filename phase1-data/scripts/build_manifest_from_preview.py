"""
Builds a Phase 1 manifest CSV (see PLAN.md's schema) from a dataset
previewable via hf_preview_client.py — the lightweight, verified-working
path, as opposed to inspect_datasets.py's full `datasets`-library streaming
(which works for NaijaVoices once gated access is set up, but stalled on
FLEURS in this environment — see PLAN.md).

This does NOT download every audio file (that's a lot of individual HTTP
requests for a real corpus) — it writes the manifest with the CDN preview
URL in an `audio_url` column, plus real duration computed by actually
downloading and probing each one (so duration numbers are real, not
estimated). Downloading the WAVs referenced by the manifest for actual
training/benchmarking is a separate, later step — see Phase 2's PLAN.md.

Usage:
    python3 scripts/build_manifest_from_preview.py \\
        --dataset shunyalabs/yoruba-speech-dataset --split train --n 10 \\
        --text-field transcript --out manifests/shunyalabs_train_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_preview_client import download_audio, preview_rows  # noqa: E402

MANIFEST_FIELDS = [
    "source", "source_config", "row_index", "text",
    "duration_sec", "speaker_id", "gender", "license", "audio_url",
]


def probe_duration(url: str) -> float:
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        download_audio(url, tmp.name)
        data, sr = sf.read(tmp.name)
        return round(len(data) / sr, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--text-field", required=True, help="Column name holding the transcript")
    parser.add_argument("--speaker-field", default=None)
    parser.add_argument("--gender-field", default=None)
    parser.add_argument("--license", default="UNKNOWN — verify against the dataset's card before use")
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe-duration", action="store_true",
                         help="Actually download each audio file to compute real duration "
                              "(slower — one HTTP request per row)")
    args = parser.parse_args()

    result = preview_rows(args.dataset, args.config, args.split, length=args.n)
    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        raise SystemExit(1)

    rows_out = []
    for i, entry in enumerate(result["rows"]):
        row = entry["row"]
        text = unicodedata.normalize("NFC", row.get(args.text_field, ""))

        audio_url = ""
        for v in row.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "src" in v[0]:
                audio_url = v[0]["src"]
                break

        duration = ""
        if args.probe_duration and audio_url:
            try:
                duration = probe_duration(audio_url)
                print(f"  row {i}: {duration}s")
            except Exception as exc:
                print(f"  row {i}: duration probe failed ({exc})", file=sys.stderr)

        rows_out.append({
            "source": args.dataset,
            "source_config": f"{args.config}/{args.split}",
            "row_index": i,
            "text": text,
            "duration_sec": duration,
            "speaker_id": row.get(args.speaker_field, "") if args.speaker_field else "",
            "gender": row.get(args.gender_field, "") if args.gender_field else "",
            "license": args.license,
            "audio_url": audio_url,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
