"""
Lightweight Hugging Face dataset preview client using the `datasets-server`
REST API (datasets-server.huggingface.co) instead of the full `datasets`
Python library's streaming mode.

Why this exists: `google/fleurs` (config `yo_ng`) was the original plan for
an open, ungated Yorùbá eval set, but streaming it via `datasets.load_dataset`
stalled repeatedly in this project's dev environment. Investigating with
this same REST API revealed why — its parquet file has ~965MB row groups,
over the datasets-server's 300MB scan limit, so even fetching 2 rows means
reading nearly a gigabyte. That's a property of the file, not a network
fluke (see PLAN.md). This client hits the same lightweight REST API
directly, which works for datasets with reasonably-sized row groups and
fails fast (clear error, not a silent multi-minute hang) on ones that don't.

Verified live, 2026-08-13, against `shunyalabs/yoruba-speech-dataset`
(3,548 real utterances, train/validation/test splits, ungated) — downloaded
and validated a real 18.12s 16kHz WAV file from it. See
results/hf_preview_smoke_test.json for the saved proof.

Usage:
    python3 scripts/hf_preview_client.py --dataset shunyalabs/yoruba-speech-dataset --n 5
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"


def _get_json(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "naija-voice-ai-phase1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def check_splits(dataset: str) -> dict:
    encoded = urllib.parse.quote(dataset, safe="")
    return _get_json(f"{DATASETS_SERVER_BASE}/splits?dataset={encoded}")


def preview_rows(dataset: str, config: str = "default", split: str = "train",
                  offset: int = 0, length: int = 5) -> dict:
    encoded = urllib.parse.quote(dataset, safe="")
    url = (f"{DATASETS_SERVER_BASE}/rows?dataset={encoded}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    return _get_json(url)


def download_audio(url: str, out_path: str, retries: int = 5, backoff: float = 2.0) -> str:
    """The cached-assets CDN occasionally returns a transient 502 on first
    hit (observed live) — retry with backoff rather than failing immediately."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, out_path)
            return out_path
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_exc


def main():
    parser = argparse.ArgumentParser(description="Lightweight HF dataset preview client")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--download-first-audio", metavar="OUT_PATH",
                         help="If set, download the first row's audio file to this path")
    args = parser.parse_args()

    print(f"Checking splits for {args.dataset}...")
    splits_info = check_splits(args.dataset)
    if "error" in splits_info:
        print(f"ERROR: {splits_info['error']}")
        raise SystemExit(1)
    for s in splits_info["splits"]:
        print(f"  config={s['config']!r} split={s['split']!r}")

    print(f"\nFetching {args.n} rows from config={args.config!r} split={args.split!r}...")
    rows_info = preview_rows(args.dataset, args.config, args.split, length=args.n)
    if "error" in rows_info:
        print(f"ERROR: {rows_info['error']}")
        raise SystemExit(1)

    for i, entry in enumerate(rows_info["rows"]):
        row = entry["row"]
        print(f"\n--- row {i} ---")
        for k, v in row.items():
            s = str(v)
            print(f"  {k}: {s[:120]}")

    if args.download_first_audio:
        row0 = rows_info["rows"][0]["row"]
        audio_field = next((v for k, v in row0.items() if isinstance(v, list) and v and "src" in v[0]), None)
        if not audio_field:
            print("\nNo audio-like field found in first row (looked for a list with a 'src' key).")
            return
        url = audio_field[0]["src"]
        print(f"\nDownloading first row's audio to {args.download_first_audio}...")
        download_audio(url, args.download_first_audio)
        print("Downloaded OK.")


if __name__ == "__main__":
    main()
