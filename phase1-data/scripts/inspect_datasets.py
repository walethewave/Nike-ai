"""
Phase 1 prototype: pull a small streamed sample from each candidate Yorùbá
speech dataset, report real stats (duration/speaker/gender), and write a
manifest CSV per accessible source.

This does NOT download full datasets — it streams `--n-samples` rows per
source (default 20) to prove the pipeline works and give you real numbers
to plan Phase 2 with. Scaling up is a matter of raising --n-samples or
switching off streaming once you've decided how much data you actually need
(see PLAN.md's licensing decision first).

Sources attempted, in order of research priority (see PLAN.md):
  1. naijavoices/naijavoices-dataset (yoruba-batch-0)  — GATED, needs HF_TOKEN
     + prior registration at naijavoices.com/membership + accepting the HF gate
  2. google/fleurs (yo_ng)                              — open, CC-BY-4.0
  3. mozilla-foundation/common_voice_17_0 (yo)          — open, CC0 (but the
     HF repo still needs one-time term acceptance via the HF web UI)

A source that fails (not authenticated, not gated-accepted, network issue)
is skipped with a clear reason — the script still produces manifests for
whatever IS accessible, rather than failing the whole run.

Usage:
    python3 scripts/inspect_datasets.py --n-samples 20
    python3 scripts/inspect_datasets.py --n-samples 20 --sources fleurs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"

MANIFEST_FIELDS = [
    "source", "source_config", "row_index", "text",
    "duration_sec", "speaker_id", "gender", "license",
]


@dataclass
class SourceSpec:
    key: str
    hf_repo: str
    hf_config: str
    license: str
    text_field: str
    speaker_field: str | None
    gender_field: str | None
    needs_trust_remote_code: bool = False
    notes: str = ""


SOURCES: dict[str, SourceSpec] = {
    "naijavoices": SourceSpec(
        key="naijavoices",
        hf_repo="naijavoices/naijavoices-dataset",
        hf_config="yoruba-batch-0",
        license="CC-BY-NC-SA-4.0",
        text_field="text",
        speaker_field="speaker_id",
        gender_field="gender",
        notes="Gated — requires naijavoices.com/membership registration + HF gate acceptance + HF_TOKEN env var.",
    ),
    "fleurs": SourceSpec(
        key="fleurs",
        hf_repo="google/fleurs",
        hf_config="yo_ng",
        license="CC-BY-4.0",
        text_field="raw_transcription",
        speaker_field=None,
        gender_field="gender",
        needs_trust_remote_code=False,  # verified live 2026-07-31: current `datasets`
        # versions reject trust_remote_code outright and auto-resolve FLEURS to its
        # Parquet mirror instead — don't pass it, it's not just unnecessary, it errors.
        notes="Open. First load streams a multi-hundred-MB parquet shard — can be slow "
              "on a poor connection (observed read timeouts + auto-retry, not a hang).",
    ),
    "common_voice": SourceSpec(
        key="common_voice",
        hf_repo="mozilla-foundation/common_voice_17_0",
        hf_config="yo",
        license="CC0-1.0",
        text_field="sentence",
        speaker_field="client_id",
        gender_field="gender",
        notes="Open (CC0), but the HF repo may require accepting Mozilla's terms once via huggingface.co in a browser before streaming works.",
    ),
}


def sample_source(spec: SourceSpec, n_samples: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        spec.hf_repo,
        spec.hf_config,
        split="train",
        streaming=True,
        trust_remote_code=spec.needs_trust_remote_code,
    )

    rows = []
    for i, row in enumerate(ds):
        if i >= n_samples:
            break
        audio = row.get("audio") or {}
        sr = audio.get("sampling_rate")
        arr = audio.get("array")
        duration_sec = (len(arr) / sr) if (arr is not None and sr) else None

        text = row.get(spec.text_field, "")
        text = unicodedata.normalize("NFC", text) if text else ""

        rows.append({
            "source": spec.key,
            "source_config": spec.hf_config,
            "row_index": i,
            "text": text,
            "duration_sec": round(duration_sec, 3) if duration_sec is not None else "",
            "speaker_id": row.get(spec.speaker_field, "") if spec.speaker_field else "",
            "gender": row.get(spec.gender_field, "") if spec.gender_field else "",
            "license": spec.license,
        })
    return rows


def write_manifest(source_key: str, rows: list[dict]) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MANIFEST_DIR / f"{source_key}_sample.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def summarize(source_key: str, rows: list[dict]):
    durations = [r["duration_sec"] for r in rows if r["duration_sec"] != ""]
    genders = [r["gender"] for r in rows if r["gender"]]
    print(f"\n=== {source_key} ===")
    print(f"  rows sampled:      {len(rows)}")
    if durations:
        print(f"  mean duration:     {sum(durations)/len(durations):.2f}s")
        print(f"  min/max duration:  {min(durations):.2f}s / {max(durations):.2f}s")
    if genders:
        from collections import Counter
        print(f"  gender counts:     {dict(Counter(genders))}")
    if rows:
        print(f"  example text:      {rows[0]['text'][:80]!r}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 dataset inspection prototype")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument(
        "--sources", default="naijavoices,fleurs,common_voice",
        help="Comma-separated subset of: naijavoices, fleurs, common_voice",
    )
    args = parser.parse_args()

    requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SOURCES]
    if unknown:
        print(f"Unknown source(s): {unknown}. Valid: {list(SOURCES)}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for key in requested:
        spec = SOURCES[key]
        print(f"\nAttempting {key} ({spec.hf_repo}, config={spec.hf_config})...")
        try:
            rows = sample_source(spec, args.n_samples)
            if not rows:
                print(f"  SKIPPED — no rows returned (empty stream).")
                continue
            path = write_manifest(key, rows)
            print(f"  OK — wrote {len(rows)} rows to {path}")
            results[key] = rows
        except Exception as exc:
            print(f"  SKIPPED — {type(exc).__name__}: {str(exc)[:300]}")
            print(f"  ({spec.notes})")

    if not results:
        print("\nNo sources were accessible. Check credentials/gates per the "
              "notes above, then re-run.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, rows in results.items():
        summarize(key, rows)


if __name__ == "__main__":
    main()
