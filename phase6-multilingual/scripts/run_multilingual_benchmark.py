"""
Runs Phase 2's exact benchmark harness across every language in
lang_config.py. This file is deliberately thin — it imports
phase2-asr/scripts/{benchmark,normalize}.py rather than reimplementing
anything, which is the concrete proof of ROADMAP.md's claim that Phase 6
is "mostly re-running Phase 2/3 with different language codes, not new
infrastructure."

Usage:
    # Structural dry run — no models, no network, proves the reuse works:
    python3 scripts/run_multilingual_benchmark.py --dry-run

    # Real run (once Phase 1/2 data exists per-language):
    python3 scripts/run_multilingual_benchmark.py \\
        --manifests-dir /path/to/per_language_manifests \\
        --backend omnilingual_ctc_300m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse Phase 2's harness directly — this is the point of this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase2-asr" / "scripts"))

from lang_config import LANGUAGES  # noqa: E402
from benchmark import (  # noqa: E402
    ASRBackend, BACKEND_REGISTRY, ManifestRow, read_manifest, run_benchmark,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _dry_run():
    """Proves multi-language reuse without any model/network access: runs
    the SAME fake echo backend from Phase 2's dry-run, once per configured
    language, and confirms each gets scored independently and correctly."""
    print("Dry run: fake backend, synthetic per-language manifests, no network/model access.\n")

    class EchoBackend(ASRBackend):
        name = "echo (dry-run fake)"

        def transcribe(self, audio_paths: list, lang: str) -> list:
            return [f"REFERENCE_FOR::{lang}::{p}" for p in audio_paths]

    all_results = {}
    for key, cfg in LANGUAGES.items():
        rows = [
            ManifestRow(audio_path="a.wav", reference_text=f"REFERENCE_FOR::{cfg.omnilingual_code}::a.wav"),
            ManifestRow(audio_path="b.wav", reference_text=f"REFERENCE_FOR::{cfg.omnilingual_code}::b.wav"),
        ]
        result = run_benchmark(EchoBackend(), rows, lang=cfg.omnilingual_code)
        all_results[key] = result
        print(f"{cfg.display_name} ({cfg.omnilingual_code}): "
              f"tone_sensitive WER={result['scores']['tone_sensitive']['wer']}")
        assert result["scores"]["tone_sensitive"]["wer"] == 0.0, \
            f"Dry run should score perfect WER for {key}"

    assert set(all_results.keys()) == set(LANGUAGES.keys()), "Every configured language must produce a result"
    print(f"\nDry run OK — {len(all_results)} languages benchmarked via the reused Phase 2 harness, "
          f"zero language-specific code written.")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 multilingual benchmark wrapper (reuses Phase 2)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifests-dir", help="Directory containing <lang_key>.csv manifests (audio_path,text)")
    parser.add_argument("--backend", choices=list(BACKEND_REGISTRY.keys()))
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
        return

    if not (args.manifests_dir and args.backend):
        parser.error("--manifests-dir and --backend are required unless --dry-run is given")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    backend = BACKEND_REGISTRY[args.backend]()

    for key, cfg in LANGUAGES.items():
        manifest_path = Path(args.manifests_dir) / f"{key}.csv"
        if not manifest_path.exists():
            print(f"SKIPPING {cfg.display_name} — no manifest at {manifest_path}")
            continue

        rows = read_manifest(str(manifest_path))
        result = run_benchmark(backend, rows, lang=cfg.omnilingual_code)

        out_path = RESULTS_DIR / f"{args.backend}__{key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"{cfg.display_name}: WER={result['scores']['tone_sensitive']['wer']} -> {out_path}")


if __name__ == "__main__":
    main()
