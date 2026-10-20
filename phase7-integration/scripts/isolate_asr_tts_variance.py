"""
Isolates WHICH layer — TTS or ASR — is responsible for the run-to-run
variance observed in live_integration_test.py (2026-08-15: 5 real runs of
the identical finance sentence produced 5 different ASR transcripts).

Two separate tests, each holding one layer fixed:

  A. TTS determinism: synthesize the SAME text N times, compare the
     resulting audio. If TTS output differs run to run, that's a real
     source of variance upstream of ASR entirely.

  B. ASR determinism: synthesize ONCE (one fixed audio file), then run
     THAT SAME audio through ASR N times. If ASR output differs on
     identical audio input, ASR's decoding itself is non-deterministic
     (unexpected for standard greedy decoding — worth confirming, not
     assuming).

Usage:
    ASR_BACKEND=whisper_hf TTS_BACKEND=mms_tts python3 scripts/isolate_asr_tts_variance.py --n 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import orchestrator  # noqa: E402

TEXT = "Mo na naira ẹgbẹrun mẹwa lori ounje loni"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _audio_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def test_tts_determinism(n: int) -> dict:
    print(f"\n=== Test A: TTS determinism — synthesizing {TEXT!r} {n} times ===")
    hashes = []
    for i in range(n):
        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        orchestrator.phase3_tts.synthesize(TEXT, out_path)
        h = _audio_hash(out_path)
        hashes.append(h)
        print(f"  run {i+1}: audio hash {h}")
        os.unlink(out_path)
    identical = len(set(hashes)) == 1
    print(f"  -> {'IDENTICAL' if identical else 'DIFFERENT'} audio across {n} runs "
          f"({len(set(hashes))} distinct hash(es))")
    return {"identical": identical, "n_distinct": len(set(hashes)), "hashes": hashes}


def test_asr_determinism(n: int) -> dict:
    print(f"\n=== Test B: ASR determinism — one fixed audio file, transcribed {n} times ===")
    fixed_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    orchestrator.phase3_tts.synthesize(TEXT, fixed_audio)
    print(f"  Fixed audio: {fixed_audio} (hash {_audio_hash(fixed_audio)})")

    transcripts = []
    for i in range(n):
        t0 = time.time()
        text = orchestrator.phase3_asr.transcribe(fixed_audio)
        elapsed = round(time.time() - t0, 2)
        transcripts.append(text)
        print(f"  run {i+1} ({elapsed}s): {text!r}")

    os.unlink(fixed_audio)
    identical = len(set(transcripts)) == 1
    print(f"  -> {'IDENTICAL' if identical else 'DIFFERENT'} transcripts across {n} runs "
          f"({len(set(transcripts))} distinct result(s))")
    return {"identical": identical, "n_distinct": len(set(transcripts)), "transcripts": transcripts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("ASR_BACKEND", "whisper_hf")
    os.environ.setdefault("TTS_BACKEND", "mms_tts")

    tts_result = test_tts_determinism(args.n)
    asr_result = test_asr_determinism(args.n)

    print("\n=== Conclusion ===")
    if not tts_result["identical"] and asr_result["identical"]:
        print("TTS is the source of variance — ASR is deterministic on fixed audio.")
    elif tts_result["identical"] and not asr_result["identical"]:
        print("ASR is the source of variance — TTS produces identical audio each time.")
    elif not tts_result["identical"] and not asr_result["identical"]:
        print("BOTH layers contribute variance.")
    else:
        print("NEITHER layer shows variance in isolation — the variance seen in "
              "live_integration_test.py may come from something else (e.g. real "
              "audio-file temp-path handling) or wasn't reproduced this run.")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / "isolate_asr_tts_variance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"text": TEXT, "n": args.n, "tts": tts_result, "asr": asr_result}, f,
                   indent=2, ensure_ascii=False)
    print(f"\nFull log: {out_path}")


if __name__ == "__main__":
    main()
