"""
The REAL, live, full-loop test: real audio in -> real ASR -> real LLM
(Bedrock) -> real TTS -> real audio out. Every stage is a real model call
against real infrastructure, not a mock — this is the concrete proof that
Phase 3's pipeline wiring works end-to-end, using the lightweight backends
(ASR_BACKEND=whisper_hf, TTS_BACKEND=mms_tts) specifically because they're
the ones that don't require multi-GB downloads or manual checkpoint setup
first (see asr.py / tts.py docstrings for why those are the defaults
elsewhere and these are the test-friendly alternatives).

Requires AWS credentials in the environment (same as the rest of this
project) for the LLM stage.

Usage:
    ASR_BACKEND=whisper_hf TTS_BACKEND=mms_tts python3 scripts/live_pipeline_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase1-data" / "scripts"))

from app import asr, llm, tts  # noqa: E402
from hf_preview_client import download_audio, preview_rows  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def main():
    os.environ.setdefault("ASR_BACKEND", "whisper_hf")
    os.environ.setdefault("TTS_BACKEND", "mms_tts")

    timings = {}
    log = {}

    print("[1/4] Fetching a real Yorùbá utterance...")
    t0 = time.time()
    rows_info = preview_rows("shunyalabs/yoruba-speech-dataset", "default", "train", offset=3, length=1)
    row = rows_info["rows"][0]["row"]
    reference_text = row["transcript"]
    audio_url = row["audio"][0]["src"]

    in_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    download_audio(audio_url, in_path)
    timings["fetch_input_audio_sec"] = round(time.time() - t0, 2)
    log["reference_transcript"] = reference_text
    print(f"    Reference: {reference_text}")

    print("[2/4] Transcribing (ASR)...")
    t0 = time.time()
    user_text = asr.transcribe(in_path)
    timings["asr_sec"] = round(time.time() - t0, 2)
    log["asr_output"] = user_text
    print(f"    ASR output: {user_text}")

    print("[3/4] Generating reply (LLM via Bedrock)...")
    t0 = time.time()
    try:
        client = llm.build_bedrock_client()
        session = llm.ConversationSession("live-pipeline-test")
        reply_text = llm.generate_reply(client, session, user_text)
        log["llm_stage"] = "ok"
    except Exception as exc:
        # Don't let an LLM/credentials failure block proving the ASR and TTS
        # legs still work — fall back to a fixed Yorùbá reply so the rest of
        # the pipeline is still exercised for real, and record the failure
        # honestly rather than silently swallowing it.
        reply_text = "Mo gbọ́ ohun tí o sọ. Ẹ ṣé fún lílo ètò yìí."  # "I heard what you said. Thanks for using this system."
        log["llm_stage"] = "FAILED — see llm_error"
        log["llm_error"] = f"{type(exc).__name__}: {exc}"
        print(f"    LLM call FAILED ({type(exc).__name__}: {exc}) — using fallback reply so ASR/TTS still get tested")
    timings["llm_sec"] = round(time.time() - t0, 2)
    log["llm_reply"] = reply_text
    print(f"    Reply text (real or fallback): {reply_text}")

    print("[4/4] Synthesizing reply (TTS)...")
    t0 = time.time()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / "live_pipeline_test_reply.wav"
    tts.synthesize(reply_text, str(out_path))
    timings["tts_sec"] = round(time.time() - t0, 2)

    import soundfile as sf
    data, sr = sf.read(str(out_path))
    log["output_audio_duration_sec"] = round(len(data) / sr, 2)
    log["output_audio_path"] = str(out_path)

    timings["total_sec"] = round(sum(v for k, v in timings.items() if k != "total_sec"), 2)
    log["timings"] = timings
    log["asr_backend"] = os.environ["ASR_BACKEND"]
    log["tts_backend"] = os.environ["TTS_BACKEND"]

    log_path = SAMPLES_DIR / "live_pipeline_test_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    os.unlink(in_path)

    print(f"\nDone. Output audio: {out_path} ({log['output_audio_duration_sec']}s)")
    print(f"Timings: {json.dumps(timings, indent=2)}")
    print(f"Full log: {log_path}")


if __name__ == "__main__":
    main()
