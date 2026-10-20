"""
The REAL, live, full-stack integration test — everything Phase 7 exists to
prove, chained together: real audio in -> real ASR (Phase 3) -> real domain
routing (this folder) -> real finance DB write OR real chat LLM (Phase 3/5)
-> real TTS (Phase 3) -> real audio out.

Uses a fixed, hand-written finance sentence (not pulled from the Yorùbá
speech dataset used elsewhere in this project, which is general-topic, not
finance-domain) synthesized to audio first via TTS, then fed back through
ASR — a real "round trip" that exercises TTS quality as an ASR input too,
which is a more demanding test than reading pre-recorded audio.

Usage:
    ASR_BACKEND=whisper_hf TTS_BACKEND=mms_tts python3 scripts/live_integration_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import orchestrator  # noqa: E402
from app.router import route  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

FINANCE_TEXT = "Mo na naira ẹgbẹrun mẹwa lori ounje loni"  # "I spent ten thousand naira on food today"
CHAT_TEXT = "Bawo ni? Se daadaa ni?"  # "Hello, are you well?"


def run_case(label: str, text: str, session_id: str, lang: str) -> dict:
    print(f"\n=== {label}: {text!r} ===")
    t0 = time.time()

    # Real round trip: synthesize the input text to audio, then transcribe
    # it back — exercises TTS quality as a real ASR input, not just a
    # pre-recorded file.
    in_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    orchestrator.phase3_tts.synthesize(text, in_audio)
    tts_in_sec = time.time() - t0

    t0 = time.time()
    asr_text = orchestrator.phase3_asr.transcribe(in_audio)
    asr_sec = time.time() - t0
    print(f"  Synthesized then re-transcribed: {asr_text!r}")

    predicted_domain = route(asr_text, lang)
    print(f"  Router (on ASR output): {predicted_domain}")

    t0 = time.time()
    result = orchestrator.handle_turn(asr_text, session_id, lang)
    handle_sec = time.time() - t0
    print(f"  Domain handled: {result['domain']}, reply: {result['reply_text']!r}")
    if result.get("detail", {}).get("error"):
        print(f"  (non-fatal handling error, fallback reply used: {result['detail']['error']})")

    t0 = time.time()
    out_audio = str(SAMPLES_DIR / f"live_integration_{label}_reply.wav")
    orchestrator.phase3_tts.synthesize(result["reply_text"], out_audio)
    tts_out_sec = time.time() - t0

    os.unlink(in_audio)

    return {
        "label": label,
        "input_text": text,
        "asr_roundtrip_text": asr_text,
        "router_domain": predicted_domain,
        "handled_domain": result["domain"],
        "reply_text": result["reply_text"],
        "reply_audio_path": out_audio,
        "handling_detail": result.get("detail", {}),
        "timings_sec": {
            "tts_input": round(tts_in_sec, 2), "asr": round(asr_sec, 2),
            "handle_turn": round(handle_sec, 2), "tts_output": round(tts_out_sec, 2),
        },
    }


def main():
    os.environ.setdefault("ASR_BACKEND", "whisper_hf")
    os.environ.setdefault("TTS_BACKEND", "mms_tts")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    session_id = "live-integration-test"
    results = [
        run_case("finance", FINANCE_TEXT, session_id, "yoruba"),
        run_case("chat", CHAT_TEXT, session_id, "yoruba"),
    ]

    log_path = SAMPLES_DIR / "live_integration_test_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nFull log: {log_path}")
    for r in results:
        router_ok = "OK" if r["router_domain"] == ("finance" if r["label"] == "finance" else "chat") else "MISROUTED"
        print(f"  [{router_ok}] {r['label']}: router said {r['router_domain']}, handler used {r['handled_domain']}")


if __name__ == "__main__":
    main()
