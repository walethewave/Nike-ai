"""
The actual "connect everything together" layer. Given real audio bytes,
runs: ASR (Phase 3's `app/asr.py`) -> domain routing (this folder's
`router.py`) -> either Phase 3's general chat LLM or Phase 5's finance
intent handling -> TTS (Phase 3's `app/tts.py`).

This module deliberately imports sibling phases' code rather than
reimplementing it — same principle as Phase 6's benchmark reuse. The
cross-folder import path is set up once in main.py / the test scripts, not
duplicated here.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

# Phase 3 and Phase 5 each have a top-level package literally named `app` —
# a plain `sys.path` + `from app import x` approach silently breaks here:
# Python caches the FIRST `app` package it resolves under the name "app" in
# sys.modules, so the second sibling's submodules become unimportable (this
# was caught empirically, not assumed — a naive first version of this file
# failed with `ImportError: cannot import name 'asr' from 'app'` pointing at
# the WRONG phase's app package). Fix: load each phase's package under a
# distinct name via importlib, so both coexist in sys.modules without
# colliding.

_PHASE3_DIR = Path(__file__).resolve().parent.parent.parent / "phase3-conversational"
_PHASE5_DIR = Path(__file__).resolve().parent.parent.parent / "phase5-finance"


def _load_module(unique_name: str, file_path: Path):
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def _load_package(unique_pkg_name: str, package_dir: Path):
    """Registers `package_dir` as a package under `unique_pkg_name`, so its
    submodules' own relative imports (`from . import x`) resolve correctly."""
    if unique_pkg_name in sys.modules:
        return sys.modules[unique_pkg_name]
    spec = importlib.util.spec_from_file_location(
        unique_pkg_name, package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_pkg_name] = module
    spec.loader.exec_module(module)
    return module


_load_package("phase3_app", _PHASE3_DIR / "app")
_load_package("phase5_app", _PHASE5_DIR / "app")

phase3_asr = _load_module("phase3_app.asr", _PHASE3_DIR / "app" / "asr.py")
phase3_llm = _load_module("phase3_app.llm", _PHASE3_DIR / "app" / "llm.py")
phase3_tts = _load_module("phase3_app.tts", _PHASE3_DIR / "app" / "tts.py")
phase5_db = _load_module("phase5_app.db", _PHASE5_DIR / "app" / "db.py")
phase5_intent = _load_module("phase5_app.intent", _PHASE5_DIR / "app" / "intent.py")
# phase5's main.py does `from . import db, intent` — needs its package
# registered first (done above) so that relative import resolves.
phase5_main = _load_module("phase5_app.main", _PHASE5_DIR / "app" / "main.py")

from .router import route  # noqa: E402

_bedrock_client = None
_chat_sessions: dict = {}


def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = phase3_llm.build_bedrock_client()
    return _bedrock_client


def handle_turn(user_text: str, session_id: str, lang: str = "yoruba") -> dict:
    """
    The core routing + handling logic, separated from ASR/TTS I/O so it can
    be tested with plain text (no audio files, no models) — see
    scripts/test_orchestrator_routing.py.

    Returns {"domain": "finance"|"chat", "reply_text": str, "detail": {...}}.
    Never raises for a domain-handling failure (LLM/DB errors are caught
    and turned into a Yorùbá-appropriate fallback reply) — but DOES let
    genuinely unexpected errors propagate, since silently swallowing those
    would hide real bugs rather than expected failure modes.
    """
    domain = route(user_text, lang)

    if domain == "finance":
        try:
            client = get_bedrock_client()
            parsed = phase5_intent.parse_intent(client, phase5_main.MODEL, user_text)
        except Exception as exc:
            return {
                "domain": "finance",
                "reply_text": "Má bìnú, nǹkan kan ṣẹlẹ̀ nígbà tí mo ń gbìyànjú láti ye ọ̀rọ̀ rẹ.",
                "detail": {"error": f"{type(exc).__name__}: {exc}"},
            }

        intent_name = parsed["intent"]
        if intent_name in ("add_expense", "add_income") and parsed["amount"] is not None:
            phase5_db.add_transaction(phase5_db.Transaction(
                kind="expense" if intent_name == "add_expense" else "income",
                amount=float(parsed["amount"]), category=parsed["category"],
                description=parsed["description"],
            ))
        balance = phase5_db.get_balance()
        reply_text = phase5_main.build_reply(parsed, balance)
        return {"domain": "finance", "reply_text": reply_text,
                "detail": {"intent": intent_name, "balance_after": balance}}

    # domain == "chat"
    try:
        client = get_bedrock_client()
        session = _chat_sessions.setdefault(session_id, phase3_llm.ConversationSession(session_id))
        reply_text = phase3_llm.generate_reply(client, session, user_text)
        return {"domain": "chat", "reply_text": reply_text, "detail": {}}
    except Exception as exc:
        return {
            "domain": "chat",
            "reply_text": "Má bìnú, n kò lè dáhùn nísisìyí.",
            "detail": {"error": f"{type(exc).__name__}: {exc}"},
        }


def handle_voice_turn(audio_path: str, session_id: str, lang: str = "yoruba") -> dict:
    """Full pipeline: real audio path in, real audio path out (plus the
    intermediate text at every stage, for debugging/logging)."""
    user_text = phase3_asr.transcribe(audio_path)
    result = handle_turn(user_text, session_id, lang)
    result["asr_text"] = user_text

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    phase3_tts.synthesize(result["reply_text"], out_path)
    result["reply_audio_path"] = out_path
    return result
