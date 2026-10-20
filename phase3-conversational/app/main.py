"""
Phase 3 prototype: the generic conversational voice AI, wired as a FastAPI
app. Speech-in → speech-out is the target shape, but the endpoints are
split so each layer can be tested independently:

  POST /chat/text   — text in, text out. Fully real right now: only needs
                       AWS credentials (llm.py), no other model downloads.
  POST /chat/voice   — audio in, audio out. The full pipeline (asr.py →
                       llm.py → tts.py). Needs ASR + TTS deps installed and
                       model weights available; will raise a clear error
                       naming the missing piece if they aren't.
  GET  /health       — liveness + which layers are actually importable.

Conversation state is in-memory, keyed by session_id (see llm.py — this is
explicitly a prototype limitation, not a production design).

Run:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import asr, llm, tts

app = FastAPI(title="Naija Voice AI — Phase 3 Conversational Prototype")

_sessions: dict[str, llm.ConversationSession] = {}
_client = None


def get_client():
    global _client
    if _client is None:
        _client = llm.build_bedrock_client()
    return _client


def get_session(session_id: str) -> llm.ConversationSession:
    if session_id not in _sessions:
        _sessions[session_id] = llm.ConversationSession(session_id)
    return _sessions[session_id]


class ChatTextRequest(BaseModel):
    session_id: Optional[str] = None
    text: str


class ChatTextResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/health")
def health():
    layers = {}
    try:
        import anthropic  # noqa: F401
        layers["llm"] = "importable"
    except ImportError:
        layers["llm"] = "MISSING (pip install anthropic[bedrock])"

    try:
        import omnilingual_asr  # noqa: F401
        layers["asr"] = "importable"
    except ImportError:
        layers["asr"] = "not installed (optional — see PLAN.md)"

    try:
        import yarngpt  # noqa: F401
        layers["tts"] = "importable"
    except ImportError:
        layers["tts"] = "not installed (optional — see PLAN.md)"

    return {"status": "ok", "layers": layers}


@app.post("/chat/text", response_model=ChatTextResponse)
def chat_text(req: ChatTextRequest):
    """Text-only round trip — proves the LLM layer + conversation memory
    without needing any audio model installed. This is the endpoint to hit
    first when setting this up."""
    session_id = req.session_id or str(uuid.uuid4())
    session = get_session(session_id)

    try:
        client = get_client()
        reply = llm.generate_reply(client, session, req.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    return ChatTextResponse(session_id=session_id, reply=reply)


@app.post("/chat/voice")
async def chat_voice(session_id: Optional[str] = Form(None), audio: UploadFile = File(...)):
    """Full pipeline: audio in -> ASR -> LLM -> TTS -> audio out."""
    session_id = session_id or str(uuid.uuid4())
    session = get_session(session_id)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_f.write(await audio.read())
        in_path = in_f.name

    try:
        user_text = asr.transcribe(in_path)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    finally:
        os.unlink(in_path)

    try:
        client = get_client()
        reply_text = llm.generate_reply(client, session, user_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        tts.synthesize(reply_text, out_path)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return FileResponse(
        out_path, media_type="audio/wav",
        headers={"X-Session-Id": session_id, "X-User-Text": user_text, "X-Reply-Text": reply_text},
    )


@app.get("/")
def root():
    return JSONResponse({
        "name": "Naija Voice AI — Phase 3 Conversational Prototype",
        "endpoints": ["/health", "/chat/text (POST)", "/chat/voice (POST)"],
    })
