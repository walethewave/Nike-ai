"""
Phase 7 prototype: the actual "connect everything" service. One FastAPI
app, wrapping the orchestrator that routes each turn to either Phase 5's
finance logic or Phase 3's general chat, through Phase 3's ASR/TTS layers.

Run:
    python3 -m uvicorn app.main:app --reload --port 8002
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import orchestrator

app = FastAPI(title="Naija Voice AI — Phase 7 Integration Prototype")


class ConverseTextRequest(BaseModel):
    session_id: Optional[str] = None
    text: str
    lang: str = "yoruba"


class ConverseTextResponse(BaseModel):
    session_id: str
    domain: str
    reply: str


@app.get("/health")
def health():
    return {"status": "ok", "connects": ["phase3-conversational (ASR/LLM/TTS)", "phase5-finance (intent+db)"]}


@app.post("/converse/text", response_model=ConverseTextResponse)
def converse_text(req: ConverseTextRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = orchestrator.handle_turn(req.text, session_id, req.lang)
    return ConverseTextResponse(session_id=session_id, domain=result["domain"], reply=result["reply_text"])


@app.post("/converse/voice")
async def converse_voice(
    session_id: Optional[str] = Form(None),
    lang: str = Form("yoruba"),
    audio: UploadFile = File(...),
):
    session_id = session_id or str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_f.write(await audio.read())
        in_path = in_f.name

    try:
        result = orchestrator.handle_voice_turn(in_path, session_id, lang)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {exc}") from exc
    finally:
        os.unlink(in_path)

    return FileResponse(
        result["reply_audio_path"], media_type="audio/wav",
        headers={
            "X-Session-Id": session_id,
            "X-Domain": result["domain"],
            "X-ASR-Text": result["asr_text"],
            "X-Reply-Text": result["reply_text"],
        },
    )


@app.get("/")
def root():
    return {
        "name": "Naija Voice AI — Phase 7 Integration Prototype",
        "endpoints": ["/health", "/converse/text (POST)", "/converse/voice (POST)"],
    }
