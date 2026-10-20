"""
ASR layer for the conversational pipeline. Two backends:

  "omnilingual" (default) — Omnilingual ASR, per Phase 2's research. Not yet
      live-tested in this repo (multi-GB, not installed here) — see
      phase2-asr/PLAN.md for why it's the recommended long-term choice.
  "whisper_hf"  — a real, live-tested, LIGHTWEIGHT alternative:
      ndeclarke/whisper-base-yoruba-colab-CV17.0 via transformers (~292MB).
      This is the one actually exercised by
      phase3-conversational/scripts/live_pipeline_test.py — proven to load
      and transcribe real audio in this repo, 2026-08-13. Same model Phase
      2's live_smoke_test.py uses; see that file's PLAN.md note for its
      measured (fairly high, ~70%) WER — good enough to prove the pipeline
      wiring, not yet good enough to ship. Swap for whatever Phase 2's real
      benchmark matrix ends up recommending.

Select via ASR_BACKEND env var. Lazy-loaded either way: importing this
module never downloads or loads a model.
"""
from __future__ import annotations

import os

ASR_BACKEND = os.environ.get("ASR_BACKEND", "omnilingual")
DEFAULT_ASR_MODEL_CARD = os.environ.get("ASR_MODEL_CARD", "omniASR_CTC_300M_v2")
DEFAULT_ASR_LANG = os.environ.get("ASR_LANG", "yor_Latn")
# Phase 2 benchmark (2026-08-14): steja/whisper-small-yoruba achieves 46% WER,
# significantly better than ndeclarke/whisper-base-yoruba (69% WER)
WHISPER_HF_MODEL_ID = os.environ.get("WHISPER_HF_MODEL_ID", "steja/whisper-small-yoruba")

_omnilingual_pipeline = None
_whisper_hf_pipeline = None


def _load_omnilingual():
    global _omnilingual_pipeline
    if _omnilingual_pipeline is None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        _omnilingual_pipeline = ASRInferencePipeline(model_card=DEFAULT_ASR_MODEL_CARD)
    return _omnilingual_pipeline


def _load_whisper_hf():
    global _whisper_hf_pipeline
    if _whisper_hf_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _whisper_hf_pipeline = hf_pipeline("automatic-speech-recognition", model=WHISPER_HF_MODEL_ID)
    return _whisper_hf_pipeline


def transcribe(audio_path: str, lang: str = DEFAULT_ASR_LANG) -> str:
    """
    Raises ImportError with a clear message if the selected backend's deps
    aren't installed, rather than a confusing traceback — this module is
    meant to be optional at import time (see module docstring).
    """
    if ASR_BACKEND == "whisper_hf":
        try:
            pipeline = _load_whisper_hf()
        except ImportError as exc:
            raise ImportError(
                "whisper_hf ASR backend not installed. Run: pip install transformers torch"
            ) from exc
        return pipeline(audio_path)["text"].strip()

    try:
        pipeline = _load_omnilingual()
    except ImportError as exc:
        raise ImportError(
            "omnilingual ASR backend not installed. Run: pip install omnilingual-asr, "
            "or set ASR_BACKEND=whisper_hf for the lighter, already-verified alternative "
            "(see phase2-asr/PLAN.md)."
        ) from exc

    results = pipeline.transcribe([audio_path], lang=[lang], batch_size=1)
    return results[0]
