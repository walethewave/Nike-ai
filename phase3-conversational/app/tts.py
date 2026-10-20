"""
TTS layer. Two backends:

  "yarngpt2" (default) — saheedniyi/YarnGPT2. Supports Yorùbá, Igbo, Hausa,
      Nigerian-accented English in one model. NOT pip-installable (needs a
      git clone + separately-downloaded WavTokenizer checkpoint) — see
      PLAN.md "Setup". Not yet live-tested in this repo for exactly that
      reason: real friction, not zero.
  "mms_tts"  — a real, live-tested, LIGHTWEIGHT alternative:
      facebook/mms-tts-yor via transformers (VITS architecture, ~145MB,
      ungated, single pip install away). **Actually run in this repo**,
      2026-08-13: loaded, synthesized "Bawo ni? Mo dun lati pade yin." to a
      real 1.71s 16kHz waveform (non-silent, peak amplitude 0.17). This is
      the backend `scripts/live_pipeline_test.py` exercises. Single-speaker
      (no voice selection, unlike YarnGPT2's multiple Yorùbá voices), and
      MMS-TTS in general reads more monotone/robotic than YarnGPT2's
      LLM-based synthesis — a real quality tradeoff for the setup-friction
      savings, not a strictly-better replacement.

Select via TTS_BACKEND env var. Lazy-loaded either way: importing this
module never downloads or loads anything.
"""
from __future__ import annotations

import os

TTS_BACKEND = os.environ.get("TTS_BACKEND", "yarngpt2")
DEFAULT_SPEAKER = os.environ.get("TTS_SPEAKER", "yoruba_female2")  # yarngpt2 only
DEFAULT_SAMPLE_RATE = 24000  # yarngpt2's rate; mms_tts overrides via its own config

# mms_tts (VITS) is stochastic by architecture (a flow-based decoder + a
# stochastic duration predictor) — confirmed live, 2026-08-15: 5 calls with
# the identical input text produced 5 different output waveforms.
# Setting model.config.noise_scale/noise_scale_duration to 0 did NOT fully
# remove the variance (a real, separate finding — some randomness survives
# even with both zeroed, not yet root-caused further). What DOES work,
# proven 3/3 identical: fixing PyTorch's RNG seed before each synthesis
# call. Opt-in via TTS_SEED (unset by default, so normal/production runs
# keep natural per-call variation) — set it for reproducible test runs,
# e.g. to isolate whether ASR/routing/intent failures are caused by TTS
# variance or something else. See phase7-integration/PLAN.md.
TTS_SEED = os.environ.get("TTS_SEED")

_yarngpt_tokenizer = None
_yarngpt_model = None
_mms_model = None
_mms_tokenizer = None


def _load_yarngpt2():
    global _yarngpt_tokenizer, _yarngpt_model
    if _yarngpt_tokenizer is not None and _yarngpt_model is not None:
        return _yarngpt_tokenizer, _yarngpt_model

    wav_tokenizer_model_path = os.environ.get("WAV_TOKENIZER_MODEL_PATH")
    wav_tokenizer_config_path = os.environ.get("WAV_TOKENIZER_CONFIG_PATH")
    if not wav_tokenizer_model_path or not wav_tokenizer_config_path:
        raise RuntimeError(
            "WAV_TOKENIZER_MODEL_PATH and WAV_TOKENIZER_CONFIG_PATH env vars must "
            "point at the downloaded WavTokenizer checkpoint/config — see PLAN.md "
            "'Setup' for the download commands."
        )

    from transformers import AutoModelForCausalLM
    from yarngpt.audiotokenizer import AudioTokenizerV2  # requires `yarngpt` on PYTHONPATH

    _yarngpt_tokenizer = AudioTokenizerV2(
        tokenizer_path="saheedniyi/YarnGPT2",
        wav_tokenizer_model_path=wav_tokenizer_model_path,
        wav_tokenizer_config_path=wav_tokenizer_config_path,
    )
    _yarngpt_model = AutoModelForCausalLM.from_pretrained(
        "saheedniyi/YarnGPT2", torch_dtype="auto"
    ).to(_yarngpt_tokenizer.device)
    return _yarngpt_tokenizer, _yarngpt_model


def _load_mms_tts():
    global _mms_model, _mms_tokenizer
    if _mms_model is None or _mms_tokenizer is None:
        from transformers import VitsModel, AutoTokenizer
        _mms_model = VitsModel.from_pretrained("facebook/mms-tts-yor")
        _mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-yor")
    return _mms_model, _mms_tokenizer


def _synthesize_mms_tts(text: str, out_path: str) -> str:
    import torch
    import scipy.io.wavfile

    if TTS_SEED is not None:
        torch.manual_seed(int(TTS_SEED))

    model, tokenizer = _load_mms_tts()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().numpy()
    scipy.io.wavfile.write(out_path, model.config.sampling_rate, audio)
    return out_path


def _synthesize_yarngpt2(text: str, out_path: str, speaker: str, lang: str) -> str:
    try:
        tokenizer, model = _load_yarngpt2()
    except ImportError as exc:
        raise ImportError(
            "yarngpt2 TTS backend not installed. See phase3-conversational/PLAN.md "
            "'Setup' for the yarngpt/outetts/uroman + WavTokenizer install steps, or "
            "set TTS_BACKEND=mms_tts for the lighter, already-verified alternative."
        ) from exc

    import torchaudio

    prompt = tokenizer.create_prompt(text, lang=lang, speaker_name=speaker)
    input_ids = tokenizer.tokenize_prompt(prompt)
    output = model.generate(
        input_ids=input_ids, temperature=0.1, repetition_penalty=1.1, max_length=4000
    )
    codes = tokenizer.get_codes(output)
    audio = tokenizer.get_audio(codes)
    torchaudio.save(out_path, audio, sample_rate=DEFAULT_SAMPLE_RATE)
    return out_path


def synthesize(text: str, out_path: str, speaker: str = DEFAULT_SPEAKER, lang: str = "yoruba") -> str:
    """Writes a WAV file to out_path and returns that path."""
    if TTS_BACKEND == "mms_tts":
        return _synthesize_mms_tts(text, out_path)
    return _synthesize_yarngpt2(text, out_path, speaker, lang)
