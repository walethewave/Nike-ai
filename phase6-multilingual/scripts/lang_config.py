"""
Language configuration for the Igbo/Hausa extension. This is the whole
point of Phase 6 being cheap: everything here is a config table, not new
infrastructure. The actual benchmarking logic is imported from
phase2-asr/scripts/benchmark.py and normalize.py — Phase 6 doesn't
reimplement the harness, it parameterizes it.

Omnilingual ASR language codes (`*_Latn`) are the FLORES-200 codes the
project uses. ALL THREE — yor_Latn, ibo_Latn, hau_Latn — are CONFIRMED
supported, verified 2026-08-13 directly against the real source file
(github.com/facebookresearch/omnilingual-asr,
src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py), fetched without
installing the multi-GB package. The project's own published per-language
results (per_language_results_table_7B_llm_asr.csv, same repo) additionally
give real reported CER for the 7B LLM-ASR model on each language:

    hau_Latn: 836.4 training hours -> 3.5% CER
    ibo_Latn: 729.6 training hours -> 9.1% CER
    yor_Latn: 828.1 training hours -> 9.0% CER

This is a genuinely useful early data point for the comparative-analysis
step in PLAN.md: Hausa is non-tonal and Yorùbá/Igbo are both tonal, and
Hausa's reported CER is ~2.5x lower — directly consistent with (though not
proof of) the tonal-languages-are-harder-for-ASR hypothesis this phase set
out to test. These are Meta's own numbers on their own eval set, not this
project's — a real benchmark run (Phase 2/6 Steps) on a shared held-out set
across all three languages is still the thing to actually cite.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LanguageConfig:
    key: str                        # short internal key
    display_name: str
    omnilingual_code: str           # e.g. "yor_Latn" — FLORES-200 style
    whisper_code: str               # ISO 639-1, e.g. "yo" — for openai-whisper's `language=` param
    naijavoices_prefix: str         # e.g. "yoruba" — batch configs are "{prefix}-batch-{0,1,2}"
    naijavoices_batches: tuple      # which batch indices exist, e.g. (0, 1, 2)
    verified: bool                  # was omnilingual_code confirmed against supported_langs?


LANGUAGES: dict = {
    "yoruba": LanguageConfig(
        key="yoruba",
        display_name="Yorùbá",
        omnilingual_code="yor_Latn",
        whisper_code="yo",
        naijavoices_prefix="yoruba",
        naijavoices_batches=(0, 1, 2),
        verified=True,  # confirmed via omnilingual_asr.models.wav2vec2_llama.lang_ids.supported_langs
    ),
    "igbo": LanguageConfig(
        key="igbo",
        display_name="Igbo",
        omnilingual_code="ibo_Latn",
        whisper_code="ig",
        naijavoices_prefix="igbo",
        naijavoices_batches=(0, 1, 2),  # confirmed to exist via HF dataset_info siblings listing
        verified=True,  # confirmed 2026-08-13 directly against omnilingual-asr's real lang_ids.py source
    ),
    "hausa": LanguageConfig(
        key="hausa",
        display_name="Hausa",
        omnilingual_code="hau_Latn",
        whisper_code="ha",
        naijavoices_prefix="hausa",
        naijavoices_batches=(0, 1, 2),  # confirmed to exist via HF dataset_info siblings listing
        verified=True,  # confirmed 2026-08-13 directly against omnilingual-asr's real lang_ids.py source
    ),
}
