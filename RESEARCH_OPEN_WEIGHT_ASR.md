# Open-Weight Speech→Text Models for Yorùbá — Research Brief

**Compiled:** 2026-07-28 · **Purpose:** decide which open-weight ASR model to build Phase 2 on, and what data to fine-tune it with.

> **Read this before Phase 1.** Two things here invalidate assumptions in the original roadmap: the data-collection phase is largely already solved by an existing dataset, and there is a newer/better base model than Whisper. Both save you weeks.

---

## TL;DR — what to actually use

| Layer | Pick | Why |
|---|---|---|
| **Base ASR model** | **Meta Omnilingual ASR** (`omniASR_CTC_300M_v2` on your Mac; `omniASR_LLM_7B_v2` on rented GPU) | Apache 2.0, 1,600+ languages, `yor_Latn` natively supported, purpose-built for low-resource languages. Released Nov 2025 — after most Yorùbá-ASR papers, so *nobody has benchmarked it properly on Yorùbá yet*. That gap is your paper. |
| **Fallback / comparison base** | Whisper large-v3 + `NCAIR1/Yoruba-ASR` (existing Whisper-small Yorùbá fine-tune) | Whisper is the field's default baseline; NCAIR1 is the existing Yorùbá SOTA-ish checkpoint you must beat or explain. |
| **Also compare** | MMS-1B-all (`facebook/mms-1b-all`) | The pre-Omnilingual multilingual standard; heavily used in African-NLP papers, so reviewers expect it. |
| **Training data** | **NaijaVoices** (~600 h Yorùbá) + ÌròyìnSpeech (~48 h) + Common Voice yo (~120 h) | ~750 h available *today* without scraping anything. |
| **TTS (Phase 3)** | **YarnGPT2** (`saheedniyi/YarnGPT2`) | Supports Yorùbá/Igbo/Hausa + Nigerian English in one model. Supersedes YarnGPT-local. |

---

## 1. The base-model landscape (open weights only)

### Meta Omnilingual ASR — the recommended base

Released Nov 2025, Apache 2.0 (commercial use OK). Built with Mozilla Common Voice, **African Next Voices**, and Lanfrica — so African languages were a design target, not an afterthought.

- **Coverage:** 1,600+ languages trained; 5,400+ reachable via zero-shot in-context learning. Yorùbá is natively supported as `yor_Latn`.
- **Architecture families:** wav2vec 2.0 encoder (up to 7B, pre-trained on 4.3M hours), plus CTC-head and LLM-decoder ASR variants.
- **Reported quality:** CER < 10% on 78% of its 1,600+ languages.

**Model cards and hardware** (VRAM at batch=1, 30 s audio, BF16):

| Model card | Size (FP32) | VRAM | Runs on your M5 Mac? |
|---|---|---|---|
| `omniASR_CTC_300M_v2` | 1.3 GiB | ~5 GiB | **Yes** — start here |
| `omniASR_CTC_1B_v2` | 3.7 GiB | ~6 GiB | Likely (unified memory) |
| `omniASR_LLM_300M_v2` | 6.1 GiB | ~5 GiB | Yes |
| `omniASR_LLM_1B_v2` | 8.5 GiB | ~6 GiB | Probably |
| `omniASR_LLM_3B_v2` | 17.0 GiB | ~10 GiB | Rent a GPU |
| `omniASR_LLM_7B_v2` | 30.0 GiB | ~17 GiB | Rent a GPU (A100/H100) |

There are also `omniASR_LLM_Unlimited_*_v2` variants — the CTC/LLM models cap at **40 seconds** of audio; the Unlimited ones handle 15+ minutes. For conversational turns (Phase 3) 40 s is fine; for long-form data prep (Phase 1) use Unlimited or pre-segment.

```bash
pip install omnilingual-asr
```

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_300M_v2")
out = pipeline.transcribe(["/path/to/audio.wav"], lang=["yor_Latn"], batch_size=2)
```

Check support before assuming:
```python
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
assert "yor_Latn" in supported_langs
```

### The rest of the 2026 open-ASR field — and why most of it is irrelevant to you

The Hugging Face Open ASR Leaderboard is currently topped by models that are **English- or European-centric**. Don't be seduced by their WER numbers; they're measured on English benchmarks:

| Model | Params | License | Languages | Avg WER | Useful for Yorùbá? |
|---|---|---|---|---|---|
| ARK-ASR-3B | 3B | Apache 2.0 | n/s | 5.04% | ❌ not Yorùbá-relevant |
| Granite Speech 4.1 | 2B | Apache 2.0 | 6 | 5.33% | ❌ |
| Cohere Transcribe | 2B | Apache 2.0 | 14 | 5.42% | ❌ |
| Canary-Qwen-2.5B | 2.5B | CC-BY-4.0 | English only | 5.63% | ❌ |
| Qwen3-ASR-1.7B | 1.7B | Apache 2.0 | 52 + dialects | 5.76% | ❌ (no Yorùbá) |
| Parakeet TDT 0.6B v3 | 0.6B | CC-BY-4.0 | 25 European | 6.32% | ❌ |
| Whisper large-v3 | 1.55B | MIT | 99 | — | ✅ baseline (Yorùbá is in the 99, poorly) |
| MMS-1B-all | 1B | CC-BY-NC-4.0 ⚠️ | 1,100+ | — | ✅ comparison |
| **Omnilingual ASR** | 300M–7B | **Apache 2.0** | **1,600+** | — | ✅ **primary** |

⚠️ **MMS is CC-BY-NC** — non-commercial. Fine for a paper, a licensing problem for Phase 5. Omnilingual being Apache 2.0 is a real strategic advantage.

### Existing Yorùbá-specific checkpoints (your baselines to beat)

- **`NCAIR1/Yoruba-ASR`** — Whisper-small fine-tuned for Yorùbá, from Awarri Technologies + Nigeria's Federal Ministry of Communications, Innovation and Digital Economy. This is the closest thing to an official Nigerian Yorùbá ASR. **Benchmark against it or a reviewer will ask why you didn't.**
- `ogbi/wav2vec2-large-mms-1b-yoruba-test` — community MMS-1B Yorùbá adapter fine-tune.
- Niger-Volta-LTI (`yoruba-audio`) — long-running Yorùbá speech/NLP resource group; check for corpora and prior art.

---

## 2. Data — you do **not** need to scrape YouTube for a month

This is the biggest change to the original plan. ~750 hours of Yorùbá speech is downloadable today.

| Dataset | Yorùbá hours | License | Access |
|---|---|---|---|
| **NaijaVoices** | **~600 h** (of 1,800 h total across ig/ha/yo) | CC BY-NC-SA 4.0 ⚠️ | `naijavoices/naijavoices-dataset` on HF |
| Mozilla Common Voice (yo) | ~120 h | CC0 ✅ | HF / commonvoice.mozilla.org |
| ÌròyìnSpeech | ~48 h (42 h in-house + 6 h CV) | Check paper (open) | arXiv 2307.16071 |
| FLEURS (yo) | ~10 h | CC-BY-4.0 ✅ | `google/fleurs` on HF |
| BibleTTS (yo) | ~80 h (single-speaker, read) | CC-BY-SA ✅ | Useful for TTS more than ASR |

**NaijaVoices** — 5,000+ speakers, balanced gender and age range, each row carrying audio + transcript + language + speaker id + gender + age range. Loads as:

```python
from datasets import load_dataset
ds = load_dataset("naijavoices/naijavoices-dataset", "yoruba-batch-1", streaming=True)
```

Stream it. Do not try to materialise 600 h on a laptop.

Their reported fine-tuning gains on NaijaVoices are large: **~75.86% WER improvement on Whisper, 52.06% on MMS, 42.33% on XLSR**. Those are relative improvements over the un-finetuned baselines — which mostly tells you how bad the baselines are on Yorùbá, and that the data works.

### ⚠️ The licensing trap you need to decide on now

**NaijaVoices is CC BY-NC-SA 4.0.** Three separate problems for your roadmap:

1. **NC (non-commercial)** — a model fine-tuned on it is arguably a derivative. Phase 5 (finance tracker) as a *commercial product* is legally murky.
2. **SA (share-alike)** — viral: derivatives must carry the same license.
3. It's the **only** source big enough to matter (600 h vs ~120 h for the next best).

**Your options, decide before Phase 2:**
- **(a) Research-only track** — use NaijaVoices, publish the model under CC BY-NC-SA, keep Phase 5 a non-commercial demo. Simplest, and fine if the goal is papers + PhD/visa.
- **(b) Commercial-clean track** — train only on CC0/CC-BY data (Common Voice ~120 h + FLEURS + BibleTTS). Weaker model, fully commercialisable.
- **(c) Two models** — publish a strong NC research model *and* a weaker permissive one. Doubles training cost, but the license comparison ("what does NC-only data cost you in WER?") is itself a publishable result and a genuinely useful finding for African-NLP builders.

I'd suggest **(c)** if compute allows, **(a)** otherwise. This is a real decision, not a formality — make it deliberately.

---

## 3. Prior art you must cite (and must not duplicate)

Yorùbá ASR is **not** an empty field. The original roadmap's framing ("nobody has done this") is too strong for ASR specifically. Where the gap genuinely still exists is the **end-to-end conversational voice loop**, not ASR alone.

- **ÒWE-Voice** (AfricaNLP 2026) — evaluates monolingual vs multilingual ASR on a Yorùbá *proverb* dataset; finds current systems miss cultural nuance. Directly adjacent to your work.
- **"Tone in Yoruba ASR"** (LoResLM 2026) — evaluates the impact of tone marking. **Critical for you**: Yorùbá is tonal and diacritics carry meaning, so *how you normalise text before computing WER changes your headline number*. See §4.
- **"Voices Unheard: NLP Resources and Models for Yorùbá Regional Dialects"** (arXiv 2406.19564) — dialect variation.
- **ASR for African Low-Resource Languages: A Systematic Literature Review** (arXiv 2510.01145) — read this first for related-work framing.
- **Omnilingual ASR** (arXiv 2511.09690) — your base model's paper.
- **NaijaVoices** (Interspeech 2025, arXiv 2505.20564) — your data's paper.

**Revised novelty claim.** Not "first Yorùbá ASR" (false). Defensible instead:
> First open, reproducible, evaluated **end-to-end conversational voice agent** (speech-in → speech-out, with dialogue state) for Yorùbá — plus the first systematic benchmark of Omnilingual ASR on Yorùbá against Whisper/MMS/NCAIR1 under tone-aware and tone-agnostic scoring.

That is honest, still novel, and much harder to reject.

---

## 4. Evaluation gotcha: WER on a tonal language

Do not just call `jiwer.wer()` and report one number. Yorùbá uses underdots (ẹ ọ ṣ) and tone marks (á à é è í ì ó ò ú ù ń). Decisions that swing WER by large margins:

- **Diacritic-sensitive vs diacritic-stripped scoring.** Stripping makes every model look better and hides the hardest part of the problem. Report **both** — that contrast is a finding.
- **Unicode normalisation (NFC vs NFD).** `ọ́` can be one codepoint or three. Without normalising, identical-looking strings score as errors. Normalise to NFC on both sides.
- **CER alongside WER.** For morphologically rich / tonal languages CER is often more informative; Omnilingual's own paper reports CER.

Phase 2's prototype (`normalize.py`) implements all of these as switchable modes.

---

## 5. Recommended benchmark matrix for Phase 2

Run every cell, report the whole table. This *is* the paper's core result.

| Model | Zero-shot | Fine-tuned on CV-only (permissive) | Fine-tuned on NaijaVoices (NC) |
|---|---|---|---|
| Whisper large-v3 | ☐ | ☐ | ☐ |
| `NCAIR1/Yoruba-ASR` | ☐ (already tuned) | — | ☐ |
| MMS-1B-all | ☐ | ☐ | ☐ |
| `omniASR_CTC_300M_v2` | ☐ | ☐ | ☐ |
| `omniASR_LLM_7B_v2` | ☐ | ☐ (if compute) | ☐ (if compute) |

× scored under {diacritic-sensitive WER, diacritic-stripped WER, CER} on a **held-out test set you never train on**.

---

## Sources

- [Best Open Speech Recognition (ASR) Models in 2026 — MarkTechPost](https://www.marktechpost.com/2026/07/23/best-open-speech-recognition-asr-models-in-2026-wer-languages-latency-and-license-compared/)
- [Meta AI Releases Omnilingual ASR — MarkTechPost](https://www.marktechpost.com/2025/11/11/meta-ai-releases-omnilingual-asr-a-suite-of-open-source-multilingual-speech-recognition-models-for-1600-languages/)
- [facebookresearch/omnilingual-asr — GitHub](https://github.com/facebookresearch/omnilingual-asr)
- [Omnilingual ASR paper — arXiv 2511.09690](https://arxiv.org/pdf/2511.09690)
- [The NaijaVoices Dataset — arXiv 2505.20564](https://arxiv.org/abs/2505.20564)
- [naijavoices/naijavoices-dataset — Hugging Face](https://huggingface.co/datasets/naijavoices/naijavoices-dataset)
- [NCAIR1/Yoruba-ASR — Hugging Face](https://huggingface.co/NCAIR1/Yoruba-ASR)
- [ÒWE-Voice: Evaluating ASR with Yoruba Proverbs — AfricaNLP 2026](https://aclanthology.org/2026.africanlp-main.8/)
- [Tone in Yoruba ASR — LoResLM 2026](https://aclanthology.org/2026.loreslm-1.14.pdf)
- [ASR for African Low-Resource Languages: Systematic Literature Review — arXiv 2510.01145](https://arxiv.org/pdf/2510.01145)
- [Voices Unheard: Yorùbá Regional Dialects — arXiv 2406.19564](https://arxiv.org/pdf/2406.19564)
- [ÌròyìnSpeech: A multi-purpose Yorùbá Speech Corpus — arXiv 2307.16071](https://arxiv.org/html/2307.16071v2)
- [YarnGPT2 — Hugging Face](https://huggingface.co/saheedniyi/YarnGPT2)
- [saheedniyi02/yarngpt — GitHub](https://github.com/saheedniyi02/yarngpt)
- [AfricaNLP Workshop — Call for Papers](https://africanlp-workshop.github.io/cfp.html)
