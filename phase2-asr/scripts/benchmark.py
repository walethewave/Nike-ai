"""
Phase 2 prototype: the benchmark harness skeleton for the model x condition
matrix described in RESEARCH_OPEN_WEIGHT_ASR.md §5.

This wires together:
  - a pluggable ASR backend interface (`ASRBackend`) with two real,
    API-verified implementations (Omnilingual ASR, Whisper) — verified
    against each library's actual documented usage, not guessed
  - the tone-aware/tone-agnostic scorer from normalize.py (already tested)
  - a manifest reader matching Phase 1's CSV schema

It does NOT ship model weights or require a GPU to import/construct — the
backends are lazy (only load the model when .transcribe() is first called),
so this file is safe to import and unit-test without downloading anything.
Actually running a backend requires installing its extra (see
requirements.txt) and having the model weights accessible (local disk or
first-run download).

Usage:
    # Structural dry run — no model downloads, proves the harness wiring:
    python3 scripts/benchmark.py --dry-run

    # Real run against a manifest (requires the relevant extras installed):
    python3 scripts/benchmark.py \\
        --manifest ../phase1-data/manifests/fleurs_sample.csv \\
        --backend omnilingual_ctc_300m \\
        --lang yor_Latn \\
        --out results/omnilingual_ctc_300m__fleurs.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import evaluate_both_modes  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class ASRBackend(ABC):
    """Common interface every model in the benchmark matrix implements."""

    name: str

    @abstractmethod
    def transcribe(self, audio_paths: list[str], lang: str) -> list[str]:
        """Return one transcription string per input audio path, in order."""
        raise NotImplementedError


class OmnilingualASRBackend(ASRBackend):
    """
    Wraps facebookresearch/omnilingual-asr. API verified against the
    project's README (2026-07-28) — see RESEARCH_OPEN_WEIGHT_ASR.md §1.
    Install with: pip install omnilingual-asr
    """

    def __init__(self, model_card: str = "omniASR_CTC_300M_v2"):
        self.name = f"omnilingual/{model_card}"
        self.model_card = model_card
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
            self._pipeline = ASRInferencePipeline(model_card=self.model_card)
        return self._pipeline

    def transcribe(self, audio_paths: list[str], lang: str) -> list[str]:
        pipeline = self._load()
        langs = [lang] * len(audio_paths)
        return pipeline.transcribe(audio_paths, lang=langs, batch_size=min(8, len(audio_paths)))


class WhisperBackend(ASRBackend):
    """
    Wraps openai-whisper (or a Whisper-family fine-tune, e.g. NCAIR1/Yoruba-ASR,
    loaded via transformers instead — see `hf_model_id`). Two modes:
      - whisper_size set: uses the `openai-whisper` package directly
      - hf_model_id set: uses transformers' pipeline("automatic-speech-recognition")
        so fine-tuned checkpoints like NCAIR1/Yoruba-ASR work the same way.
    """

    def __init__(self, whisper_size: str | None = None, hf_model_id: str | None = None):
        if bool(whisper_size) == bool(hf_model_id):
            raise ValueError("Pass exactly one of whisper_size or hf_model_id")
        self.name = f"whisper/{whisper_size or hf_model_id}"
        self.whisper_size = whisper_size
        self.hf_model_id = hf_model_id
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        if self.whisper_size:
            import whisper
            self._model = ("openai_whisper", whisper.load_model(self.whisper_size))
        else:
            from transformers import pipeline
            self._model = ("hf_pipeline", pipeline("automatic-speech-recognition", model=self.hf_model_id))
        return self._model

    def transcribe(self, audio_paths: list[str], lang: str) -> list[str]:
        kind, model = self._load()
        results = []
        # Map lang codes: yor_Latn -> yo, ibo_Latn -> ig, hau_Latn -> ha
        whisper_lang = lang.split("_")[0][:2] if "_" in lang else lang[:2]
        lang_map = {"yo": "yo", "ig": "ig", "ha": "ha", "yor": "yo", "ibo": "ig", "hau": "ha"}
        whisper_lang = lang_map.get(whisper_lang, whisper_lang)

        for path in audio_paths:
            if kind == "openai_whisper":
                out = model.transcribe(path, language=whisper_lang)
                results.append(out["text"].strip())
            else:
                # HF pipeline: try with language forcing, fall back to no forcing
                # (fine-tuned models may have outdated configs that reject language param)
                try:
                    out = model(path, generate_kwargs={"language": whisper_lang, "task": "transcribe"})
                except (ValueError, TypeError):
                    # Fine-tuned model with outdated config — skip language forcing
                    out = model(path)
                results.append(out["text"].strip())
        return results


BACKEND_REGISTRY = {
    # Omnilingual ASR (Meta) — recommended primary, Apache 2.0
    # NOTE: Not on PyPI yet, install from GitHub: pip install git+https://github.com/facebookresearch/omnilingual-asr
    "omnilingual_ctc_300m": lambda: OmnilingualASRBackend("omniASR_CTC_300M_v2"),
    "omnilingual_llm_7b": lambda: OmnilingualASRBackend("omniASR_LLM_7B_v2"),
    # Whisper family via openai-whisper package
    "whisper_large_v3": lambda: WhisperBackend(whisper_size="large-v3"),
    # Whisper family via HuggingFace transformers (fine-tunes + original)
    "whisper_large_v3_hf": lambda: WhisperBackend(hf_model_id="openai/whisper-large-v3"),
    "whisper_base_yoruba": lambda: WhisperBackend(hf_model_id="ndeclarke/whisper-base-yoruba-colab-CV17.0"),
    "whisper_small_yoruba": lambda: WhisperBackend(hf_model_id="steja/whisper-small-yoruba"),
    "ncair1_yoruba": lambda: WhisperBackend(hf_model_id="NCAIR1/Yoruba-ASR"),  # GATED — needs HF login
    # MMS (Meta) — CC-BY-NC, good for research comparison
    "mms_1b_all": lambda: WhisperBackend(hf_model_id="facebook/mms-1b-all"),
}


@dataclass
class ManifestRow:
    audio_path: str
    reference_text: str


def read_manifest(path: str, audio_root: str | None = None) -> list[ManifestRow]:
    """
    Reads a Phase 1 manifest CSV. Phase 1's manifests reference streamed
    dataset rows (source/source_config/row_index), not local file paths —
    materializing them to local WAV files for repeated benchmark runs is a
    Phase 1→2 handoff step not yet implemented here (streaming datasets
    don't give stable local paths). For now this reads a SIMPLER manifest
    shape with an explicit `audio_path` column, produced once you've decided
    to materialize a fixed benchmark set locally. See PLAN.md "Handoff note".
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            audio_path = r["audio_path"]
            if audio_root:
                audio_path = str(Path(audio_root) / audio_path)
            rows.append(ManifestRow(audio_path=audio_path, reference_text=r["text"]))
    return rows


def run_benchmark(backend: ASRBackend, manifest_rows: list[ManifestRow], lang: str) -> dict:
    audio_paths = [r.audio_path for r in manifest_rows]
    references = [r.reference_text for r in manifest_rows]

    hypotheses = backend.transcribe(audio_paths, lang=lang)
    if len(hypotheses) != len(references):
        raise RuntimeError(
            f"Backend {backend.name} returned {len(hypotheses)} hypotheses "
            f"for {len(references)} references — must be 1:1 and in order."
        )

    scores = evaluate_both_modes(references, hypotheses)
    return {
        "backend": backend.name,
        "lang": lang,
        "n_utterances": len(references),
        "scores": scores,
        "sample_predictions": [
            {"reference": r, "hypothesis": h} for r, h in list(zip(references, hypotheses))[:5]
        ],
    }


def _dry_run():
    """Proves the harness wiring (manifest → backend interface → scorer →
    result JSON) without downloading any model or audio — uses a fake
    backend that echoes the reference text back, so WER should be exactly 0."""
    print("Dry run: fake backend, synthetic manifest, no network/model access.\n")

    class EchoBackend(ASRBackend):
        name = "echo (dry-run fake)"

        def transcribe(self, audio_paths: list[str], lang: str) -> list[str]:
            return [f"REFERENCE_FOR::{p}" for p in audio_paths]

    rows = [
        ManifestRow(audio_path="a.wav", reference_text="REFERENCE_FOR::a.wav"),
        ManifestRow(audio_path="b.wav", reference_text="REFERENCE_FOR::b.wav"),
    ]
    result = run_benchmark(EchoBackend(), rows, lang="yor_Latn")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    assert result["scores"]["tone_sensitive"]["wer"] == 0.0, "Dry run should score perfect WER"
    print("\nDry run OK — harness wiring is correct.")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 ASR benchmark harness")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", help="CSV with columns: audio_path,text")
    parser.add_argument("--audio-root", help="Optional root dir to prepend to audio_path")
    parser.add_argument("--backend", choices=list(BACKEND_REGISTRY.keys()))
    parser.add_argument("--lang", default="yor_Latn")
    parser.add_argument("--out", help="Where to write the result JSON")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
        return

    if not (args.manifest and args.backend and args.out):
        parser.error("--manifest, --backend, and --out are required unless --dry-run is given")

    rows = read_manifest(args.manifest, args.audio_root)
    backend = BACKEND_REGISTRY[args.backend]()
    result = run_benchmark(backend, rows, args.lang)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"backend={result['backend']}  n={result['n_utterances']}")
    print(f"tone_sensitive WER={result['scores']['tone_sensitive']['wer']}  "
          f"CER={result['scores']['tone_sensitive']['cer']}")
    print(f"tone_agnostic  WER={result['scores']['tone_agnostic']['wer']}  "
          f"CER={result['scores']['tone_agnostic']['cer']}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
