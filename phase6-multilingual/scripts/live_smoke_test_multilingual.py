"""
A REAL, live, end-to-end ASR smoke test for Igbo and Hausa -- not a mock,
not a dry-run. Mirrors phase2-asr/scripts/live_smoke_test.py's pattern
exactly (same REST preview client, same scorer, same output schema) but
parameterized per language, per Phase 6's "reuse Phase 2's harness, don't
rebuild it" design (see PLAN.md).

This is PLAN.md's Step 3: "find a small Whisper fine-tune for each [Igbo,
Hausa] on HF the same way Phase 2 found the Yorùbá one, get a first real
WER/CER data point per language before committing to the full Omnilingual
ASR matrix."

Model choice, both picked by the same method as Phase 2's Yorùbá pick
(search HF, verify ungated + a real/complete checkpoint, not a promise):
  - Hausa: CLEAR-Global/whisper-small-clearglobal-hausa-asr-1.0.0
    (openai/whisper-small fine-tuned on the TWB Voice 1.0 Hausa subset,
    cc-by-sa-4.0, real complete `model.safetensors` checkpoint).
  - Igbo: benjaminogbonna/whisper-tiny-igbo
    (openai/whisper-tiny fine-tune, apache-2.0, real complete checkpoint).
    Two other Igbo candidates that showed up in search
    (anthonysoronnadi/whisper-fine-tuned-on-igbo,
    isurut/whisper_base_finetune_cv_igbo) were checked via the HF API and
    are EMPTY repos (only a .gitattributes file, 0 bytes stored) -- not
    usable, a real trap worth flagging for anyone repeating this search.
    NCAIR1/Igbo-ASR and NCAIR1/Hausa-ASR both exist and are Whisper-small
    but are gated (`"gated": "auto"` on the HF API), same as
    NCAIR1/Yoruba-ASR in Phase 2 -- not used here for the same reason.

Data: reuses the exact datasets phase6-multilingual/manifests/ were
pulled from (shunyalabs/hausa-speech-dataset, SilencioNetwork/igbo-speech
config igbo_nigeria) but re-fetches fresh rows via the REST preview API
rather than reading the manifest CSVs -- the CSVs' audio URLs are signed
CDN links with a real ~2-week expiry and were already dead by the time
this script was written (confirmed: Hausa manifest's `Expires=` timestamp
had passed). Igbo's manifest is a permanent HF `resolve/` URL, so it
wouldn't have this specific problem, but re-fetching fresh keeps both
languages on one code path.

Igbo caveat (see PLAN.md and manifests/igbo_sample.csv): the dataset's
own `transcript` field is literally the string "unknown" for every row --
the usable reference text is in the `script` field, which is the prompt
the speaker was given to read/discuss, not a verified transcript of what
they actually said (this is a `free_speech`-type recording). Treat Igbo's
numbers here as noisier than Hausa's for that reason, not just because of
model size.

Usage:
    python3 scripts/live_smoke_test_multilingual.py --lang hausa
    python3 scripts/live_smoke_test_multilingual.py --lang igbo --n 5
    python3 scripts/live_smoke_test_multilingual.py --lang both
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase1-data" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phase2-asr" / "scripts"))

from normalize import evaluate_both_modes  # noqa: E402
from hf_preview_client import download_audio, preview_rows  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


@dataclass
class SmokeConfig:
    lang_code: str          # FLORES-style code, matches lang_config.py
    model_id: str
    dataset: str
    dataset_config: str
    text_field: str
    text_prefix_to_strip: str | None = None  # dataset-formatting artifact, not part of the utterance


SMOKE_CONFIGS: dict[str, SmokeConfig] = {
    "hausa": SmokeConfig(
        lang_code="hau_Latn",
        model_id="CLEAR-Global/whisper-small-clearglobal-hausa-asr-1.0.0",
        dataset="shunyalabs/hausa-speech-dataset",
        dataset_config="default",
        text_field="transcript",
    ),
    "igbo": SmokeConfig(
        lang_code="ibo_Latn",
        model_id="benjaminogbonna/whisper-tiny-igbo",
        dataset="SilencioNetwork/igbo-speech",
        dataset_config="igbo_nigeria",
        text_field="script",           # NOT "transcript" -- see module docstring
        text_prefix_to_strip="# ",     # dataset stores the read-prompt as a markdown heading
    ),
}


def run(lang: str, n: int) -> dict:
    cfg = SMOKE_CONFIGS[lang]
    print(f"[{lang}] Loading {cfg.model_id}...")
    from transformers import pipeline
    # chunk_length_s=30 enables Whisper's long-form chunked generation path;
    # without it, any clip over 30s (real for Igbo's free_speech recordings,
    # confirmed hitting this live: audio.wav clips run 52-68s) raises
    # ValueError instead of transcribing. Harmless for shorter clips (Hausa).
    asr = pipeline("automatic-speech-recognition", model=cfg.model_id, chunk_length_s=30)

    print(f"[{lang}] Fetching {n} real utterance(s) from {cfg.dataset} (config={cfg.dataset_config})...")
    rows_info = preview_rows(cfg.dataset, cfg.dataset_config, "train", length=n)
    if "error" in rows_info:
        raise RuntimeError(f"Could not fetch preview rows: {rows_info['error']}")

    references, hypotheses, per_utterance = [], [], []
    for i, entry in enumerate(rows_info["rows"]):
        row = entry["row"]
        reference = row[cfg.text_field]
        if cfg.text_prefix_to_strip and reference.startswith(cfg.text_prefix_to_strip):
            reference = reference[len(cfg.text_prefix_to_strip):].strip()
        audio_url = row["audio"][0]["src"]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            print(f"  [{i}] downloading audio...")
            download_audio(audio_url, tmp.name)
            print(f"  [{i}] transcribing...")
            hypothesis = asr(tmp.name)["text"].strip()

        references.append(reference)
        hypotheses.append(hypothesis)
        per_utterance.append({"reference": reference, "hypothesis": hypothesis})
        print(f"  [{i}] REF: {reference[:80]}")
        print(f"  [{i}] HYP: {hypothesis[:80]}")

    scores = evaluate_both_modes(references, hypotheses)
    note = (
        f"LIVE smoke test against a real downloaded model and real downloaded "
        f"audio from a dataset with an unstated license -- treat these numbers "
        f"as a pipeline proof, not a benchmark result to cite."
    )
    if lang == "igbo":
        note += (
            " Additional caveat: reference text is the dataset's 'script' "
            "field (the read/discussion prompt), not a verified transcript "
            "of what was actually said -- this dataset's own 'transcript' "
            "field is literally the placeholder string 'unknown'."
        )
    return {
        "backend": f"whisper/{cfg.model_id}",
        "lang": cfg.lang_code,
        "n_utterances": len(references),
        "scores": scores,
        "sample_predictions": per_utterance,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["hausa", "igbo", "both"], default="both")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model-override", default=None,
                         help="Use a different HF model id than SMOKE_CONFIGS' default, "
                              "for one-off comparisons without editing the file. Only "
                              "valid with a single --lang, not --lang both.")
    parser.add_argument("--out-suffix", default="",
                         help="Appended to the output filename, e.g. '_altmodel', so a "
                              "one-off --model-override run doesn't overwrite the default result.")
    args = parser.parse_args()

    if args.model_override and args.lang == "both":
        parser.error("--model-override requires a single --lang, not 'both'")

    langs = ["hausa", "igbo"] if args.lang == "both" else [args.lang]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        if args.model_override:
            SMOKE_CONFIGS[lang].model_id = args.model_override
        result = run(lang, args.n)
        out_path = RESULTS_DIR / f"live_smoke_test_{lang}{args.out_suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n[{lang}] n={result['n_utterances']}")
        print(f"[{lang}] tone_sensitive WER={result['scores']['tone_sensitive']['wer']}  "
              f"CER={result['scores']['tone_sensitive']['cer']}")
        print(f"[{lang}] tone_agnostic  WER={result['scores']['tone_agnostic']['wer']}  "
              f"CER={result['scores']['tone_agnostic']['cer']}")
        print(f"[{lang}] Saved to {out_path}\n")


if __name__ == "__main__":
    main()
