"""
Reads Phase 2's benchmark result JSONs (the exact schema produced by
phase2-asr/scripts/benchmark.py's run_benchmark()) and emits a results
table in both LaTeX (booktabs) and Markdown, for dropping into
paper/sections/06_results.tex.

Each input JSON is expected to look like:
{
  "backend": "omnilingual/omniASR_CTC_300M_v2",
  "lang": "yor_Latn",
  "n_utterances": 500,
  "scores": {
    "tone_sensitive": {"mode": "...", "wer": 0.42, "cer": 0.18, "n": 500},
    "tone_agnostic":  {"mode": "...", "wer": 0.31, "cer": 0.12, "n": 500}
  },
  "sample_predictions": [...]
}

Usage:
    # Against real Phase 2 output, once it exists:
    python3 scripts/generate_results_table.py --results-dir ../phase2-asr/results

    # Against the bundled synthetic sample (proves this script works NOW):
    python3 scripts/generate_results_table.py --results-dir scripts/sample_results
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    paths = sorted(glob.glob(str(Path(results_dir) / "*.json")))
    if not paths:
        raise FileNotFoundError(f"No *.json files found in {results_dir}")
    results = []
    skipped = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        # phase2-asr/results also holds fine-tuning run logs (a different
        # schema, no "scores" key) alongside benchmark.py's output — only
        # the latter belongs in the ASR benchmark table.
        if "scores" not in data:
            skipped.append(p)
            continue
        results.append(data)
    if skipped:
        print(f"Skipped {len(skipped)} non-benchmark file(s): {', '.join(Path(p).name for p in skipped)}")
    return results


def to_markdown_table(results: list[dict]) -> str:
    lines = [
        "| Model | n | Tone-sensitive WER | Tone-sensitive CER | Tone-agnostic WER | Tone-agnostic CER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: r["scores"]["tone_sensitive"]["wer"]):
        ts = r["scores"]["tone_sensitive"]
        ta = r["scores"]["tone_agnostic"]
        lines.append(
            f"| {r['backend']} | {r['n_utterances']} | "
            f"{ts['wer']:.3f} | {ts['cer']:.3f} | {ta['wer']:.3f} | {ta['cer']:.3f} |"
        )
    return "\n".join(lines)


def _escape_latex(s: str) -> str:
    return s.replace("_", r"\_")


def to_latex_table(results: list[dict], caption: str = "Yor\\`ub\\'a ASR benchmark results.") -> str:
    rows = []
    for r in sorted(results, key=lambda r: r["scores"]["tone_sensitive"]["wer"]):
        ts = r["scores"]["tone_sensitive"]
        ta = r["scores"]["tone_agnostic"]
        rows.append(
            f"{_escape_latex(r['backend'])} & {r['n_utterances']} & "
            f"{ts['wer']:.3f} & {ts['cer']:.3f} & {ta['wer']:.3f} & {ta['cer']:.3f} \\\\"
        )
    rows_str = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Model & $n$ & \\multicolumn{{2}}{{c}}{{Tone-sensitive}} & \\multicolumn{{2}}{{c}}{{Tone-agnostic}} \\\\
 & & WER & CER & WER & CER \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{tab:asr-results}}
\\end{{table}}"""


def main():
    parser = argparse.ArgumentParser(description="Generate paper results tables from Phase 2 benchmark JSONs")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-md", default=None, help="Write Markdown table here (else stdout)")
    parser.add_argument("--out-tex", default=None, help="Write LaTeX table here (else stdout)")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result file(s) from {args.results_dir}\n", flush=True)

    md = to_markdown_table(results)
    tex = to_latex_table(results)

    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"Markdown table written to {args.out_md}")
    else:
        print("=== Markdown ===")
        print(md)

    if args.out_tex:
        Path(args.out_tex).write_text(tex, encoding="utf-8")
        print(f"LaTeX table written to {args.out_tex}")
    else:
        print("\n=== LaTeX ===")
        print(tex)


if __name__ == "__main__":
    main()
