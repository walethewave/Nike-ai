"""
Analyze a saved multi-provider experiment run: print a per-provider A-vs-B
report plus a cross-provider comparison (Claude Haiku 4.5 vs Kimi vs GLM),
and export a CSV for a native Yoruba speaker to rate fluency, naturalness,
task accuracy, faithfulness, and groundedness by hand.

Does not require any provider SDK/credentials — it only reads the saved
JSON, so it works fine offline / on a machine without API keys.

Usage:
    python analyze.py results/run_20260727_120000.json
    python analyze.py                # uses the most recent run in results/
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

FALLBACK_DISPLAY_NAMES = {
    "claude_haiku": "Claude Haiku 4.5",
    "kimi": "Kimi",
    "glm": "GLM",
}


def latest_run_path() -> str:
    runs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_*.json")))
    if not runs:
        print("No result files found in results/. Run experiment.py first.", file=sys.stderr)
        sys.exit(1)
    return runs[-1]


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _new_accumulator():
    return {"cs": [], "ascii": [], "diac": [], "nonlatin": [], "words": [],
            "faith": [], "ground": [], "halluc": 0, "refusals": 0, "errors": 0}


def _finalize(acc: dict) -> dict:
    out = {
        "mean_code_switch_rate": mean(acc["cs"]),
        "mean_ascii_only_word_ratio": mean(acc["ascii"]),
        "mean_diacritic_density_per_100_chars": mean(acc["diac"]),
        "total_non_latin_chars": sum(acc["nonlatin"]) if acc["nonlatin"] else 0,
        "mean_word_count": mean(acc["words"]),
        "refusals": acc["refusals"],
        "errors": acc["errors"],
    }
    if acc["faith"] or acc["ground"]:
        out["mean_faithfulness_1to5"] = mean(acc["faith"])
        out["mean_groundedness_1to5"] = mean(acc["ground"])
        out["hallucinations_detected"] = acc["halluc"]
    return out


def display_name(run: dict, provider_key: str) -> str:
    names = run.get("provider_display_names") or {}
    return names.get(provider_key, FALLBACK_DISPLAY_NAMES.get(provider_key, provider_key))


def summarize(run: dict) -> dict:
    providers_run = run.get("providers_run") or sorted(
        {pk for item in run["items"] for pk in item.get("results", {})}
    )

    summary = {
        "n_items": len(run["items"]),
        "judge_ran": bool(run.get("with_judge")),
        "providers_run": providers_run,
        "providers": {},
    }

    for provider_key in providers_run:
        acc_a, acc_b = _new_accumulator(), _new_accumulator()

        for item in run["items"]:
            res = item.get("results", {}).get(provider_key)
            if not res:
                continue
            for acc, cond_key in [(acc_a, "condition_A_english_prompt"), (acc_b, "condition_B_yoruba_prompt")]:
                cond = res[cond_key]
                if cond["stop_reason"] == "error":
                    acc["errors"] += 1
                    continue
                if cond["stop_reason"] == "refusal":
                    acc["refusals"] += 1
                    continue
                m = cond["metrics"]
                acc["cs"].append(m["code_switch_rate"])
                acc["ascii"].append(m["ascii_only_word_ratio"])
                acc["diac"].append(m["diacritic_density_per_100_chars"])
                acc["nonlatin"].append(m["non_latin_script_char_count"])
                acc["words"].append(m["word_count"])
                if "llm_judge" in cond:
                    j = cond["llm_judge"]
                    acc["faith"].append(j.get("faithfulness_score"))
                    acc["ground"].append(j.get("groundedness_score"))
                    if j.get("hallucination_detected"):
                        acc["halluc"] += 1

        summary["providers"][provider_key] = {
            "display_name": display_name(run, provider_key),
            "model": (run.get("provider_models") or {}).get(provider_key, "unknown"),
            "condition_A_english_prompt": _finalize(acc_a),
            "condition_B_yoruba_prompt": _finalize(acc_b),
        }

    return summary


def _print_ab_table(label: str, a: dict, b: dict, judge_ran: bool):
    rows = [
        ("Mean code-switch rate (lower=better)", a["mean_code_switch_rate"], b["mean_code_switch_rate"]),
        ("Mean ASCII-only word ratio", a["mean_ascii_only_word_ratio"], b["mean_ascii_only_word_ratio"]),
        ("Mean diacritic density /100 chars", a["mean_diacritic_density_per_100_chars"],
         b["mean_diacritic_density_per_100_chars"]),
        ("Total non-Latin stray chars", a["total_non_latin_chars"], b["total_non_latin_chars"]),
        ("Mean response word count", a["mean_word_count"], b["mean_word_count"]),
        ("Refusals", a["refusals"], b["refusals"]),
        ("Errors", a["errors"], b["errors"]),
    ]
    if judge_ran and "mean_faithfulness_1to5" in a:
        rows.extend([
            ("LLM-judge mean faithfulness (1-5)", a.get("mean_faithfulness_1to5"), b.get("mean_faithfulness_1to5")),
            ("LLM-judge mean groundedness (1-5)", a.get("mean_groundedness_1to5"), b.get("mean_groundedness_1to5")),
            ("LLM-judge hallucinations (count)", a.get("hallucinations_detected"), b.get("hallucinations_detected")),
        ])

    print(f"\n--- {label} ---")
    print(f"{'Metric':<50} {'A: English':>14} {'B: Yoruba':>14}")
    print("-" * 80)
    for name, av, bv in rows:
        print(f"{name:<50} {str(av):>14} {str(bv):>14}")


def _print_cross_provider_table(title: str, summary: dict, condition_key: str, judge_ran: bool):
    providers = summary["providers_run"]
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")

    metrics_rows = [
        ("Mean code-switch rate (lower=better)", "mean_code_switch_rate"),
        ("Mean diacritic density /100 chars", "mean_diacritic_density_per_100_chars"),
        ("Total non-Latin stray chars", "total_non_latin_chars"),
        ("Mean response word count", "mean_word_count"),
        ("Refusals", "refusals"),
        ("Errors", "errors"),
    ]
    if judge_ran:
        metrics_rows.extend([
            ("LLM-judge mean faithfulness (1-5)", "mean_faithfulness_1to5"),
            ("LLM-judge mean groundedness (1-5)", "mean_groundedness_1to5"),
            ("LLM-judge hallucinations (count)", "hallucinations_detected"),
        ])

    header = f"{'Metric':<42}" + "".join(f"{p:>13}" for p in providers)
    print(header)
    print("-" * len(header))
    for label, field in metrics_rows:
        cells = "".join(
            f"{str(summary['providers'][p][condition_key].get(field)):>13}" for p in providers
        )
        print(f"{label:<42}{cells}")


def print_report(run: dict, summary: dict):
    print("=" * 80)
    print("YORUBA SYSTEM-PROMPT EXPERIMENT — CROSS-PROVIDER COMPARISON REPORT")
    print("=" * 80)
    print(f"Items: {summary['n_items']}")
    print(f"Judge: {'on (' + str(run.get('judge_model')) + ')' if summary['judge_ran'] else 'off'}")
    print("Providers run:")
    for p in summary["providers_run"]:
        info = summary["providers"][p]
        print(f"  - {p}: {info['display_name']}  (model: {info['model']})")

    for provider_key in summary["providers_run"]:
        info = summary["providers"][provider_key]
        print(f"\n{'#' * 80}")
        print(f"# {info['display_name']}  ({provider_key})")
        print(f"{'#' * 80}")
        _print_ab_table(
            f"{info['display_name']}: Condition A (English prompt) vs B (Yoruba prompt)",
            info["condition_A_english_prompt"],
            info["condition_B_yoruba_prompt"],
            summary["judge_ran"],
        )

    if len(summary["providers_run"]) > 1:
        _print_cross_provider_table(
            "CROSS-PROVIDER COMPARISON — Condition A (English system prompt)",
            summary, "condition_A_english_prompt", summary["judge_ran"],
        )
        _print_cross_provider_table(
            "CROSS-PROVIDER COMPARISON — Condition B (Yoruba system prompt)",
            summary, "condition_B_yoruba_prompt", summary["judge_ran"],
        )

    print("\nNOTE: code-switch rate, diacritic density, and LLM-judge scores are all")
    print("heuristics, not ground truth, and are not comparable across providers with")
    print("perfect fairness (different models have different verbosity/style baselines).")
    print("The decisive signal is the native-speaker rating CSV — see")
    print("human_rating_<run>.csv, generated alongside this report.")

    print("\nPer-item breakdown:")
    for item in run["items"]:
        line = f"  {item['id']:<20} ({item['category']})"
        for provider_key, res in item.get("results", {}).items():
            a_cs = res["condition_A_english_prompt"]["metrics"]["code_switch_rate"]
            b_cs = res["condition_B_yoruba_prompt"]["metrics"]["code_switch_rate"]
            line += f"  |  {provider_key}: A={a_cs} B={b_cs}"
        print(line)


def export_human_rating_csv(run: dict, out_path: str):
    """
    One row per (item, provider, condition). Columns for a native speaker:
      - fluency (1-5), naturalness (1-5), task_accuracy (1-5)
      - faithfulness (1-5), groundedness (1-5)
      - code_switch_observed (yes/no), notes
    LLM-judge scores (if the run used --with-judge) are included as
    reference columns so the human rater can agree/disagree, not as a
    substitute. For a blind rating pass, shuffle rows and hide the
    'provider'/'condition'/'llm_judge_*' columns before handing this off,
    then re-join by item_id + response text afterward.
    """
    fieldnames = [
        "item_id", "category", "provider", "provider_model", "condition",
        "input", "reference", "response",
        "fluency_1to5", "naturalness_1to5", "task_accuracy_1to5",
        "faithfulness_1to5", "groundedness_1to5",
        "code_switch_observed_yes_no", "notes",
        "llm_judge_faithfulness_1to5", "llm_judge_groundedness_1to5",
        "llm_judge_hallucination_detected", "llm_judge_justification",
    ]
    provider_models = run.get("provider_models") or {}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in run["items"]:
            for provider_key, res in item.get("results", {}).items():
                for cond_key, cond_label in [
                    ("condition_A_english_prompt", "A_english_system_prompt"),
                    ("condition_B_yoruba_prompt", "B_yoruba_system_prompt"),
                ]:
                    cond = res[cond_key]
                    judge = cond.get("llm_judge", {})
                    if cond["stop_reason"] == "error":
                        response_text = f"[ERROR: {cond.get('error', 'unknown')}]"
                    elif cond["stop_reason"] == "refusal":
                        response_text = "[REFUSED]"
                    else:
                        response_text = cond["text"]
                    writer.writerow(
                        {
                            "item_id": item["id"],
                            "category": item["category"],
                            "provider": provider_key,
                            "provider_model": provider_models.get(provider_key, ""),
                            "condition": cond_label,
                            "input": item["input"],
                            "reference": item.get("reference") or "",
                            "response": response_text,
                            "fluency_1to5": "",
                            "naturalness_1to5": "",
                            "task_accuracy_1to5": "",
                            "faithfulness_1to5": "",
                            "groundedness_1to5": "",
                            "code_switch_observed_yes_no": "",
                            "notes": "",
                            "llm_judge_faithfulness_1to5": judge.get("faithfulness_score", ""),
                            "llm_judge_groundedness_1to5": judge.get("groundedness_score", ""),
                            "llm_judge_hallucination_detected": judge.get("hallucination_detected", ""),
                            "llm_judge_justification": judge.get("justification", ""),
                        }
                    )
    print(f"Human-rating CSV written to {out_path}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_run_path()
    with open(path, "r", encoding="utf-8") as f:
        run = json.load(f)

    summary = summarize(run)
    print_report(run, summary)

    base = os.path.splitext(os.path.basename(path))[0]
    summary_path = os.path.join(RESULTS_DIR, f"summary_{base}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary JSON written to {summary_path}")

    csv_path = os.path.join(RESULTS_DIR, f"human_rating_{base}.csv")
    export_human_rating_csv(run, csv_path)


if __name__ == "__main__":
    main()
