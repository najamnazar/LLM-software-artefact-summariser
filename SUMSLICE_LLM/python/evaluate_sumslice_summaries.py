"""evaluate_sumslice_summaries.py

Compare SUMSLICE LLM-generated method summaries against human ground-truth
summaries using BERTScore and TF-IDF cosine similarity, then produce violin
plot visualisations.

Workflow
--------
1. Load per-model JSON summary files from ``output/``.
2. Load per-project ground-truth JSON files from ``input/ground-truth/``.
3. Match pairs by (project, methodName, className).
4. Batch-compute BERTScore (roberta-large) and TF-IDF cosine similarity.
5. Write per-method CSV, per-project CSV, and an overall comparison CSV to
   ``evaluation_results/``.
6. Save a 1×2 violin plot (cosine similarity | BERTScore F1) to
   ``evaluation_results/violin_scores.png``.

Mirrors the evaluation approach used in DPS_LLM and PR_LLM:
- BERTScore uses the functional API with ``lang='en'`` (roberta-large), batching
  all pairs for a model into one forward pass.
- Cosine similarity uses sklearn TfidfVectorizer fitted fresh per pair.

Usage
-----
    python evaluate_sumslice_summaries.py [options]

    optional:
      --output-dir DIR      Directory containing SUMSLICE_*_SUMMARY.json files
                            Default: <SUMSLICE_LLM>/output/
      --gt-dir DIR          Directory containing ground-truth JSON files
                            Default: <SUMSLICE_LLM>/input/ground-truth/
      --results-dir DIR     Directory for output CSVs and plots
                            Default: <SUMSLICE_LLM>/evaluation_results/
      --models MODEL [...]  Subset of: CLAUDE GPT MISTRAL QWEN  (default: all)
      --bertscore-lang S    BERTScore language code  (default: en)

:author: Najam Nazar
:version: 1.0.0
:date: 2026-07-01
:license: MIT
"""

from __future__ import annotations

__author__ = "Najam Nazar"
__version__ = "1.0.0"
__date__ = "2026-07-01"
__license__ = "MIT"

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from bert_score import score as bert_score_fn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

try:
    import transformers as _tf
    _tf.logging.set_verbosity_error()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMSLICE_ROOT = Path(__file__).resolve().parent.parent

_MODEL_FILES: Dict[str, str] = {
    "CLAUDE":  "SUMSLICE_CLAUDE_SUMMARY.json",
    "GPT":     "SUMSLICE_GPT_SUMMARY.json",
    "MISTRAL": "SUMSLICE_MISTRAL_SUMMARY.json",
    "QWEN":    "SUMSLICE_QWEN_SUMMARY.json",
}

_MODEL_DISPLAY: Dict[str, str] = {
    "CLAUDE":  "Claude",
    "GPT":     "GPT",
    "MISTRAL": "Mistral",
    "QWEN":    "Qwen",
}

# Ground-truth file names keyed by the project name used in LLM output.
_PROJECT_GT_FILES: Dict[str, str] = {
    "jajuk":        "jajuk-example-summaries.json",
    "jEdit":        "jedit-example-summaries.json",
    "jhotdraw":     "jhotdraw-example-summaries.json",
    "jtopas":       "jtopas-example-summaries.json",
    "nanoxml":      "nanoXML-example-summaries.json",
    "siena-master": "siena-example-summaries.json",
}

# Shared colour palette — matches DPS_LLM / PR_LLM for visual consistency.
_VIOLIN_COLORS: Dict[str, str] = {
    "Claude":  "#16a085",
    "GPT":     "#d35400",
    "Mistral": "#2c3e50",
    "Qwen":    "#27ae60",
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two strings."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
        return float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0, 0])
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_llm_summaries(json_path: Path) -> List[dict]:
    """Return the ``summaries`` list from a SUMSLICE_*_SUMMARY.json file."""
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["summaries"]


def _load_ground_truth(gt_dir: Path) -> Dict[Tuple[str, str, str], str]:
    """Build a (project, methodName, className) -> summary lookup from all GT files."""
    lookup: Dict[Tuple[str, str, str], str] = {}
    for llm_project, gt_filename in _PROJECT_GT_FILES.items():
        gt_path = gt_dir / gt_filename
        if not gt_path.exists():
            print(f"  WARNING: Ground-truth file not found: {gt_path}")
            continue
        with gt_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data["summaries"]:
            key = (llm_project, entry["methodName"], entry["className"])
            lookup[key] = entry["summary"]
    return lookup


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    short_name: str,
    llm_summaries: List[dict],
    gt_lookup: Dict[Tuple[str, str, str], str],
    bert_lang: str,
    results_dir: Path,
) -> Optional[Dict]:
    """Compute metrics for one model and write per-method + per-project CSVs.

    Returns a dict of corpus-level aggregate stats, or None if no pairs matched.
    """
    display = _MODEL_DISPLAY[short_name]
    print(f"\n{'='*60}")
    print(f"Evaluating {display} vs Ground Truth")
    print(f"{'='*60}")

    # Build a (project, name, class) -> [summary, ...] map preserving LLM order,
    # so that first and second occurrences can be retrieved separately.
    key_occurrences: Dict[Tuple[str, str, str], List[str]] = {}
    for entry in llm_summaries:
        key = (entry["project"], entry["name"], entry["class"])
        if key not in gt_lookup:
            continue
        if key not in key_occurrences:
            key_occurrences[key] = []
        key_occurrences[key].append(entry["summary"])

    # First pass: one pair per unique (project, name, class) key.
    key_seen: set = set()
    pairs_by_project: Dict[str, List[dict]] = {}
    for entry in llm_summaries:
        key = (entry["project"], entry["name"], entry["class"])
        if key not in gt_lookup or key in key_seen:
            continue
        key_seen.add(key)
        proj = entry["project"]
        if proj not in pairs_by_project:
            pairs_by_project[proj] = []
        pairs_by_project[proj].append({
            "project":     proj,
            "method_name": entry["name"],
            "class_name":  entry["class"],
            "gt_summary":  gt_lookup[key],
            "llm_summary": entry["summary"],
        })

    # Second pass: fill each project to TARGET_PER_PROJECT (25) using the second
    # occurrence of duplicate keys.  12 extras across 4 projects bring the total
    # from 138 to 150 (25 x 6 = the intended corpus size).
    TARGET_PER_PROJECT = 25
    dup_keys_by_project: Dict[str, List[Tuple]] = {}
    for key, summaries in key_occurrences.items():
        if len(summaries) >= 2:
            proj = key[0]
            if proj not in dup_keys_by_project:
                dup_keys_by_project[proj] = []
            dup_keys_by_project[proj].append(key)

    extras_added = 0
    for proj, dup_keys in dup_keys_by_project.items():
        needed = TARGET_PER_PROJECT - len(pairs_by_project.get(proj, []))
        if needed <= 0:
            continue
        for key in dup_keys[:needed]:
            pairs_by_project[proj].append({
                "project":     proj,
                "method_name": key[1],
                "class_name":  key[2],
                "gt_summary":  gt_lookup[key],
                "llm_summary": key_occurrences[key][1],  # second LLM occurrence
            })
            extras_added += 1

    records: List[dict] = [r for proj_list in pairs_by_project.values() for r in proj_list]

    print(f"  LLM summaries loaded: {len(llm_summaries)}")
    print(f"  Unique matched pairs: {len(key_seen)}")
    print(f"  Extra pairs added (duplicates, to reach 25/project): {extras_added}")
    print(f"  Total pairs for evaluation: {len(records)}")

    if not records:
        print(f"  ERROR: No matching entries found for {display}")
        return None

    df = pd.DataFrame(records)

    # Cosine similarity (per pair, no model weights).
    df["cosine_similarity"] = df.apply(
        lambda row: _tfidf_cosine(row["llm_summary"], row["gt_summary"]),
        axis=1,
    )

    # BERTScore — batch all pairs in one forward pass.
    candidates = df["llm_summary"].tolist()
    references = df["gt_summary"].tolist()
    print(f"  Computing BERTScore for {len(candidates)} pairs (may take 2-3 min)...")
    P, R, F = bert_score_fn(
        candidates,
        references,
        lang=bert_lang,
        rescale_with_baseline=False,
        verbose=False,
    )
    df["bert_precision"] = P.cpu().numpy()
    df["bert_recall"]    = R.cpu().numpy()
    df["bert_f1"]        = F.cpu().numpy()
    print("  BERTScore complete.")

    # --- per-method CSV ---
    method_csv = results_dir / f"{short_name.lower()}_vs_gt_method_scores.csv"
    df.to_csv(method_csv, index=False)
    print(f"  Saved: {method_csv}")

    # --- per-project CSV ---
    project_stats = (
        df.groupby("project")
        .agg(
            methods=("method_name", "count"),
            avg_cosine=("cosine_similarity", "mean"),
            avg_bert_precision=("bert_precision", "mean"),
            avg_bert_recall=("bert_recall", "mean"),
            avg_bert_f1=("bert_f1", "mean"),
        )
        .reset_index()
        .sort_values("avg_bert_f1", ascending=False)
    )
    project_csv = results_dir / f"{short_name.lower()}_vs_gt_project_scores.csv"
    project_stats.to_csv(project_csv, index=False)
    print(f"  Saved: {project_csv}")

    avg_cosine = float(df["cosine_similarity"].mean())
    avg_bp     = float(df["bert_precision"].mean())
    avg_br     = float(df["bert_recall"].mean())
    avg_bf1    = float(df["bert_f1"].mean())
    std_cosine = float(df["cosine_similarity"].std(ddof=0))
    std_bf1    = float(df["bert_f1"].std(ddof=0))

    print(f"\n  {display} Results:")
    print(f"    Methods evaluated:  {len(df)}")
    print(f"    Avg Cosine:         {avg_cosine:.4f} (±{std_cosine:.4f})")
    print(f"    Avg BERT Precision: {avg_bp:.4f}")
    print(f"    Avg BERT Recall:    {avg_br:.4f}")
    print(f"    Avg BERT F1:        {avg_bf1:.4f} (±{std_bf1:.4f})")

    return {
        "model":              display,
        "methods_evaluated":  len(df),
        "avg_cosine":         avg_cosine,
        "cosine_std":         std_cosine,
        "avg_bert_precision": avg_bp,
        "avg_bert_recall":    avg_br,
        "avg_bert_f1":        avg_bf1,
        "bert_f1_std":        std_bf1,
        "_df":                df,   # kept in-memory for violin plot; stripped before CSV
    }


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def _add_mean_markers(ax: plt.Axes, data: pd.DataFrame, order: List[str], metric: str) -> None:
    for idx, model in enumerate(order):
        mean_val = data.loc[data["model"] == model, metric].mean()
        ax.plot(
            idx, mean_val,
            marker="D", markersize=8, color="darkred", zorder=3,
            label="Mean" if idx == 0 else "",
        )
    ax.legend(loc="upper left")


def plot_violin(results: List[dict], results_dir: Path) -> None:
    """Draw 1×2 violin plot (cosine similarity | BERTScore F1) and save PNG."""
    rows: List[dict] = []
    order: List[str] = []
    for stats in results:
        df = stats["_df"]
        display = stats["model"]
        if display not in order:
            order.append(display)
        for _, row in df.iterrows():
            rows.append({
                "model":            display,
                "cosine_similarity": row["cosine_similarity"],
                "bert_f1":           row["bert_f1"],
            })

    if not rows:
        print("[violin] No data — skipping.")
        return

    data = pd.DataFrame(rows)
    palette = {lbl: _VIOLIN_COLORS[lbl] for lbl in order if lbl in _VIOLIN_COLORS}

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(order) * 2.5), 6))
    fig.suptitle("SUMSLICE LLM Evaluation: Score Distributions vs Ground Truth",
                 fontsize=14, fontweight="bold")

    for ax, metric, ylabel, panel_title in (
        (axes[0], "cosine_similarity", "Cosine Similarity Score", "Cosine Similarity"),
        (axes[1], "bert_f1",           "BERTScore F1 Score",      "BERTScore F1"),
    ):
        sns.violinplot(
            data=data, x="model", y=metric,
            ax=ax, order=order,
            palette=palette or None,
            hue="model", legend=False,
        )
        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        _add_mean_markers(ax, data, order, metric)

    plt.tight_layout()
    out_path = results_dir / "violin_scores.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[violin] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary files
# ---------------------------------------------------------------------------

def _write_summary_files(results: List[dict], results_dir: Path) -> None:
    """Write evaluation_summary.txt and append to results.txt."""
    summary_path = results_dir / "evaluation_summary.txt"
    results_path = results_dir / "results.txt"

    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("EVALUATION SUMMARY: SUMSLICE LLM vs Ground Truth\n")
        fh.write("=" * 60 + "\n\n")
        for stats in results:
            fh.write(f"\n{stats['model']}:\n")
            fh.write(f"  Methods Evaluated: {stats['methods_evaluated']}\n")
            fh.write(f"  Avg Cosine:        {stats['avg_cosine']:.4f} (±{stats['cosine_std']:.4f})\n")
            fh.write(f"  Avg BERT Precision:{stats['avg_bert_precision']:.4f}\n")
            fh.write(f"  Avg BERT Recall:   {stats['avg_bert_recall']:.4f}\n")
            fh.write(f"  Avg BERT F1:       {stats['avg_bert_f1']:.4f} (±{stats['bert_f1_std']:.4f})\n")
    print(f"Saved: {summary_path}")

    with results_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 60 + "\n")
        fh.write(f"SUMSLICE LLM EVALUATION — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write("=" * 60 + "\n")
        for stats in results:
            fh.write(f"\n{stats['model']}\n")
            fh.write(f"  Methods Evaluated:  {stats['methods_evaluated']}\n")
            fh.write(f"  Avg Cosine:         {stats['avg_cosine']:.4f}\n")
            fh.write(f"  Avg BERT Precision: {stats['avg_bert_precision']:.4f}\n")
            fh.write(f"  Avg BERT Recall:    {stats['avg_bert_recall']:.4f}\n")
            fh.write(f"  Avg BERT F1:        {stats['avg_bert_f1']:.4f}\n")
    print(f"Appended to: {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SUMSLICE LLM summaries against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=SUMSLICE_ROOT / "output",
        help="Directory containing SUMSLICE_*_SUMMARY.json files.",
    )
    parser.add_argument(
        "--gt-dir", type=Path,
        default=SUMSLICE_ROOT / "input" / "ground-truth",
        help="Directory containing ground-truth JSON files.",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=SUMSLICE_ROOT / "evaluation_results",
        help="Output directory for CSVs and plots.",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(_MODEL_FILES.keys()),
        choices=list(_MODEL_FILES.keys()),
        metavar="MODEL",
        help="Models to evaluate: CLAUDE GPT MISTRAL QWEN (default: all).",
    )
    parser.add_argument(
        "--bertscore-lang", default="en",
        help="BERTScore language — 'en' selects roberta-large (default: en).",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("SUMSLICE LLM Evaluation vs Ground Truth")
    print("=" * 60)
    print(f"Output dir:   {args.output_dir}")
    print(f"GT dir:       {args.gt_dir}")
    print(f"Results dir:  {args.results_dir}")
    print(f"Models:       {args.models}")

    print("\nLoading ground-truth summaries...")
    gt_lookup = _load_ground_truth(args.gt_dir)
    print(f"  Ground-truth entries: {len(gt_lookup)}")

    all_results: List[dict] = []

    for short_name in args.models:
        json_path = args.output_dir / _MODEL_FILES[short_name]
        if not json_path.exists():
            print(f"\nWARNING: {json_path} not found — skipping {short_name}")
            continue

        llm_summaries = _load_llm_summaries(json_path)
        stats = evaluate_model(
            short_name=short_name,
            llm_summaries=llm_summaries,
            gt_lookup=gt_lookup,
            bert_lang=args.bertscore_lang,
            results_dir=args.results_dir,
        )
        if stats is not None:
            all_results.append(stats)

    if not all_results:
        print("\nNo results generated.")
        return

    # Strip internal _df before writing the overall comparison CSV.
    comparison_rows = [{k: v for k, v in s.items() if k != "_df"} for s in all_results]
    comparison_df = pd.DataFrame(comparison_rows)
    overall_csv = args.results_dir / "overall_comparison.csv"
    comparison_df.to_csv(overall_csv, index=False)
    print(f"\nSaved: {overall_csv}")

    _write_summary_files(all_results, args.results_dir)

    print("\nGenerating violin plot...")
    plot_violin(all_results, args.results_dir)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.results_dir}")
    print("Generated files:")
    print("  overall_comparison.csv")
    print("  evaluation_summary.txt")
    print("  results.txt")
    print("  violin_scores.png")
    for short_name in args.models:
        n = short_name.lower()
        print(f"  {n}_vs_gt_method_scores.csv")
        print(f"  {n}_vs_gt_project_scores.csv")


if __name__ == "__main__":
    main()
