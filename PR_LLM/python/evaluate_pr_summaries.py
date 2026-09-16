"""evaluate_pr_summaries.py

Compute BERTScore and TF-IDF cosine similarity for PR summaries against the
human gold descriptions.

Workflow
--------
1. Load the gold summaries from ``PR_LLM/input/human_summaries/``.
2. Load every system under comparison — the deterministic tool baseline from
   ``PR_LLM/input/tool_summaries/`` and each LLM's JSONL from ``PR_LLM/output/``.
3. Batch-compute BERTScore (roberta-large, one call per system) and TF-IDF
   cosine similarity (sklearn) for every matched pair.
4. Write per-item scores to ``evaluation-results/<SYSTEM>_vs_human_scores.jsonl``
   and dataset-level averages to ``evaluation-results/overall_comparison.jsonl``.
5. Merge the aggregates into ``output/results.json``.

Mirrors DPS_LLM's evaluate_summaries.py: the deterministic baselines (NLG,
SWUM there; the PR tool here) are scored against the human gold on exactly the
same footing as the LLMs, so one table ranks every system.

- BERTScore uses the functional API with ``lang='en'`` (roberta-large), batching
  all pairs for a system into one forward pass so the model loads once.
- Cosine similarity uses sklearn TfidfVectorizer + cosine_similarity; no model
  download or load required.

Usage:
    python evaluate_pr_summaries.py [SYSTEM ...] [options]

    positional:
      SYSTEM              One or more of: TOOL, GPT, QWEN, CLAUDE, MISTRAL, ALL
                          Default: ALL

    optional:
      --output DIR        Directory containing the LLM JSONL files
                          Default: <PR_LLM>/output/
      --results-dir DIR   Directory for per-item and aggregate score files
                          Default: <PR_LLM>/evaluation-results/
      --limit N           Only evaluate the first N entries per system
      --bertscore-lang S  Language for BERTScore ('en' selects roberta-large)
                          Default: en
      --log-level LEVEL   DEBUG, INFO, WARNING, ERROR  (default: INFO)

:author: Najam Nazar
:version: 2.0.0
:date: 2026-09-16
:license: MIT
"""

from __future__ import annotations

__author__ = "Najam Nazar"
__version__ = "2.0.0"
__date__ = "2026-09-16"
__license__ = "MIT"

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Silence HuggingFace/Transformers output before the heavy libraries are imported.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

# bert_score functional API — loads roberta-large on first call, batches all
# pairs in one forward pass, matching DPS_LLM's MetricsCalculator.bert_scores.
from bert_score import score as bert_score
from dotenv import load_dotenv
# TF-IDF cosine — matches DPS_LLM's MetricsCalculator.cosine_similarity exactly;
# no model download required.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

import pr_corpus
from pr_corpus import (
    DISPLAY_NAMES,
    MODEL_ALIASES,
    OUTPUT_DIR,
    RESULTS_DIR,
    REPO_ROOT,
    TOOL_LABEL,
    system_order,
    write_jsonl,
)

# Configure logging before any library installs its own handler.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logging.getLogger(__name__).addHandler(_console)
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger(__name__).propagate = False
_log = logging.getLogger(__name__)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    import transformers as _tf
    _tf.logging.set_verbosity_error()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two strings (DPS_LLM MetricsCalculator port)."""
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
        return float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0, 0])
    except ValueError:
        # Empty vocabulary — one or both strings are blank or all stop-words.
        return 0.0


# ---------------------------------------------------------------------------
# Per-system evaluation
# ---------------------------------------------------------------------------

def evaluate_system(
    system: str,
    summaries: Dict[str, str],
    gold: Dict[str, str],
    results_dir: Path,
    bert_lang: str,
    device: Optional[str],
) -> Tuple[Dict[str, object], List[dict]]:
    """Score one system against the gold summaries and persist per-item results.

    Parameters
    ----------
    system:      System name (``TOOL``, ``GPT``, ...).
    summaries:   pr_id -> generated summary for this system.
    gold:        pr_id -> human reference summary.
    results_dir: Directory to write ``<SYSTEM>_vs_human_scores.jsonl`` into.
    bert_lang:   Language code for BERTScore (``'en'`` selects roberta-large).
    device:      Torch device (``None`` = auto-detect).

    Returns
    -------
    (stats, rows) where *stats* holds the dataset-level averages and *rows*
    holds the per-item score records.
    """
    display = DISPLAY_NAMES.get(system, system)

    # Only items the system actually produced a summary for are scored, so a
    # missing generation is excluded rather than counted as an empty string.
    pr_ids = sorted(pid for pid, text in summaries.items() if text and gold.get(pid))
    missing = sorted(set(gold) - set(pr_ids))
    print(f"\n[{display}] {len(pr_ids)} matched items ({len(missing)} missing vs gold).", flush=True)
    if not pr_ids:
        _log.warning("[%s] No matched items — skipping.", display)
        return {}, []

    candidates = [summaries[pid] for pid in pr_ids]
    references = [gold[pid] for pid in pr_ids]

    print(f"[{display}] Batch BERTScore — {len(candidates)} pairs ...", flush=True)
    P, R, F = bert_score(
        candidates,
        references,
        lang=bert_lang,
        rescale_with_baseline=False,
        verbose=False,
        device=device,
    )

    rows: List[dict] = []
    for idx, pr_id in enumerate(pr_ids):
        rows.append({
            "pr_id": pr_id,
            "system": system,
            "summary": candidates[idx],
            "human_summary": references[idx],
            "metrics": {
                "bert_precision": float(P[idx].item()),
                "bert_recall": float(R[idx].item()),
                "bert_f1": float(F[idx].item()),
                "cosine_similarity": _tfidf_cosine(candidates[idx], references[idx]),
            },
        })

    scores_path = results_dir / f"{system}_vs_human_scores.jsonl"
    write_jsonl(scores_path, rows)
    print(f"[{display}] Per-item scores written to {scores_path.name}", flush=True)

    def _avg(key: str) -> float:
        return float(sum(r["metrics"][key] for r in rows) / len(rows))

    stats: Dict[str, object] = {
        "system": system,
        "count": len(rows),
        "missing_vs_gold": len(missing),
        "avg_bert_precision": _avg("bert_precision"),
        "avg_bert_recall": _avg("bert_recall"),
        "avg_bert_f1": _avg("bert_f1"),
        "avg_cosine_similarity": _avg("cosine_similarity"),
    }

    print(
        f"[{display}] BERT P={stats['avg_bert_precision']:.4f}  "
        f"R={stats['avg_bert_recall']:.4f}  F1={stats['avg_bert_f1']:.4f}  "
        f"cos={stats['avg_cosine_similarity']:.4f}  (n={stats['count']})",
        flush=True,
    )
    return stats, rows


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

# The tool baseline gets its own colour so it reads as a distinct system class
# in every figure; the LLM colours match DPS_LLM's palette.
_VIOLIN_COLORS: Dict[str, str] = {
    "Tool":    "#7f8c8d",
    "GPT":     "#d35400",
    "Qwen":    "#27ae60",
    "Claude":  "#16a085",
    "Mistral": "#2c3e50",
}


def _add_mean_markers(ax: plt.Axes, data: pd.DataFrame, order: List[str], metric: str) -> None:
    """Overlay a red diamond at the mean on each violin body."""
    for idx, model in enumerate(order):
        mean_val = data.loc[data["system"] == model, metric].mean()
        ax.plot(
            idx,
            mean_val,
            marker="D",
            markersize=8,
            color="darkred",
            zorder=3,
            label="Mean" if idx == 0 else "",
        )
    ax.legend(loc="upper left")


def plot_violin_scores(rows_by_system: Dict[str, List[dict]], results_dir: Path) -> None:
    """Draw a 1x2 violin plot of cosine similarity and BERTScore F1 per system.

    Mirrors DPS_LLM's VisualizationManager: two side-by-side violin panels with
    a mean-value diamond marker overlaid on each violin body.
    """
    rows: List[dict] = []
    order: List[str] = []

    for system, system_rows in rows_by_system.items():
        display = DISPLAY_NAMES.get(system, system)
        if not system_rows:
            continue
        if display not in order:
            order.append(display)
        for rec in system_rows:
            metrics = rec["metrics"]
            rows.append({
                "system": display,
                "bert_f1": metrics["bert_f1"],
                "cosine_similarity": metrics["cosine_similarity"],
            })

    if not rows:
        print("[violin] No metric data found — skipping violin plot.", flush=True)
        return

    data = pd.DataFrame(rows)
    palette = {lbl: _VIOLIN_COLORS[lbl] for lbl in order if lbl in _VIOLIN_COLORS}

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(order) * 2.5), 6))
    fig.suptitle("PR Summary Evaluation: Score Distributions", fontsize=14, fontweight="bold")

    for ax, metric, ylabel, panel_title in (
        (axes[0], "cosine_similarity", "Cosine Similarity Score", "Cosine Similarity"),
        (axes[1], "bert_f1",           "BERTScore F1 Score",      "BERTScore F1"),
    ):
        sns.violinplot(
            data=data,
            x="system",
            y=metric,
            ax=ax,
            order=order,
            palette=palette or None,
            hue="system",
            legend=False,
        )
        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("System", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        _add_mean_markers(ax, data, order, metric)

    plt.tight_layout()
    out_path = results_dir / "PR_violin_scores.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[violin] Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    valid_systems = system_order()

    parser = argparse.ArgumentParser(
        description=(
            "Compute BERTScore and TF-IDF cosine metrics for the tool baseline "
            "and each LLM against the human gold PR descriptions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "systems",
        metavar="SYSTEM",
        nargs="*",
        default=["ALL"],
        help=f"One or more of: {', '.join(valid_systems)}, ALL. Default: ALL",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Directory containing the LLM JSONL files. Default: <PR_LLM>/output/",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        help="Directory for score files. Default: <PR_LLM>/evaluation-results/",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N entries per system (useful for testing).",
    )
    parser.add_argument(
        "--bertscore-lang", default="en",
        help="Language for BERTScore — 'en' selects roberta-large (default: en).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).",
    )
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    _log.setLevel(level)
    _log.handlers[0].setLevel(level)

    load_dotenv(REPO_ROOT / ".env")
    output_dir = args.output or OUTPUT_DIR
    results_dir = args.results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    selected = [s.upper() for s in args.systems]
    if selected == ["ALL"]:
        systems = valid_systems
    else:
        unknown = [s for s in selected if s not in valid_systems]
        if unknown:
            raise SystemExit(
                f"Unknown system(s): {unknown}. Valid choices: {valid_systems}"
            )
        systems = selected

    gold = pr_corpus.load_human_summaries()
    print(f"[gold] {len(gold)} human summaries loaded from {pr_corpus.HUMAN_DIR}", flush=True)

    aggregates: Dict[str, dict] = {}
    rows_by_system: Dict[str, List[dict]] = {}

    for system in systems:
        try:
            summaries = pr_corpus.load_system_summaries(system, output_dir, args.limit)
        except FileNotFoundError as exc:
            _log.warning("[%s] %s — skipping", system, exc)
            continue

        stats, rows = evaluate_system(
            system=system,
            summaries=summaries,
            gold=gold,
            results_dir=results_dir,
            bert_lang=args.bertscore_lang,
            device=None,
        )
        if stats:
            aggregates[system] = stats
            rows_by_system[system] = rows

    if not aggregates:
        _log.error("No systems could be evaluated. Checked: %s", systems)
        raise SystemExit(1)

    # Dataset-level comparison table, ordered best-to-worst on BERTScore F1.
    overall = sorted(aggregates.values(), key=lambda s: s["avg_bert_f1"], reverse=True)
    overall_path = results_dir / "overall_comparison.jsonl"
    write_jsonl(overall_path, overall)
    print(f"\n[Done] Overall comparison written to: {overall_path}", flush=True)

    # Merge into results.json so rubric_evaluation data from the ranking step
    # is preserved alongside the automated metrics.
    results_path = output_dir / "results.json"
    existing: dict = {}
    if results_path.exists():
        try:
            with results_path.open(encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            _log.warning("Could not parse %s — starting fresh.", results_path)
    for system, stats in aggregates.items():
        existing.setdefault(system, {}).update(stats)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    print(f"[Done] Aggregated results merged into: {results_path}", flush=True)

    print("\n[violin] Generating violin plot ...", flush=True)
    plot_violin_scores(rows_by_system, results_dir)

    print("Next: run rank_pr_summaries.py for rubric-based LLM ranking.", flush=True)


if __name__ == "__main__":
    main()
