"""
Run evaluate_summaries.py for B1..B5 against human summaries and aggregate results.

Outputs:
- evaluation-results/b_iterations_pairwise.csv (concatenated per-class metrics with iteration tag)
- evaluation-results/b_iterations_summary.txt (per-iteration means and overall averages for BERTScore and Cosine Similarity)

Assumptions:
- evaluate_summaries.py supports CLI args: --human-csv, --nlg-csv, --output-dir
- It writes per-class metrics CSV containing columns including 'bert_score' and 'cosine_similarity'
  and may write a summary CSV; we rely on the per-class metrics for aggregation.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "python" / "evaluate_summaries.py"
HUMAN_CSV = REPO_ROOT / "input" / "human_summaries" / "DPS_Human_Summaries.csv"
OUTPUT_DIR = REPO_ROOT / "evaluation-results"
SUMMARY_OUT = OUTPUT_DIR / "b_iterations_summary.txt"
PAIRWISE_OUT = OUTPUT_DIR / "b_iterations_pairwise.csv"

B_FILES = [
    REPO_ROOT / "output" / "summary-output" / f"B{i}.csv" for i in range(1, 6)
]

@dataclass
class IterationResult:
    iteration: int
    pairwise_csv: Path


def run_iteration(iter_idx: int, nlg_csv: Path) -> IterationResult:
    iter_dir = OUTPUT_DIR / f"iter_{iter_idx}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(EVAL_SCRIPT),
        "--human-csv",
        str(HUMAN_CSV),
        "--nlg-csv",
        str(nlg_csv),
        "--output-dir",
        str(iter_dir),
    ]
    print(f"Running iteration {iter_idx} with {nlg_csv} ...")
    subprocess.run(cmd, check=True)
    # Find the per-class metrics CSV in iter_dir; prefer files containing 'metrics' or 'pairwise'
    candidates = list(iter_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV output found in {iter_dir}")
    # Heuristic: choose the largest CSV as pairwise metrics
    pairwise = max(candidates, key=lambda p: p.stat().st_size)
    return IterationResult(iteration=iter_idx, pairwise_csv=pairwise)


def aggregate_results(results: List[IterationResult]) -> Tuple[pd.DataFrame, str]:
    frames: List[pd.DataFrame] = []
    summaries: List[str] = []
    for res in results:
        df = pd.read_csv(res.pairwise_csv)
        df["iteration"] = res.iteration
        frames.append(df)
        # Handle potential column naming variations
        bert_cols = [c for c in df.columns if c.lower().startswith("bert")]
        cos_cols = [c for c in df.columns if "cosine" in c.lower()]
        if not bert_cols or not cos_cols:
            summaries.append(f"Iteration {res.iteration}: missing bert/cosine columns in {res.pairwise_csv.name}")
            continue
        bert_mean = float(df[bert_cols[0]].astype(float).mean())
        cos_mean = float(df[cos_cols[0]].astype(float).mean())
        summaries.append(
            f"Iteration {res.iteration}: mean BERT={bert_mean:.4f}, mean Cosine={cos_mean:.4f}"
        )
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Overall averages
    overall_text = ""
    if not combined.empty:
        bert_cols_all = [c for c in combined.columns if c.lower().startswith("bert")] 
        cos_cols_all = [c for c in combined.columns if "cosine" in c.lower()] 
        if bert_cols_all and cos_cols_all:
            overall_bert = float(combined[bert_cols_all[0]].astype(float).mean())
            overall_cos = float(combined[cos_cols_all[0]].astype(float).mean())
            overall_text = (
                f"\nOverall averages across 5 iterations:\n"
                f"  BERTScore mean: {overall_bert:.4f}\n"
                f"  Cosine similarity mean: {overall_cos:.4f}\n"
            )
        else:
            overall_text = "\nOverall averages unavailable: missing columns across outputs.\n"
    summary_text = (
        "B Iterations: Human vs Bi (i=1..5)\n"
        + "\n".join(summaries)
        + overall_text
    )
    return combined, summary_text


def write_outputs(combined: pd.DataFrame, summary_text: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PAIRWISE_OUT, index=False)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Wrote combined pairwise to: {PAIRWISE_OUT}")
    print(f"Wrote summary to: {SUMMARY_OUT}")


def main() -> None:
    missing = [p for p in B_FILES if not p.exists()]
    if missing:
        print("Missing Bi files:", ", ".join(str(m) for m in missing))
        sys.exit(1)
    results: List[IterationResult] = []
    for i, nlg in enumerate(B_FILES, start=1):
        results.append(run_iteration(i, nlg))
    combined, summary_text = aggregate_results(results)
    write_outputs(combined, summary_text)


if __name__ == "__main__":
    main()
