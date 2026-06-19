"""Compute paired Wilcoxon signed-rank tests for DPS approaches.

This script compares per-file metric scores (cosine similarity, BERTScore F1)
for each of the four LLM model comparisons:
  - NLG vs LLM (QWEN, GPT, CLAUDE, MISTRAL)
  - SWUM vs LLM (QWEN, GPT, CLAUDE, MISTRAL)

Results are appended to evaluation-results/results.txt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ALPHA = 0.05

MODELS = [
    ("QWEN", "llm (qwen)_vs_human_class_scores.csv"),
    ("GPT", "llm (gpt)_vs_human_class_scores.csv"),
    ("CLAUDE", "llm (claude)_vs_human_class_scores.csv"),
    ("MISTRAL", "llm (mistral)_vs_human_class_scores.csv"),
]


@dataclass
class TestResult:
    comparison: str
    metric: str
    n_pairs: int
    w_statistic: float
    p_value: float
    median_diff: float
    direction: str
    significant: bool
    note: str = ""


def normalize_text(value: object) -> str:
    """Normalize text keys to improve stable matching across files."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("\\", "/")
    return " ".join(text.split())


def normalize_filename(value: object) -> str:
    """Normalize filename by extracting basename and lowercasing."""
    text = normalize_text(value)
    if not text:
        return ""
    return Path(text).name


def build_pair_keys(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Build a stable per-file key used for pairwise merges.

    Returns (primary_key, fallback_key):
    - primary_key: base_project + project_path + filename (if project_path exists)
    - fallback_key: base_project + filename
    """
    required_common = {"base_project", "filename"}
    missing = required_common.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required key columns: {sorted(missing)}")

    base_project = df["base_project"].map(normalize_text)
    filename = df["filename"].map(normalize_filename)

    fallback_key = base_project + "::" + filename

    if "project_path" in df.columns:
        project_path = df["project_path"].map(normalize_text)
        primary_key = base_project + "::" + project_path + "::" + filename
    else:
        primary_key = fallback_key

    return primary_key, fallback_key


def validate_input_frame(name: str, df: pd.DataFrame) -> None:
    """Validate minimum columns and duplicate key conditions."""
    required_cols = {"cosine_similarity", "bert_f1", "base_project", "filename"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

    dup_count = int(df["pair_key"].duplicated().sum())
    if dup_count:
        sample = df.loc[df["pair_key"].duplicated(), "pair_key"].head(5).tolist()
        raise ValueError(
            f"{name}: found {dup_count} duplicate pair keys. Sample: {sample}"
        )


def prepare_frame(csv_path: Path, label: str) -> pd.DataFrame:
    """Load and normalize a class-score CSV for pairwise matching."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    pair_key, fallback_key = build_pair_keys(df)
    df["pair_key"] = pair_key
    df["fallback_key"] = fallback_key

    # Keep only columns needed for paired statistical testing.
    out = df[["pair_key", "fallback_key", "cosine_similarity", "bert_f1"]].copy()
    out.rename(
        columns={
            "cosine_similarity": f"cosine_similarity_{label}",
            "bert_f1": f"bert_f1_{label}",
        },
        inplace=True,
    )
    validate_input_frame(label, df)
    return out


def merge_pairwise(
    lhs: pd.DataFrame,
    rhs: pd.DataFrame,
    lhs_label: str,
    rhs_label: str,
) -> pd.DataFrame:
    """Two-stage pairwise merge to maximize safe coverage.

    Stage 1: strict merge on primary pair_key.
    Stage 2: for unmatched rows only, merge on fallback_key with deterministic
    occurrence index per fallback key.
    """
    strict = lhs.merge(rhs.drop(columns=["fallback_key"]), on="pair_key", how="inner")

    lhs_unmatched = lhs.loc[~lhs["pair_key"].isin(strict["pair_key"])].copy()
    rhs_unmatched = rhs.loc[~rhs["pair_key"].isin(strict["pair_key"])].copy()

    recovered = pd.DataFrame(columns=strict.columns)
    if not lhs_unmatched.empty and not rhs_unmatched.empty:
        lhs_unmatched["fallback_idx"] = lhs_unmatched.groupby("fallback_key").cumcount()
        rhs_unmatched["fallback_idx"] = rhs_unmatched.groupby("fallback_key").cumcount()

        recovered = lhs_unmatched.merge(
            rhs_unmatched.drop(columns=["pair_key"]),
            on=["fallback_key", "fallback_idx"],
            how="inner",
        )
        if not recovered.empty:
            recovered.drop(columns=["fallback_idx"], inplace=True)

    merged = pd.concat([strict, recovered], ignore_index=True)
    if merged.empty:
        raise ValueError(f"No matched pairs after merge: {lhs_label} vs {rhs_label}")

    merged.drop(columns=["fallback_key"], errors="ignore", inplace=True)
    return merged


def run_wilcoxon_test(
    merged: pd.DataFrame,
    metric: str,
    lhs_label: str,
    rhs_label: str,
) -> TestResult:
    """Run Wilcoxon signed-rank test for a single metric and pairwise comparison."""
    lhs_col = f"{metric}_{lhs_label}"
    rhs_col = f"{metric}_{rhs_label}"

    pair_df = merged[[lhs_col, rhs_col]].dropna()
    x = pair_df[lhs_col].to_numpy(dtype=float)
    y = pair_df[rhs_col].to_numpy(dtype=float)

    n_pairs = int(len(pair_df))
    if n_pairs == 0:
        raise ValueError(f"No valid non-NaN pairs for {lhs_label} vs {rhs_label} ({metric})")

    diffs = x - y
    median_diff = float(np.median(diffs))

    if np.allclose(diffs, 0.0):
        return TestResult(
            comparison=f"{lhs_label} vs {rhs_label}",
            metric=metric,
            n_pairs=n_pairs,
            w_statistic=0.0,
            p_value=1.0,
            median_diff=median_diff,
            direction="no difference",
            significant=False,
            note="All paired differences are zero.",
        )

    result = wilcoxon(x, y, alternative="two-sided")

    if median_diff > 0:
        direction = f"{lhs_label} tends higher"
    elif median_diff < 0:
        direction = f"{rhs_label} tends higher"
    else:
        direction = "no median difference"

    return TestResult(
        comparison=f"{lhs_label} vs {rhs_label}",
        metric=metric,
        n_pairs=n_pairs,
        w_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        median_diff=median_diff,
        direction=direction,
        significant=bool(result.pvalue < ALPHA),
    )


def format_results_section(results: Sequence[TestResult], matched_info: Dict[str, int]) -> str:
    """Format an append-ready text block for evaluation-results/results.txt."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("\n\n" + "=" * 60)
    lines.append("WILCOXON SIGNED-RANK TESTS (PAIRED)")
    lines.append("=" * 60)
    lines.append(f"Generated at: {ts}")
    lines.append("Dataset: paired per-file class scores")
    lines.append(f"Significance level (alpha): {ALPHA:.2f}")
    lines.append("Tests: two-sided; metrics = cosine_similarity, bert_f1")
    lines.append("")
    lines.append("Matched pairs before NaN filtering:")
    for comp, n in matched_info.items():
        lines.append(f"  - {comp}: {n}")

    lines.append("")
    lines.append(
        "comparison           metric              n_pairs   W_stat        p_value       median_diff   interpretation"
    )
    lines.append("-" * 118)

    for r in results:
        sig_label = "significant" if r.significant else "not-significant"
        interpretation = f"{sig_label}; {r.direction}"
        row = (
            f"{r.comparison:<20} "
            f"{r.metric:<18} "
            f"{r.n_pairs:>7} "
            f"{r.w_statistic:>10.4f} "
            f"{r.p_value:>13.6g} "
            f"{r.median_diff:>12.6f} "
            f"{interpretation}"
        )
        lines.append(row)
        if r.note:
            lines.append(f"  note: {r.note}")

    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    eval_dir = root / "evaluation-results"

    nlg_csv = eval_dir / "nlg_vs_human_class_scores.csv"
    swum_csv = eval_dir / "swum_vs_human_class_scores.csv"
    results_txt = eval_dir / "results.txt"

    nlg = prepare_frame(nlg_csv, "NLG")
    swum = prepare_frame(swum_csv, "SWUM")

    matched_info: Dict[str, int] = {}
    test_results: List[TestResult] = []

    for model_name, model_csv_name in MODELS:
        llm = prepare_frame(eval_dir / model_csv_name, model_name)

        nlg_vs_llm = merge_pairwise(nlg, llm, "NLG", model_name)
        swum_vs_llm = merge_pairwise(swum, llm, "SWUM", model_name)

        matched_info[f"NLG vs {model_name}"] = int(len(nlg_vs_llm))
        matched_info[f"SWUM vs {model_name}"] = int(len(swum_vs_llm))

        for metric in ("cosine_similarity", "bert_f1"):
            test_results.append(run_wilcoxon_test(nlg_vs_llm, metric, "NLG", model_name))
            test_results.append(run_wilcoxon_test(swum_vs_llm, metric, "SWUM", model_name))

    section = format_results_section(test_results, matched_info)
    with results_txt.open("a", encoding="utf-8") as fh:
        fh.write(section)

    print("Wilcoxon signed-rank tests completed.")
    print(f"Appended results to: {results_txt}")
    for result in test_results:
        print(
            f"- {result.comparison} | {result.metric} | n={result.n_pairs} | "
            f"W={result.w_statistic:.4f} | p={result.p_value:.6g}"
        )


if __name__ == "__main__":
    main()
