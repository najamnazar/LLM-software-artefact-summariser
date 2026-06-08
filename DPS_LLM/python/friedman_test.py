"""Friedman test comparing DPS_NLG, DPS_LLM, and DPS_SWUM across 5 criteria.

Two-level analysis:
  PART 1 — Five per-criterion Friedman tests.
            Each test uses n ≈ 150 class files as blocks and k = 3 systems
            as treatments (one test per criterion).
  PART 2 — One cross-criteria Friedman test.
            Uses n = 5 criteria as blocks, k = 3 systems as treatments.
            Each cell is the mean score of a system on that criterion.

Scoring convention (from multi_criteria_rankings.csv):
    1st place = 3 pts,  2nd place = 2 pts,  3rd place = 1 pt

Post-hoc pairwise Wilcoxon signed-rank tests (Bonferroni-corrected) are run
for any significant Friedman result where n >= 10.  For Part 2 (n = 5) the
pairwise mean-rank differences are reported as a descriptive alternative.

Results are printed to the console and appended to evaluation-results/results.txt.

Systems mapping in the CSV:
    summary_a / total_points_a  →  DPS_NLG
    summary_b / total_points_b  →  DPS_LLM
    summary_c / total_points_c  →  DPS_SWUM
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA = 0.05
K = 3                               # number of systems (treatments)
BONFERRONI_PAIRS = 3                # C(3,2) pairwise comparisons
ALPHA_ADJ = ALPHA / BONFERRONI_PAIRS  # = 0.01667

BASE_DIR = Path(__file__).resolve().parent.parent
# Old default retained as comment for reference:
# RANKINGS_CSV = BASE_DIR / "evaluation-results" / "multi_criteria_rankings.csv"
RANKINGS_CSV = BASE_DIR / "evaluation-results" / "model_comparisons_ranking_detail.csv"
RESULTS_TXT = BASE_DIR / "evaluation-results" / "results.txt"

CRITERIA = [
    ("accuracy",        "ACCURACY"),
    ("conciseness",     "CONCISENESS"),
    ("adequacy",        "ADEQUACY"),
    ("code_context",    "CODE CONTEXT"),
    ("design_patterns", "DESIGN PATTERN"),
]

SYSTEM_A_LABEL = "NLG"
SYSTEM_B_LABEL = "MODEL"
SYSTEM_C_LABEL = "SWUM"
SYSTEMS = [SYSTEM_A_LABEL, SYSTEM_B_LABEL, SYSTEM_C_LABEL]  # maps to a, b, c in the CSV


def set_system_labels(system_b_label: str = "MODEL") -> None:
    """Set display labels for system A/B/C used in report text."""
    global SYSTEM_A_LABEL, SYSTEM_B_LABEL, SYSTEM_C_LABEL, SYSTEMS
    SYSTEM_A_LABEL = "NLG"
    SYSTEM_B_LABEL = system_b_label
    SYSTEM_C_LABEL = "SWUM"
    SYSTEMS = [SYSTEM_A_LABEL, SYSTEM_B_LABEL, SYSTEM_C_LABEL]


def _norm_rank_as_int(value) -> int | None:
    """Normalize rank values that may appear as 1, 1.0, '1', etc."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        num = float(text)
    except ValueError:
        return None

    rounded = int(round(num))
    if rounded in (1, 2, 3) and abs(num - rounded) < 1e-6:
        return rounded
    return None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def decode_scores(
    df: pd.DataFrame, criterion: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-system score arrays (values 3/2/1) for one criterion.

    Columns *_rank_1st / _2nd / _3rd encode which system (1=NLG, 2=LLM,
    3=SWUM) holds each rank position.  Rows containing any NaN are dropped
    so all three returned arrays have identical length.
    """
    c1 = f"{criterion}_rank_1st"
    c2 = f"{criterion}_rank_2nd"
    c3 = f"{criterion}_rank_3rd"

    sub = df[[c1, c2, c3]].copy()
    sub[c1] = sub[c1].apply(_norm_rank_as_int)
    sub[c2] = sub[c2].apply(_norm_rank_as_int)
    sub[c3] = sub[c3].apply(_norm_rank_as_int)
    sub = sub.dropna()

    # Keep only rows that contain a strict 1/2/3 permutation.
    valid = sub.apply(lambda row: {int(row[c1]), int(row[c2]), int(row[c3])} == {1, 2, 3}, axis=1)
    sub = sub[valid]

    r1 = sub[c1].astype(int).values
    r2 = sub[c2].astype(int).values
    r3 = sub[c3].astype(int).values

    def scores_for(sys_id: int) -> np.ndarray:
        # Exactly one of the three conditions is True per row.
        return ((r1 == sys_id) * 3 + (r2 == sys_id) * 2 + (r3 == sys_id) * 1).astype(float)

    return scores_for(1), scores_for(2), scores_for(3)


def kendalls_w(chi2: float, n: int, k: int) -> float:
    """Kendall's W from a Friedman chi² statistic.

    W = chi2 / (n * (k – 1))  where n = blocks, k = treatments.
    Interpretation: 0.1 weak, 0.3 moderate, 0.5 strong, 0.7 very strong.
    """
    if n == 0 or k <= 1:
        return float("nan")
    return chi2 / (n * (k - 1))


def concordance_label(W: float) -> str:
    if W >= 0.7:
        return "very strong"
    if W >= 0.5:
        return "strong"
    if W >= 0.3:
        return "moderate"
    return "weak"


# ---------------------------------------------------------------------------
# Post-hoc pairwise Wilcoxon (Bonferroni-corrected)
# ---------------------------------------------------------------------------

class PostHocResult(NamedTuple):
    label: str
    w_stat: Optional[float]
    p_raw: Optional[float]
    p_adj: Optional[float]
    significant: bool
    direction: str
    median_diff: float
    note: str = ""


def pairwise_posthoc(
    sa: np.ndarray, sb: np.ndarray, sc: np.ndarray
) -> list[PostHocResult]:
    """Run three pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = [
        (f"{SYSTEM_A_LABEL:<8} vs {SYSTEM_B_LABEL:<8}", sa, sb),
        (f"{SYSTEM_A_LABEL:<8} vs {SYSTEM_C_LABEL:<8}", sa, sc),
        (f"{SYSTEM_B_LABEL:<8} vs {SYSTEM_C_LABEL:<8}", sb, sc),
    ]

    results: list[PostHocResult] = []
    for label, x, y in pairs:
        diff = (x - y).astype(float)
        med = float(np.median(diff))

        if np.all(diff == 0.0):
            results.append(PostHocResult(
                label=label, w_stat=None, p_raw=None, p_adj=None,
                significant=False, direction="tie", median_diff=med,
                note="all differences = 0",
            ))
            continue

        try:
            w_stat, p_raw = wilcoxon(x, y, alternative="two-sided")
        except ValueError as exc:
            results.append(PostHocResult(
                label=label, w_stat=None, p_raw=None, p_adj=None,
                significant=False, direction="n/a", median_diff=med,
                note=str(exc),
            ))
            continue

        p_adj = min(p_raw * BONFERRONI_PAIRS, 1.0)
        sig = p_adj < ALPHA

        if med > 0:
            name1 = label.split("vs")[0].strip()
            name2 = label.split("vs")[1].strip()
            direction = f"{name1} > {name2}"
        elif med < 0:
            name1 = label.split("vs")[0].strip()
            name2 = label.split("vs")[1].strip()
            direction = f"{name2} > {name1}"
        else:
            direction = "tie"

        results.append(PostHocResult(
            label=label, w_stat=float(w_stat), p_raw=float(p_raw), p_adj=float(p_adj),
            significant=sig, direction=direction, median_diff=med,
        ))

    return results


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

SEP_HEAVY = "=" * 72
SEP_LIGHT = "-" * 72


def fmt_section_header(title: str) -> str:
    return f"\n{SEP_HEAVY}\n{title}\n{SEP_HEAVY}"


def fmt_posthoc_block(results: list[PostHocResult]) -> list[str]:
    lines = [
        "",
        f"  Post-hoc pairwise Wilcoxon signed-rank tests"
        f" (Bonferroni α_adj = {ALPHA_ADJ:.4f}):",
        f"  {'Comparison':<26} {'W-stat':>10} {'p (raw)':>12} "
        f"{'p (adj)':>12} {'Sig?':>6}  Direction",
        f"  {'-'*82}",
    ]
    for r in results:
        if r.note:
            lines.append(f"  {r.label:<26}  [skipped — {r.note}]")
        else:
            sig_flag = "YES *" if r.significant else "no"
            lines.append(
                f"  {r.label:<26} {r.w_stat:>10.2f} {r.p_raw:>12.6f} "
                f"{r.p_adj:>12.6f} {sig_flag:>6}  {r.direction}"
            )
    return lines


# ---------------------------------------------------------------------------
# PART 1 — Per-criterion Friedman tests (n ≈ 150 blocks)
# ---------------------------------------------------------------------------

def run_part1(df: pd.DataFrame) -> tuple[str, list[dict]]:
    """One Friedman test per criterion; class files are blocks (n ≈ 150)."""
    lines: list[str] = [
        fmt_section_header("PART 1: PER-CRITERION FRIEDMAN TESTS  (class files as blocks)"),
        "",
        f"  Test design : one Friedman test per criterion",
        f"  Blocks      : individual class files  (n ≈ 150 per test)",
        f"  Treatments  : k = 3 systems  ({SYSTEM_A_LABEL}, {SYSTEM_B_LABEL}, {SYSTEM_C_LABEL})",
        f"  Scoring     : 1st = 3 pts | 2nd = 2 pts | 3rd = 1 pt",
        f"  H0          : no significant difference among the 3 systems",
        f"  α           : {ALPHA}  |  post-hoc Bonferroni α_adj = {ALPHA_ADJ:.4f}",
    ]

    summary_rows: list[dict] = []

    for crit_key, crit_label in CRITERIA:
        sa, sb, sc = decode_scores(df, crit_key)
        n = len(sa)

        chi2, pval = friedmanchisquare(sa, sb, sc)
        W = kendalls_w(chi2, n, K)

        means = {
            SYSTEM_A_LABEL: float(sa.mean()),
            SYSTEM_B_LABEL: float(sb.mean()),
            SYSTEM_C_LABEL: float(sc.mean()),
        }
        best = max(means, key=means.get)
        sig = pval < ALPHA

        lines += [
            "",
            SEP_LIGHT,
            f"  Criterion      : {crit_label}",
            f"  N (blocks)     : {n} class files",
            f"  Friedman chi²  = {chi2:.4f}",
            f"  p-value        = {pval:.6f}  "
            + ("  < 0.05  →  REJECT H0 *" if sig else "  >= 0.05  →  fail to reject H0"),
            f"  Kendall's W    = {W:.4f}  ({concordance_label(W)} concordance)",
            f"  Significant    : {'YES' if sig else 'NO'}",
            "",
            "  Mean scores per system (higher = better ranked):",
        ]

        for sys_name in SYSTEMS:
            marker = "  <-- best" if sys_name == best else ""
            lines.append(f"    {sys_name:<12s} : {means[sys_name]:.4f}{marker}")

        if sig:
            ph = pairwise_posthoc(sa, sb, sc)
            lines.extend(fmt_posthoc_block(ph))
        else:
            lines.append(
                "\n  Post-hoc tests : not conducted (overall Friedman not significant)"
            )

        summary_rows.append({
            "criterion":  crit_label,
            "n":          n,
            "chi2":       chi2,
            "p_value":    pval,
            "kendalls_w": W,
            "significant": sig,
            "best":       best,
        })

    return "\n".join(lines), summary_rows


# ---------------------------------------------------------------------------
# PART 2 — Cross-criteria Friedman test (n = 5 criteria as blocks)
# ---------------------------------------------------------------------------

def run_part2(df: pd.DataFrame) -> str:
    """One Friedman test with 5 criteria as blocks and 3 systems as treatments.

    Each cell in the 5 × 3 matrix is the mean score of a system on that
    criterion, averaged across all class files.
    """
    lines: list[str] = [
        fmt_section_header("PART 2: CROSS-CRITERIA FRIEDMAN TEST  (criteria as blocks)"),
        "",
        "  Test design : one Friedman test across all criteria",
        "  Blocks      : n = 5 criteria  (ACCURACY, CONCISENESS, ADEQUACY,",
        "                                  CODE CONTEXT, DESIGN PATTERN)",
        f"  Treatments  : k = 3 systems  ({SYSTEM_A_LABEL}, {SYSTEM_B_LABEL}, {SYSTEM_C_LABEL})",
        "  Cell value  : mean score of system on criterion across all class files",
        "  H0          : rank ordering of systems does not differ across criteria",
        f"  α           : {ALPHA}",
        "",
    ]

    # Build 5 × 3 mean-score matrix
    mean_matrix = np.zeros((len(CRITERIA), K), dtype=float)
    for i, (crit_key, _) in enumerate(CRITERIA):
        sa, sb, sc = decode_scores(df, crit_key)
        mean_matrix[i, 0] = sa.mean()
        mean_matrix[i, 1] = sb.mean()
        mean_matrix[i, 2] = sc.mean()

    # Print the data matrix
    lines += [
        "  Mean score matrix:",
        f"  {'Criterion':<22s} {SYSTEM_A_LABEL:>10s} {SYSTEM_B_LABEL:>10s} {SYSTEM_C_LABEL:>10s}",
        f"  {'-' * 54}",
    ]
    for i, (_, crit_label) in enumerate(CRITERIA):
        row = mean_matrix[i]
        best_col = int(np.argmax(row))
        markers = ["", "", ""]
        markers[best_col] = " *"
        lines.append(
            f"  {crit_label:<22s} {row[0]:>9.4f}{markers[0]}"
            f" {row[1]:>9.4f}{markers[1]}"
            f" {row[2]:>9.4f}{markers[2]}"
        )
    col_means = mean_matrix.mean(axis=0)
    lines += [
        f"  {'-' * 54}",
        f"  {'COLUMN MEAN':<22s} {col_means[0]:>10.4f} {col_means[1]:>10.4f}"
        f" {col_means[2]:>10.4f}",
        "  (* = best system for that criterion)",
        "",
    ]

    sa = mean_matrix[:, 0]
    sb = mean_matrix[:, 1]
    sc = mean_matrix[:, 2]
    n = len(CRITERIA)

    chi2, pval = friedmanchisquare(sa, sb, sc)
    W = kendalls_w(chi2, n, K)
    sig = pval < ALPHA

    lines += [
        f"  Friedman chi²  = {chi2:.4f}",
        f"  p-value        = {pval:.6f}  "
        + ("  < 0.05  →  REJECT H0 *" if sig else "  >= 0.05  →  fail to reject H0"),
        f"  Kendall's W    = {W:.4f}  ({concordance_label(W)} concordance)",
        f"  Significant    : {'YES' if sig else 'NO'}",
    ]

    if sig:
        lines += [
            "",
            f"  Note: n = 5 blocks — Wilcoxon signed-rank requires n >= 6.",
            f"  Pairwise mean-score differences (descriptive):",
            f"  {'Comparison':<26} {'Mean diff':>10}  Direction",
            f"  {'-' * 52}",
        ]
        pairs_desc = [
            (f"{SYSTEM_A_LABEL:<8} vs {SYSTEM_B_LABEL:<8}", sa - sb, SYSTEM_A_LABEL, SYSTEM_B_LABEL),
            (f"{SYSTEM_A_LABEL:<8} vs {SYSTEM_C_LABEL:<8}", sa - sc, SYSTEM_A_LABEL, SYSTEM_C_LABEL),
            (f"{SYSTEM_B_LABEL:<8} vs {SYSTEM_C_LABEL:<8}", sb - sc, SYSTEM_B_LABEL, SYSTEM_C_LABEL),
        ]
        for label, diff, n1, n2 in pairs_desc:
            md = float(diff.mean())
            direction = f"{n1} > {n2}" if md > 0 else (f"{n2} > {n1}" if md < 0 else "tie")
            lines.append(f"  {label:<26} {md:>+10.4f}  {direction}")
    else:
        lines.append(
            "\n  Post-hoc : not conducted (overall Friedman not significant)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary table (PART 1 results)
# ---------------------------------------------------------------------------

def run_summary_table(summary_rows: list[dict]) -> str:
    lines: list[str] = [
        fmt_section_header("SUMMARY TABLE — PER-CRITERION FRIEDMAN RESULTS"),
        "",
        f"  {'Criterion':<22s} {'n':>5s} {'chi²':>10s} {'p-value':>12s} "
        f"{'Kendall W':>10s} {'Sig?':>5s}  Best System",
        f"  {'-' * 76}",
    ]
    for r in summary_rows:
        sig = "YES *" if r["significant"] else "no"
        lines.append(
            f"  {r['criterion']:<22s} {r['n']:>5d} {r['chi2']:>10.4f} "
            f"{r['p_value']:>12.6f} {r['kendalls_w']:>10.4f} {sig:>5s}  {r['best']}"
        )
    lines += [
        "",
        f"  α = {ALPHA}  |  post-hoc Bonferroni α_adj = {ALPHA_ADJ:.4f}",
        f"  Kendall's W : 0.1 weak | 0.3 moderate | 0.5 strong | 0.7 very strong",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Friedman tests on ranking detail data.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RANKINGS_CSV,
        help="Path to rankings detail CSV",
    )
    parser.add_argument(
        "--append-to",
        type=Path,
        default=RESULTS_TXT,
        help="Path to report file where results are appended",
    )
    parser.add_argument(
        "--by-comparison",
        action="store_true",
        help="Run one Friedman analysis per comparison value (when comparison column exists)",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=None,
        help="Run only one comparison value (e.g., GEMINI, GPT, CLAUDE, MISTRAL)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Rankings file not found: {args.csv}")

    df_all = pd.read_csv(args.csv)
    if "status" in df_all.columns:
        df_all = df_all[df_all["status"].astype(str).str.lower() == "ranked"].copy()

    def run_single_analysis(df_subset: pd.DataFrame, label: str, b_label: str) -> str:
        set_system_labels(system_b_label=b_label)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = "\n".join([
            "",
            SEP_HEAVY,
            f"FRIEDMAN TEST ANALYSIS — {label}",
            SEP_HEAVY,
            f"  Run date   : {timestamp}",
            f"  Data file  : {args.csv.name}",
            f"  Rows used  : {len(df_subset)}",
            "",
            f"  Systems    :  A = {SYSTEM_A_LABEL}   |  B = {SYSTEM_B_LABEL}   |  C = {SYSTEM_C_LABEL}",
            "  Scoring    :  1st place = 3 pts  |  2nd = 2 pts  |  3rd = 1 pt",
            f"  Alpha      :  {ALPHA}",
            f"  Post-hoc   :  Wilcoxon signed-rank, Bonferroni α_adj = {ALPHA_ADJ:.4f}",
            "",
            "  H0 (each Friedman test): no significant difference in scores",
            "  among the three systems under the given criterion / criteria.",
        ])

        part1_text, summary_rows = run_part1(df_subset)
        part2_text = run_part2(df_subset)
        summary_text = run_summary_table(summary_rows)

        return (
            header + "\n"
            + part1_text + "\n"
            + part2_text + "\n"
            + summary_text + "\n"
        )

    outputs: list[str] = []

    if args.comparison is not None:
        if "comparison" not in df_all.columns:
            raise ValueError("--comparison requested but comparison column is missing")
        target = args.comparison.strip().upper()
        df_target = df_all[df_all["comparison"].astype(str).str.upper() == target].copy()
        if df_target.empty:
            raise ValueError(f"No rows found for comparison={target}")
        outputs.append(run_single_analysis(df_target, f"NLG vs {target} vs SWUM", target))

    elif args.by_comparison and "comparison" in df_all.columns:
        ordered = sorted(df_all["comparison"].dropna().astype(str).str.upper().unique().tolist())
        for comparison_name in ordered:
            df_target = df_all[df_all["comparison"].astype(str).str.upper() == comparison_name].copy()
            outputs.append(run_single_analysis(df_target, f"NLG vs {comparison_name} vs SWUM", comparison_name))

    else:
        outputs.append(run_single_analysis(df_all, "NLG vs MODEL vs SWUM (ALL COMPARISONS COMBINED)", "MODEL"))

    combined_output = "\n".join(outputs)
    print(combined_output)

    with open(args.append_to, "a", encoding="utf-8") as fh:
        fh.write(combined_output)
        fh.write("\n")

    print(f"Results successfully appended to: {args.append_to}")


if __name__ == "__main__":
    main()
