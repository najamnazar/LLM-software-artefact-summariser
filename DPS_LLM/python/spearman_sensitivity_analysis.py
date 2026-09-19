"""Compute per-criterion Spearman sensitivity analysis from ranking results.

This script computes Spearman correlation between:
- X: criterion points per system per class file (1st=3, 2nd=2, 3rd=1)
- Y: points of that same system on the OTHER four criteria for the same class file

Y excludes the criterion being tested. The earlier definition correlated X against the
full total, which already contained X as one fifth of its value, so every criterion was
partly being correlated with itself and rho was inflated by construction — a criterion
contributing nothing but noise would still have scored well above zero. The part-whole
value is still reported alongside the corrected one so older numbers remain traceable.

Input:
- evaluation-results/model_comparisons_ranking_detail.csv (default)
- evaluation-results/multi_criteria_rankings.csv (legacy)

Output:
- Prints a formatted table to stdout
- Optionally appends the section to evaluation-results/results.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_paths import guard_report_target  # noqa: E402  - sys.path must be extended first


@dataclass
class CriterionResult:
    criterion: str
    blocks: int
    n_obs: int
    rho: float
    p_value: float
    rho_partwhole: float
    p_value_partwhole: float


def _norm_rank(value) -> str | None:
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
        return str(rounded)
    return None


def compute_spearman_sensitivity_from_df(df: pd.DataFrame) -> list[CriterionResult]:
    """Compute criterion-level Spearman sensitivity from an already loaded DataFrame."""
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ranked"].copy()

    systems = {"1": "a", "2": "b", "3": "c"}
    criteria = [
        ("accuracy", "ACCURACY"),
        ("conciseness", "CONCISENESS"),
        ("adequacy", "ADEQUACY"),
        ("code_context", "CODE CONTEXT"),
        ("design_patterns", "DESIGN PATTERN"),
    ]

    results: list[CriterionResult] = []

    for criterion_key, criterion_name in criteria:
        rank_cols = [
            f"{criterion_key}_rank_1st",
            f"{criterion_key}_rank_2nd",
            f"{criterion_key}_rank_3rd",
        ]

        if not all(col in df.columns for col in rank_cols):
            continue

        rows: list[tuple[int, float, float]] = []
        valid_blocks = 0

        for _, row in df.iterrows():
            first = _norm_rank(row.get(rank_cols[0]))
            second = _norm_rank(row.get(rank_cols[1]))
            third = _norm_rank(row.get(rank_cols[2]))

            if {first, second, third} != {"1", "2", "3"}:
                continue

            valid_blocks += 1
            points = {first: 3, second: 2, third: 1}

            for system_id, suffix in systems.items():
                total_points = row.get(f"total_points_{suffix}")
                if pd.isna(total_points):
                    continue
                criterion_points = points[system_id]
                # Remove this criterion's own contribution from the total so X is not
                # correlated against a Y that contains it.
                rest_points = float(total_points) - criterion_points
                rows.append((criterion_points, rest_points, float(total_points)))

        data = pd.DataFrame(rows, columns=["criterion_points", "rest_points", "total_points"])
        if len(data) < 2:
            continue
        rho, p_value = spearmanr(data["criterion_points"], data["rest_points"])
        rho_pw, p_pw = spearmanr(data["criterion_points"], data["total_points"])

        results.append(
            CriterionResult(
                criterion=criterion_name,
                blocks=valid_blocks,
                n_obs=len(data),
                rho=float(rho),
                p_value=float(p_value),
                rho_partwhole=float(rho_pw),
                p_value_partwhole=float(p_pw),
            )
        )

    return results


def compute_spearman_sensitivity(csv_path: Path) -> list[CriterionResult]:
    """Compatibility wrapper that loads CSV then computes sensitivity."""
    return compute_spearman_sensitivity_from_df(pd.read_csv(csv_path))


def format_section(results: list[CriterionResult], title_suffix: str = "") -> str:
    lines: list[str] = []
    lines.append("========================================================================")
    title = "SENSITIVITY ANALYSIS — SPEARMAN CORRELATION (PER CRITERION)"
    if title_suffix:
        title = f"{title} [{title_suffix}]"
    lines.append(title)
    lines.append("========================================================================")
    lines.append("")
    lines.append("  Definition:")
    lines.append("  For each criterion, Spearman's rho is computed between:")
    lines.append("    X = criterion points per system per class file (1st=3, 2nd=2, 3rd=1)")
    lines.append("    Y = points of that same system on the OTHER four criteria")
    lines.append("  This quantifies how strongly each criterion agrees with the remaining criteria.")
    lines.append("")
    lines.append("  Y deliberately excludes the criterion under test. Correlating X against the")
    lines.append("  full total would correlate it partly with itself (X is one fifth of the total),")
    lines.append("  inflating rho for every criterion regardless of its actual agreement. The")
    lines.append("  part-whole column reproduces that older, inflated figure for comparison only.")
    lines.append("")
    lines.append("------------------------------------------------------------------------")
    lines.append("  Criterion                  blocks   n(obs)   Spearman rho      p-value   Sig?    part-whole rho")
    lines.append("  ------------------------------------------------------------------------------------------------")

    for r in results:
        sig = "YES *" if r.p_value < 0.05 else "no"
        lines.append(
            f"  {r.criterion:<25} {r.blocks:>6} {r.n_obs:>8} {r.rho:>15.6f} {r.p_value:>14.5e}  {sig:<6} "
            f"{r.rho_partwhole:>15.6f}"
        )

    lines.append("")
    lines.append("  α = 0.05   |   part-whole rho = superseded definition, reported for traceability")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Spearman sensitivity per criterion.")
    # Old default retained as comment for reference:
    # legacy_default_csv = Path("evaluation-results/multi_criteria_rankings.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("evaluation-results/model_comparisons_ranking_detail.csv"),
        help="Path to rankings detail CSV",
    )
    parser.add_argument(
        "--by-comparison",
        action="store_true",
        help="Compute one Spearman section per comparison value (when comparison column exists)",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=None,
        help="Run only one comparison value (e.g., QWEN, GPT, CLAUDE, MISTRAL)",
    )
    parser.add_argument(
        "--append-to",
        type=Path,
        default=None,
        help="Append the formatted section to this text report file.",
    )
    parser.add_argument(
        "--allow-curated-append",
        action="store_true",
        help="Permit appending to a hand-curated report such as results.txt",
    )

    args = parser.parse_args()

    df_all = pd.read_csv(args.csv)
    sections: list[str] = []

    if args.comparison is not None:
        if "comparison" not in df_all.columns:
            raise ValueError("--comparison requested but comparison column is missing")
        target = args.comparison.strip().upper()
        df_target = df_all[df_all["comparison"].astype(str).str.upper() == target].copy()
        if df_target.empty:
            raise ValueError(f"No rows found for comparison={target}")
        results = compute_spearman_sensitivity_from_df(df_target)
        sections.append(format_section(results, title_suffix=target))

    elif args.by_comparison and "comparison" in df_all.columns:
        ordered = sorted(df_all["comparison"].dropna().astype(str).str.upper().unique().tolist())
        for comparison_name in ordered:
            df_target = df_all[df_all["comparison"].astype(str).str.upper() == comparison_name].copy()
            results = compute_spearman_sensitivity_from_df(df_target)
            sections.append(format_section(results, title_suffix=comparison_name))

    else:
        results = compute_spearman_sensitivity_from_df(df_all)
        sections.append(format_section(results))

    section = "\n\n".join(sections)
    print(section)

    if args.append_to is not None:
        guard_report_target(args.append_to, args.allow_curated_append)
        args.append_to.parent.mkdir(parents=True, exist_ok=True)
        with args.append_to.open("a", encoding="utf-8") as handle:
            handle.write("\n" + section + "\n")
        print(f"Appended section to: {args.append_to}")


if __name__ == "__main__":
    main()
