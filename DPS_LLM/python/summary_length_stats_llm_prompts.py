"""
Compute length statistics for each prompt-specific LLM summary CSV.

Generates four separate tables (one per prompt alias) mirroring the format
produced by summary_length_stats.py, but scoped to the following files:
- output/summary-output/llm_summaries_20.csv
- output/summary-output/llm_summaries_40.csv
- output/summary-output/llm_summaries_60.csv
- output/summary-output/llm_summaries_80.csv

The aggregated report is written to
  evaluation-results/llm_prompt_summary_length_stats.txt.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

from summary_length_stats import SummaryLengthAnalyzer


def analyze_prompt_files(prompt_files: List[Tuple[str, str]]) -> Dict[str, str]:
    """Return a mapping of prompt alias to rendered stats table."""
    tables: Dict[str, str] = {}
    for alias, csv_path in prompt_files:
        # Reuse the existing analyzer to benefit from identical token/char logic
        analyzer = SummaryLengthAnalyzer({alias: csv_path}, output_path="unused")
        analyzer.load_all()
        analyzer.compute()
        tables[alias] = analyzer.render_table()
    return tables


def write_report(output_path: str, tables: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    banner_lines = [
        "Prompt-Specific LLM Summary Length Statistics",
        "Each section mirrors summary_length_stats.py but focuses on a single CSV.",
        "Lengths are reported in words (tokens) and non-whitespace characters.",
    ]
    with open(output_path, mode="w", encoding="utf-8") as handle:
        handle.write("\n".join(banner_lines))
        handle.write("\n\n")
        for alias, table in tables.items():
            # Separate sections so each prompt constraint is easy to scan
            handle.write(f"=== {alias} ===\n")
            handle.write(table)
            handle.write("\n\n")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    prompt_files: List[Tuple[str, str]] = [
        ("llm_summaries_20", os.path.join(repo_root, "output", "summary-output", "llm_summaries_20.csv")),
        ("llm_summaries_40", os.path.join(repo_root, "output", "summary-output", "llm_summaries_40.csv")),
        ("llm_summaries_60", os.path.join(repo_root, "output", "summary-output", "llm_summaries_60.csv")),
        ("llm_summaries_80", os.path.join(repo_root, "output", "summary-output", "llm_summaries_80.csv")),
    ]
    tables = analyze_prompt_files(prompt_files)

    output_path = os.path.join(
        repo_root, "evaluation-results", "llm_prompt_summary_length_stats.txt"
    )
    write_report(output_path, tables)
    print(f"Wrote prompt-specific summary length stats to: {output_path}")


if __name__ == "__main__":
    main()
