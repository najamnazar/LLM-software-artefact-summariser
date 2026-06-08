"""
Compute average, min, and max summary lengths (words and characters)
for multiple NLG output files.

Targets:
- output/summary-output/nlg_summaries.csv
- output/summary-output/LLM_CLAUDE_SUMMARY.csv
- output/summary-output/LLM_GEMINI_SUMMARY.csv
- output/summary-output/LLM_GPT_SUMMARY.csv
- output/summary-output/LLM_MISTRAL_SUMMARY.csv
- output/summary-output/LLM_CLAUDE_NC_SUMMARY.csv
- output/summary-output/LLM_GEMINI_NC_SUMMARY.csv
- output/summary-output/LLM_GPT_NC_SUMMARY.csv
- output/summary-output/LLM_MISTRAL_NC_SUMMARY.csv
- output/summary-output/swum_summaries.csv

Outputs a plain-text table at evaluation-results/summary_length_stats.txt.
Object-oriented implementation with useful comments.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SummaryRow:
    source: str  # dataset/source name (e.g., dps_nlg)
    summary: str

    def token_count(self) -> int:
        # Simple word tokenizer: sequences of alphanumerics; stable across datasets
        return len(re.findall(r"\b\w+\b", self.summary))

    def char_count(self) -> int:
        # Count non-whitespace characters to reduce formatting differences
        return sum(1 for c in self.summary if not c.isspace())


class SummaryFileLoader:
    """Load summaries from a CSV, locating the Summary column robustly."""

    def __init__(self, path: str, source_name: str) -> None:
        self.path = path
        self.source_name = source_name

    def load(self) -> List[SummaryRow]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Summary CSV not found: {self.path}")
        with open(self.path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"No headers found in {self.path}")
            # Try to find a 'Summary' column (case-insensitive)
            summary_col: Optional[str] = None
            for col in reader.fieldnames:
                if col.strip().lower() == "summary":
                    summary_col = col
                    break
            if summary_col is None:
                raise ValueError(
                    f"Summary column not found in {self.path}. Columns: {reader.fieldnames}"
                )
            rows: List[SummaryRow] = []
            for row in reader:
                summary_text = (row.get(summary_col) or "").strip()
                if not summary_text:
                    continue
                rows.append(SummaryRow(source=self.source_name, summary=summary_text))
            return rows


@dataclass
class LengthStats:
    source: str
    count: int
    avg_words: float
    min_words: int
    max_words: int
    avg_chars: float
    min_chars: int
    max_chars: int

    def as_row(self) -> Tuple[str, int, float, int, int, float, int, int]:
        return (
            self.source,
            self.count,
            self.avg_words,
            self.min_words,
            self.max_words,
            self.avg_chars,
            self.min_chars,
            self.max_chars,
        )


class SummaryLengthAnalyzer:
    """End-to-end analysis: load -> compute -> render -> write."""

    def __init__(self, files: Dict[str, str], output_path: str) -> None:
        # files: mapping of source name -> CSV path
        self.files = files
        self.output_path = output_path
        self.rows: List[SummaryRow] = []
        self.stats: List[LengthStats] = []

    def load_all(self) -> None:
        for source, path in self.files.items():
            loader = SummaryFileLoader(path, source)
            self.rows.extend(loader.load())

    def compute(self) -> None:
        by_source: Dict[str, List[SummaryRow]] = {}
        for r in self.rows:
            by_source.setdefault(r.source, []).append(r)
        stats: List[LengthStats] = []
        for source, rows in by_source.items():
            if not rows:
                continue
            word_counts = [r.token_count() for r in rows]
            char_counts = [r.char_count() for r in rows]
            count = len(rows)
            avg_words = sum(word_counts) / count
            avg_chars = sum(char_counts) / count
            stats.append(
                LengthStats(
                    source=source,
                    count=count,
                    avg_words=avg_words,
                    min_words=min(word_counts),
                    max_words=max(word_counts),
                    avg_chars=avg_chars,
                    min_chars=min(char_counts),
                    max_chars=max(char_counts),
                )
            )
        self.stats = stats

    def render_table(self) -> str:
        headers = [
            "Source",
            "Count",
            "AvgWords",
            "MinWords",
            "MaxWords",
            "AvgChars",
            "MinChars",
            "MaxChars",
        ]
        rows = [s.as_row() for s in self.stats]
        # Sort by AvgWords ascending
        rows.sort(key=lambda r: r[2])
        # Column widths based on headers and formatted data
        col_widths = [len(h) for h in headers]
        formatted_rows: List[List[str]] = []
        for r in rows:
            fr = [
                f"{r[0]}",
                f"{r[1]}",
                f"{r[2]:.2f}",
                f"{r[3]}",
                f"{r[4]}",
                f"{r[5]:.2f}",
                f"{r[6]}",
                f"{r[7]}",
            ]
            formatted_rows.append(fr)
            for i, cell in enumerate(fr):
                col_widths[i] = max(col_widths[i], len(cell))

        def pad_row(cells: List[str]) -> str:
            padded = [
                f"{cells[0]:<{col_widths[0]}}",
                f"{cells[1]:>{col_widths[1]}}",
                f"{cells[2]:>{col_widths[2]}}",
                f"{cells[3]:>{col_widths[3]}}",
                f"{cells[4]:>{col_widths[4]}}",
                f"{cells[5]:>{col_widths[5]}}",
                f"{cells[6]:>{col_widths[6]}}",
                f"{cells[7]:>{col_widths[7]}}",
            ]
            return " | ".join(padded)

        header_line = pad_row(headers)
        sep_line = "-" * len(header_line)
        body = "\n".join(pad_row(fr) for fr in formatted_rows)
        return "\n".join([header_line, sep_line, body])

    def write(self) -> str:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        table = self.render_table()
        header = (
            "Summary Length Statistics (words and non-whitespace characters)\n"
            "Shows averages and ranges for each source file.\n"
        )
        with open(self.output_path, mode="w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n")
            f.write(table)
            f.write("\n")
        return self.output_path


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    def resolve_summary_file(*relative_names: str) -> Tuple[str, str]:
        base_dir = os.path.join(repo_root, "output", "summary-output")
        for name in relative_names:
            candidate = os.path.join(base_dir, name)
            if os.path.exists(candidate):
                label = os.path.splitext(name)[0]
                return label, candidate
        joined = ", ".join(relative_names)
        raise FileNotFoundError(f"None of the candidate summary files exist: {joined}")

    # Old default source set retained for reference:
    # resolved_sources = [
    #     resolve_summary_file("nlg_summaries.csv"),
    #     resolve_summary_file("llm_summaries.csv"),
    #     resolve_summary_file("swum_summaries.csv"),
    # ]

    # Updated source set: NLG + four LLM model variants (standard + NC) + SWUM.
    resolved_sources = [
        resolve_summary_file("nlg_summaries.csv"),
        resolve_summary_file("LLM_CLAUDE_SUMMARY.csv"),
        resolve_summary_file("LLM_GEMINI_SUMMARY.csv"),
        resolve_summary_file("LLM_GPT_SUMMARY.csv"),
        resolve_summary_file("LLM_MISTRAL_SUMMARY.csv"),
        resolve_summary_file("LLM_CLAUDE_NC_SUMMARY.csv", "LLM_CLAUDE_SUMMARY.csv"),
        resolve_summary_file("LLM_GEMINI_NC_SUMMARY.csv", "LLM_GEMINI_SUMMARY.csv"),
        resolve_summary_file("LLM_GPT_NC_SUMMARY.csv", "LLM_GPT_SUMMARY.csv"),
        resolve_summary_file("LLM_MISTRAL_NC_SUMMARY.csv", "LLM_MISTRAL_SUMMARY.csv"),
        resolve_summary_file("swum_summaries.csv"),
    ]
    files = {label: path for label, path in resolved_sources}
    out_path = os.path.join(repo_root, "evaluation-results", "summary_length_stats.txt")

    analyzer = SummaryLengthAnalyzer(files, out_path)
    analyzer.load_all()
    analyzer.compute()
    output_file = analyzer.write()
    print(f"Wrote summary length table to: {output_file}")


if __name__ == "__main__":
    main()
