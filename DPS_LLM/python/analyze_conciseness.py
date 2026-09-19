"""
Conciseness analysis for corpus B summaries by design pattern.

This script reads the CSV file produced for corpus B (e.g., B.csv),
groups summaries by the design pattern (taken from the "Folder Name" column),
computes simple, interpretable conciseness metrics, and writes a plain-text table
into evaluation-results/conciseness_by_pattern.txt.

Implementation is object-oriented for clarity and extensibility.
Only Python standard library is used to avoid extra environment dependencies.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re


@dataclass
class SummaryRecord:
    """Represents a single summary row from the CSV.

    Attributes:
        pattern: The design pattern (taken from the "Folder Name" column).
        summary: The textual summary content.
    """
    pattern: str
    summary: str

    def tokens(self) -> List[str]:
        """Tokenize summary into words using a simple whitespace split.
        This is fast and robust for our purpose (relative comparisons).
        """
        # Normalize whitespace; split on whitespace
        return [t for t in self.summary.replace("\n", " ").split() if t]

    def sentences(self) -> List[str]:
        """Very simple heuristic sentence splitter on punctuation.

        We deliberately avoid external dependencies (e.g., nltk); while this is
        simplistic, it works adequately for relative conciseness comparisons.
        """
        text = self.summary.strip()
        if not text:
            return []
        # Split on period, exclamation, question mark and keep non-empty parts
        parts: List[str] = []
        start = 0
        for i, ch in enumerate(text):
            if ch in ".!?":
                seg = text[start : i].strip()
                if seg:
                    parts.append(seg)
                start = i + 1
        # Trailing tail without terminal punctuation
        tail = text[start:].strip()
        if tail:
            parts.append(tail)
        return parts

    def word_count(self) -> int:
        return len(self.tokens())

    def char_count(self) -> int:
        # Count non-whitespace characters to reduce formatting noise
        return sum(1 for c in self.summary if not c.isspace())

    def words_per_sentence(self) -> float:
        sents = self.sentences()
        if not sents:
            return float(self.word_count()) if self.word_count() else 0.0
        return self.word_count() / len(sents)

    def type_token_ratio(self) -> float:
        """Type-Token Ratio (TTR): unique words / total words (0..1).
        Higher TTR often indicates less repetition; we include it as a
        supporting conciseness/variety signal.
        """
        toks = self.tokens()
        if not toks:
            return 0.0
        # Lowercase and strip basic punctuation for stability
        cleaned = [t.strip(".,;:!?()[]{}\"'`").lower() for t in toks if t]
        unique = set(cleaned)
        return len(unique) / len(cleaned) if cleaned else 0.0


@dataclass
class PatternStats:
    """Aggregates conciseness metrics for a design pattern."""
    pattern: str
    word_counts: List[int]
    char_counts: List[int]
    wps_values: List[float]
    ttr_values: List[float]

    @property
    def n(self) -> int:
        return len(self.word_counts)

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    @staticmethod
    def _safe_median(values: List[float]) -> float:
        return float(statistics.median(values)) if values else 0.0

    @staticmethod
    def _safe_stdev(values: List[float]) -> float:
        return float(statistics.stdev(values)) if len(values) > 1 else 0.0

    def as_row(self) -> Tuple[str, int, float, float, float, float, float, float]:
        """Return a tuple of metrics suitable for tabular output."""
        avg_words = self._safe_mean([float(x) for x in self.word_counts])
        med_words = self._safe_median([float(x) for x in self.word_counts])
        std_words = self._safe_stdev([float(x) for x in self.word_counts])
        avg_chars = self._safe_mean([float(x) for x in self.char_counts])
        avg_wps = self._safe_mean(self.wps_values)
        avg_ttr = self._safe_mean(self.ttr_values)
        return (
            self.pattern,
            self.n,
            avg_words,
            med_words,
            std_words,
            avg_chars,
            avg_wps,
            avg_ttr,
        )


class ConcisenessAnalyzer:
    """Analyze conciseness of summaries by design pattern.

    This class encapsulates the full workflow: load -> compute -> render.
    """

    def __init__(self, csv_path: str, output_path: str) -> None:
        self.csv_path = csv_path
        self.output_path = output_path
        self.records: List[SummaryRecord] = []
        self.stats_by_pattern: Dict[str, PatternStats] = {}
        self._canonical_map = self._build_canonical_map()

    def _build_canonical_map(self) -> Dict[str, str]:
        """Map common variants to canonical design pattern names.

        This merges case/format variants like 'facade' and 'Facade',
        and hyphen/space variants like 'factory-method' and 'factory method'.
        Non-pattern categories (e.g., spring-*) are intentionally not included
        and will be filtered out.
        """
        patterns = {
            "abstract factory": "Abstract Factory",
            "abstract-factory": "Abstract Factory",
            "adapter": "Adapter",
            "decorator": "Decorator",
            "facade": "Facade",
            "factory method": "Factory Method",
            "factory-method": "Factory Method",
            "memento": "Memento",
            "observer": "Observer",
            "singleton": "Singleton",
            "visitor": "Visitor",
        }
        return patterns

    def _canonicalize_pattern(self, raw: str) -> Optional[str]:
        """Return canonical pattern name or None if excluded.

        Rules:
        - Exclude non-pattern groupings like 'spring-*'.
        - Normalize hyphens/underscores/spaces and case.
        - Map known variants to canonical names.
        - Drop unknown categories to keep the table focused on patterns.
        """
        if not raw:
            return None
        s = raw.strip()

        def _normalize_variants(token: str) -> List[str]:
            # Generate normalization variants including camel-case splits
            variants = set()
            lowered = token.lower().strip()
            if lowered:
                variants.add(lowered)
                variants.add(re.sub(r"[\s_-]+", " ", lowered).strip())
            camel_split = re.sub(r"(?<!^)(?=[A-Z])", " ", token).lower().strip()
            if camel_split:
                variants.add(camel_split)
                variants.add(re.sub(r"[\s_-]+", " ", camel_split).strip())
            return [v for v in variants if v]

        for candidate in _normalize_variants(s):
            if candidate in self._canonical_map:
                return self._canonical_map[candidate]
            hyphen_variant = candidate.replace(" ", "-")
            if hyphen_variant in self._canonical_map:
                return self._canonical_map[hyphen_variant]

        # If input already matches canonical (case-insensitive), title-case it
        canonical_candidates = set(self._canonical_map.values())
        if s in canonical_candidates:
            return s
        if s.title() in canonical_candidates:
            return s.title()
        # Unknown category -> exclude from analysis
        return None

    def load(self) -> None:
        """Load CSV rows as SummaryRecord objects."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"Folder Name", "Summary"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "CSV missing required columns: " + ", ".join(sorted(missing))
                )
            for row in reader:
                raw_pattern = (row.get("Folder Name") or "").strip()
                pattern = self._canonicalize_pattern(raw_pattern)
                summary = (row.get("Summary") or "").strip()
                if not pattern:
                    # Skip rows without a pattern label; they can't be grouped meaningfully
                    continue
                self.records.append(SummaryRecord(pattern=pattern, summary=summary))

    def compute(self) -> None:
        """Compute aggregated conciseness metrics by pattern."""
        buckets: Dict[str, List[SummaryRecord]] = {}
        for rec in self.records:
            buckets.setdefault(rec.pattern, []).append(rec)
        stats: Dict[str, PatternStats] = {}
        for pattern, recs in buckets.items():
            word_counts = [r.word_count() for r in recs]
            char_counts = [r.char_count() for r in recs]
            wps_values = [r.words_per_sentence() for r in recs]
            ttr_values = [r.type_token_ratio() for r in recs]
            stats[pattern] = PatternStats(
                pattern=pattern,
                word_counts=word_counts,
                char_counts=char_counts,
                wps_values=wps_values,
                ttr_values=ttr_values,
            )
        self.stats_by_pattern = stats

    @staticmethod
    def _format_float(x: float) -> str:
        # Format with 2 decimals, right-aligned for a tidy table
        return f"{x:>7.2f}"

    def render_table(self) -> str:
        """Create a plain-text table with per-pattern conciseness metrics.

        Columns:
        - Pattern: Design pattern name (from Folder Name)
        - Count: Number of summaries analyzed for the pattern
        - AvgWords: Average number of words per summary (lower is more concise)
        - MedianWords: Median number of words per summary (robust central tendency)
        - StdWords: Standard deviation of words (variability in length)
        - AvgChars: Average number of non-whitespace characters per summary
        - AvgWords/Sent: Average words per sentence (lower tends to be crisper)
        - AvgTTR: Average type-token ratio (unique/total words, 0..1)
        """
        headers = [
            "Pattern",
            "Count",
            "AvgWords",
            "MedianWords",
            "StdWords",
            "AvgChars",
            "AvgWords/Sent",
            "AvgTTR",
        ]
        # Compute rows and sort by AvgWords ascending (most concise first)
        rows = [s.as_row() for s in self.stats_by_pattern.values()]
        rows.sort(key=lambda r: r[2])  # sort by AvgWords

        # Define column widths
        col_widths = [
            max(len(headers[0]), max((len(r[0]) for r in rows), default=0)),  # pattern
            len(headers[1]),  # Count
            len(headers[2]),
            len(headers[3]),
            len(headers[4]),
            len(headers[5]),
            len(headers[6]),
            len(headers[7]),
        ]

        # Account for numeric formatting width
        def fmt_row(r: Tuple[str, int, float, float, float, float, float, float]) -> List[str]:
            return [
                f"{r[0]:<{col_widths[0]}}",
                f"{r[1]:>{col_widths[1]}}",
                f"{r[2]:>{col_widths[2]}.2f}",
                f"{r[3]:>{col_widths[3]}.2f}",
                f"{r[4]:>{col_widths[4]}.2f}",
                f"{r[5]:>{col_widths[5]}.2f}",
                f"{r[6]:>{col_widths[6]}.2f}",
                f"{r[7]:>{col_widths[7]}.2f}",
            ]

        # Recompute col widths based on formatted numbers
        formatted_rows: List[List[str]] = []
        for r in rows:
            fr = fmt_row(r)
            formatted_rows.append(fr)
            for i, cell in enumerate(fr):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build the table string
        def join_row(cells: List[str]) -> str:
            # Re-pad to final widths for alignment
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

        header_line = join_row(headers)
        sep_line = "-" * len(header_line)
        body_lines = [join_row(fr) for fr in formatted_rows]

        return "\n".join([header_line, sep_line, *body_lines])

    def write(self) -> str:
        """Write the analysis table to the output file and return its path."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        table = self.render_table()
        header = (
            "Conciseness by Design Pattern (Corpus B)\n"
            "Lower word counts and words/sentence indicate higher conciseness.\n"
            "Entries that cannot be mapped to a known design pattern are excluded. Case, hyphen,\n"
            "and camel-case variants (e.g., 'facade' vs 'Facade', 'factory-method' vs\n"
            "'FactoryMethod') are merged.\n"
        )
        with open(self.output_path, mode="w", encoding="utf-8") as f:
            f.write(header)
            f.write("\n")
            f.write(table)
            f.write("\n")
        return self.output_path


def _resolve_csv_path(explicit: Optional[str], repo_root: str) -> str:
    """Select the CSV input path, preferring explicit input, then B.csv fallback."""
    if explicit:
        return os.path.abspath(explicit)

    candidate_b = os.path.join(repo_root, "output", "summary-output", "B.csv")
    if os.path.exists(candidate_b):
        return candidate_b

    #candidate_b_v1 = os.path.join(repo_root, "output", "summary-output", "B_v1.csv")
    #if os.path.exists(candidate_b_v1):
    #    return candidate_b_v1

    # Default to B.csv path even if missing so the downstream error remains informative
    return candidate_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze conciseness metrics by design pattern.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Path to corpus B CSV file (defaults to output/summary-output/B.csv with fallback to B_v1.csv)",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional override for output file path.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = _resolve_csv_path(args.csv_path, repo_root)
    out_path = args.output_path or os.path.join(
        repo_root, "evaluation-results", "conciseness_by_pattern.txt"
    )

    analyzer = ConcisenessAnalyzer(csv_path, out_path)
    analyzer.load()
    analyzer.compute()
    output_file = analyzer.write()
    print(f"Wrote conciseness table to: {output_file}")


if __name__ == "__main__":
    main()
