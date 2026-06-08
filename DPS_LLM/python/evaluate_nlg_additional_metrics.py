#!/usr/bin/env python3
"""Evaluate NLG summaries against human references using additional MT metrics.

This script focuses on the NLG pipeline output and computes classic machine
translation metrics that complement the cosine/BERT scores produced by
``evaluate_summaries.py``:

* BLEU-4 (with smoothing)
* NIST (up to 4-grams)
* ROUGE-L (F-measure)
* FrugalScore (if the optional dependency is available)

Results are written to ``evaluation-results/nlg_vs_human_additional_metrics.csv``
with one row per matched class along with an aggregate summary file.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.nist_score import sentence_nist
from rouge_score import rouge_scorer

try:  # Optional Hugging Face evaluate fallback
    import evaluate as hf_evaluate
except Exception:  # pragma: no cover - evaluate is optional
    hf_evaluate = None  # type: ignore[misc]

try:  # FrugalScore is optional but recommended
    from frugalscore import FrugalScore
except Exception:  # pragma: no cover - fallback path when dependency missing
    FrugalScore = None  # type: ignore[misc]


###############################################################################
# Data loading helpers (trimmed versions from evaluate_summaries.py)
###############################################################################


def normalize_project_identifier(value: str) -> str:
    """
    Normalize project identifiers for consistent matching.
    
    Standardizes project names by converting to lowercase, replacing
    whitespace/underscores with hyphens, and removing special characters.
    
    Args:
        value: Raw project identifier string
    
    Returns:
        Normalized lowercase project identifier with hyphens, or empty string if invalid
    
    Examples:
        "Spring Framework" -> "spring-framework"
        "My_Project" -> "my-project"
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s_]+", "-", cleaned)  # Replace spaces/underscores with hyphens
    cleaned = re.sub(r"[^a-z0-9\-]+", "", cleaned)  # Remove special characters
    cleaned = re.sub(r"-+", "-", cleaned)  # Collapse multiple hyphens
    return cleaned


def normalize_filename(value: str) -> str:
    """
    Normalize filenames by lowercasing and removing extensions.
    
    Extracts just the filename from a full path, converts to lowercase,
    and strips the file extension for consistent matching.
    
    Args:
        value: Filename or full file path
    
    Returns:
        Normalized filename without path or extension, or empty string if invalid
    
    Examples:
        "src/Application.java" -> "application"
        "C:\\Project\\File.java" -> "file"
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower().replace("\\", "/").split("/")[-1]  # Extract filename
    cleaned = re.sub(r"\.[^.]+$", "", cleaned)  # Remove extension
    return cleaned


def extract_full_project_path_from_url(url: str, base_project: str) -> str:
    """
    Extract the full project path from a GitHub URL.
    
    Parses GitHub file URLs to extract the complete project path including
    subdirectories, filtering out GitHub-specific components.
    
    Args:
        url: GitHub URL to a file (e.g., https://github.com/user/repo/blob/main/path/File.java)
        base_project: Root project identifier
    
    Returns:
        Full project path with subdirectories (e.g., "user/repo/path")
        Falls back to base_project if parsing fails
    
    Note:
        Removes GitHub-specific path segments like 'blob' and 'main',
        and decodes URL-encoded characters (e.g., %20 -> space).
    """
    try:
        if base_project in url:
            after_project = url.split(base_project, maxsplit=1)[1]
            parts = after_project.strip("/").split("/")
            # Filter out filename and GitHub-specific path components
            filtered = [p for p in parts[:-1] if p and p not in {"blob", "main"}]
            # Decode URL encoding
            filtered = [p.replace("%20", " ") for p in filtered]
            if filtered:
                return f"{base_project}/{'/'.join(filtered)}"
            return base_project
    except Exception:
        pass  # Silently handle parsing errors
    return base_project


def extract_base_project_name(project_path: str) -> str:
    """
    Return the top-level project identifier ignoring nested folders.
    
    Extracts the root project name from a full path, handling both
    slash-separated and underscore-separated path formats.
    
    Args:
        project_path: Full project path (e.g., "user/repo/subfolder")
    
    Returns:
        Base project name (first path segment), or empty string if invalid
    
    Examples:
        "user/repo/subfolder" -> "user"
        "user_repo" -> "user" (treats first underscore as separator)
    
    Note:
        Special handling for underscore-separated paths when no slashes exist.
    """
    if not isinstance(project_path, str):
        return ""
    cleaned = project_path.strip().replace("\\", "/").replace('"', "")  # Normalize separators
    # Treat first underscore as path separator if no slashes present
    if "/" not in cleaned and "_" in cleaned:
        cleaned = cleaned.replace("_", "/", 1)
    segments = cleaned.split("/")
    return segments[0].strip() if segments else cleaned.strip()


def load_human_summaries(csv_path: Path) -> pd.DataFrame:
    """
    Load curated human summaries from CSV file.
    
    Reads and validates the human-written summaries CSV, ensuring all required
    columns are present and extracting project path information from URLs.
    
    Args:
        csv_path: Path to the CSV file containing human summaries
    
    Returns:
        DataFrame with columns: project, base_project, filename, summary
        Rows with empty summaries are filtered out
    
    Raises:
        ValueError: If any required column is missing from the CSV
    
    Note:
        Required columns: Project, File Name, Human Summary, URL
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names

    # Validate presence of required columns
    required_cols = ['Project', 'File Name', 'Human Summary', 'URL']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {csv_path}: {', '.join(missing)}")

    result = df[['Project', 'File Name', 'Human Summary', 'URL']].copy()
    result.columns = ['base_project', 'filename', 'summary', 'url']
    result = result.fillna('')
    result['project'] = result.apply(
        lambda row: extract_full_project_path_from_url(row['url'], row['base_project']),
        axis=1,
    )
    result = result[result['summary'].str.len() > 0]
    return result[['project', 'base_project', 'filename', 'summary']]


def load_nlg_summaries(csv_path: Path) -> pd.DataFrame:
    """Load NLG generated summaries from the project CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    project_col = None
    folder_col = None
    filename_col = None
    summary_col = None

    for col in df.columns:
        low = col.lower()
        if low in {'project', 'project name', 'project_name', 'project path', 'project_path'}:
            project_col = col
        elif low in {'folder', 'folder name', 'folder_name', 'folder path', 'folder_path'}:
            folder_col = col
        elif low in {'filename', 'file name', 'file_name', 'class', 'class name'}:
            filename_col = col
        elif low == 'summary':
            summary_col = col

    if not all([project_col, filename_col, summary_col]):
        raise ValueError(
            f"Unable to locate project/filename/summary columns in {csv_path}. Columns found: {list(df.columns)}"
        )

    columns = [project_col]
    if folder_col:
        columns.append(folder_col)
    columns.extend([filename_col, summary_col])
    result = df[columns].copy()

    rename_map = {
        project_col: 'project',
        filename_col: 'filename',
        summary_col: 'summary',
    }
    if folder_col:
        rename_map[folder_col] = 'folder'

    result = result.rename(columns=rename_map)
    result['project'] = result['project'].astype(str).str.strip()
    result['folder'] = result.get('folder', '').astype(str).str.strip() if folder_col else ''
    result['filename'] = result['filename'].astype(str).str.strip()
    result['summary'] = result['summary'].astype(str).str.strip()
    result['base_project'] = result['project'].apply(extract_base_project_name)
    result = result[result['summary'].str.len() > 0]
    return result[['project', 'base_project', 'filename', 'summary']]


###############################################################################
# Metric helpers
###############################################################################

_smoothing = SmoothingFunction().method3
_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
_frugal = None
_frugal_metric = None
if FrugalScore is not None:  # pragma: no branch - executed when dependency exists
    try:
        _frugal = FrugalScore()
    except Exception as exc:  # pragma: no cover - frugalscore optional failures
        warnings.warn(f"Unable to initialise FrugalScore ({exc}). Metric will be NaN.")
        _frugal = None
elif hf_evaluate is not None:  # pragma: no branch
    try:
        _frugal_metric = hf_evaluate.load("frugalscore")
    except Exception as exc:  # pragma: no cover - evaluate fallback failures
        warnings.warn(f"Unable to load frugalscore metric via evaluate ({exc}). Metric will be NaN.")
        _frugal_metric = None
else:  # pragma: no cover
    warnings.warn("frugalscore package not installed; FrugalScore metric will be NaN.")


def tokenize_for_metrics(text: str) -> List[str]:
    """Simple tokenizer suitable for BLEU/NIST calculations."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


def compute_bleu(reference_tokens: List[str], candidate_tokens: List[str]) -> float:
    if not reference_tokens or not candidate_tokens:
        return 0.0
    try:
        return float(
            sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=_smoothing,
            )
        )
    except Exception:
        return 0.0


def compute_nist(reference_tokens: List[str], candidate_tokens: List[str]) -> float:
    if not reference_tokens or not candidate_tokens:
        return 0.0
    try:
        return float(sentence_nist([reference_tokens], candidate_tokens, n=4))
    except (ZeroDivisionError, ValueError):  # no shared n-grams
        return 0.0


def compute_rouge_l(reference_text: str, candidate_text: str) -> float:
    scores = _rouge.score(reference_text, candidate_text)
    return float(scores['rougeL'].fmeasure)


def compute_frugalscore(reference_text: str, candidate_text: str) -> float:
    if _frugal is not None:
        try:
            score = _frugal.score([candidate_text], [reference_text])
            if isinstance(score, (list, tuple)):
                score = score[0]
            return float(score)
        except Exception as exc:
            warnings.warn(f"FrugalScore failed for a sample ({exc}). Returning NaN.")
            return float('nan')
    if _frugal_metric is not None:
        try:
            result = _frugal_metric.compute(predictions=[candidate_text], references=[reference_text])
            # evaluate returns a dict, extract the first numeric value
            if isinstance(result, dict):
                value = next(iter(result.values()))
                if isinstance(value, (list, tuple)):
                    value = value[0]
                return float(value)
        except Exception as exc:
            warnings.warn(f"Evaluate frugalscore failed for a sample ({exc}). Returning NaN.")
            return float('nan')
    return float('nan')


###############################################################################
# Core evaluation logic
###############################################################################


def match_human_and_nlg(human_df: pd.DataFrame, nlg_df: pd.DataFrame) -> pd.DataFrame:
    """Perform 1-to-1 matching between human and NLG summaries."""
    human_df = human_df.copy()
    nlg_df = nlg_df.copy()

    human_df['normalized_base_project'] = human_df['base_project'].apply(normalize_project_identifier)
    nlg_df['normalized_base_project'] = nlg_df['base_project'].apply(normalize_project_identifier)
    human_df['normalized_filename'] = human_df['filename'].apply(normalize_filename)
    nlg_df['normalized_filename'] = nlg_df['filename'].apply(normalize_filename)

    human_df['match_key'] = human_df['normalized_base_project'] + '::' + human_df['normalized_filename']
    nlg_df['match_key'] = nlg_df['normalized_base_project'] + '::' + nlg_df['normalized_filename']

    human_df['dup_index'] = human_df.groupby('match_key').cumcount()
    nlg_df['dup_index'] = nlg_df.groupby('match_key').cumcount()

    merged = human_df.merge(
        nlg_df,
        on=['match_key', 'dup_index'],
        how='inner',
        suffixes=('_human', '_nlg'),
    )

    merged = merged.drop(columns=['dup_index'])
    if merged.empty:
        raise ValueError(
            "No matching entries between human and NLG summaries. "
            "Ensure the CSV files contain aligned projects and filenames."
        )
    return merged


def compute_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for each matched summary pair."""
    bleu_scores: List[float] = []
    nist_scores: List[float] = []
    rouge_scores: List[float] = []
    frugal_scores: List[float] = []

    for _, row in merged.iterrows():
        reference = row['summary_human']
        candidate = row['summary_nlg']
        ref_tokens = tokenize_for_metrics(reference)
        cand_tokens = tokenize_for_metrics(candidate)

        bleu_scores.append(compute_bleu(ref_tokens, cand_tokens))
        nist_scores.append(compute_nist(ref_tokens, cand_tokens))
        rouge_scores.append(compute_rouge_l(reference, candidate))
        frugal_scores.append(compute_frugalscore(reference, candidate))

    result = pd.DataFrame({
        'base_project': merged['base_project_human'],
        'project_path': merged['project_human'],
        'filename': merged['filename_human'],
        'summary_human': merged['summary_human'],
        'summary_nlg': merged['summary_nlg'],
        'bleu_4': bleu_scores,
        'nist_4': nist_scores,
        'rouge_l_f': rouge_scores,
        'frugal_score': frugal_scores,
    })
    return result


def summarise_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Generate aggregate statistics for the computed metrics."""
    metrics = ['bleu_4', 'nist_4', 'rouge_l_f', 'frugal_score']
    summary_rows = []
    for metric in metrics:
        values = results[metric].astype(float).to_numpy()
        if values.size == 0 or np.all(np.isnan(values)):
            summary_rows.append({
                'metric': metric,
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
            })
            continue
        summary_rows.append({
            'metric': metric,
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
            'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)),
        })
    return pd.DataFrame(summary_rows)


###############################################################################
# Entry point
###############################################################################


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NLG summaries with additional MT metrics.")
    parser.add_argument(
        '--human-csv',
        type=Path,
        default=Path('input/DPS_Human_Summaries.csv'),
        help='Path to the curated human summary CSV.',
    )
    parser.add_argument(
        '--nlg-csv',
        type=Path,
        default=Path('output/summary-output/nlg_summaries.csv'),
        help='Path to the NLG summaries CSV output.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation-results'),
        help='Directory where evaluation artefacts should be written.',
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("NLG vs Human: BLEU / NIST / ROUGE-L / FrugalScore")
    print("============================================================")

    print(f"Loading human summaries from {args.human_csv} ...")
    human_df = load_human_summaries(args.human_csv)
    print(f"  Loaded {len(human_df)} human summaries")

    print(f"Loading NLG summaries from {args.nlg_csv} ...")
    nlg_df = load_nlg_summaries(args.nlg_csv)
    print(f"  Loaded {len(nlg_df)} NLG summaries")

    merged = match_human_and_nlg(human_df, nlg_df)
    print(f"Matched {len(merged)} pairs for evaluation")

    print("Computing metrics ...")
    pairwise_results = compute_metrics(merged)

    output_csv = args.output_dir / 'nlg_vs_human_additional_metrics.csv'
    pairwise_results.to_csv(output_csv, index=False)
    print(f"Saved per-class metrics to {output_csv}")

    summary_df = summarise_metrics(pairwise_results)
    summary_csv = args.output_dir / 'nlg_vs_human_additional_metrics_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved metric summary to {summary_csv}")

    print("\nAggregate metrics:")
    for _, row in summary_df.iterrows():
        metric = row['metric']
        mean = row['mean']
        std = row['std']
        min_val = row['min']
        max_val = row['max']
        print(f"  {metric:12s} mean={mean:.4f} std={std:.4f} min={min_val:.4f} max={max_val:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
