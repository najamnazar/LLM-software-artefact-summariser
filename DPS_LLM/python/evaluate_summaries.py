#!/usr/bin/env python3
"""
Summary Evaluation Against Human Summaries
Evaluates NLG, SWUM, and multiple LLM summaries against human-written summaries
using Cosine Similarity and BERTScore with violin plot visualizations.

This module now follows an object-oriented design to encapsulate the data-loading, evaluation,
visualisation, and orchestration responsibilities while retaining the CLI behaviour and outputs.
"""

import argparse
from datetime import datetime
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

try:
    from dotenv import load_dotenv
    # Load .env file from repo root (two directories up from python/)
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv not installed, will use environment variables directly

# Configure plot style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Silence noisy transformer weight-init warnings from HF
hf_logging.set_verbosity_error()


def normalize_project_identifier(value: str) -> str:
    """Normalise a project name for key matching.

    Lowercases, strips whitespace, and removes every non-alphanumeric character
    (hyphens, underscores, spaces) so that 'AbdurRKhalid', 'abdur-r-khalid', and
    'abdur_r_khalid' all collapse to the same canonical key.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def normalize_filename(value: str) -> str:
    """Normalise a file-name for key matching.

    Strips the directory path, lowercases the bare filename, and removes the
    file extension so that 'src/Foo.java', 'Foo.java', and 'foo' all map to 'foo'.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower().replace('\\', '/').split('/')[-1]
    return re.sub(r"\.[^.]+$", "", cleaned)


def extract_full_project_path_from_url(url: str, base_project: str) -> str:
    """Derive a full project path from a GitHub blob URL.

    Walks the URL segments after the base project name, drops VCS artefacts
    ('blob', 'main'), and reconstructs a slash-separated path.  Falls back to
    ``base_project`` when the URL cannot be parsed.
    """
    try:
        if base_project in url:
            after_project = url.split(base_project)[1]
            parts = after_project.strip('/').split('/')
            path_parts = [p for p in parts[:-1] if p and p not in {'blob', 'main'}]
            path_parts = [p.replace('%20', ' ') for p in path_parts]
            if path_parts:
                return f"{base_project}/{'/'.join(path_parts)}"
            return base_project
    except Exception:
        pass
    return base_project


def extract_base_project_name(project_path: str) -> str:
    """Return the top-level (owner/repository) segment from a project path.

    Handles both slash-separated paths ('AbdurRKhalid/Observer') and legacy
    underscore-separated values ('AbdurRKhalid_Observer') by treating the first
    underscore as a path separator when no slash is present.
    """
    if not isinstance(project_path, str):
        return ""
    cleaned = project_path.strip().replace('\\', '/').replace('"', '')
    if '/' not in cleaned and '_' in cleaned:
        cleaned = cleaned.replace('_', '/', 1)
    segments = cleaned.split('/')
    return segments[0].strip() if segments else cleaned.strip()


def canonicalize_design_pattern(raw_value: str) -> str:
    """Map free-form pattern labels into canonical design pattern names.

    This mirrors the normalization intent already used in conciseness analysis,
    but is kept local to this evaluator so pair-level scoring can be grouped by
    design pattern without changing any existing class/project computations.
    """
    if not isinstance(raw_value, str):
        return 'Unknown'

    token = raw_value.strip()
    if not token:
        return 'Unknown'

    # Normalize separators/case first, then map common aliases.
    norm = re.sub(r"[\s_\-/]+", " ", token.lower()).strip()
    patterns = {
        'abstract factory': 'Abstract Factory',
        'abstractfactory': 'Abstract Factory',
        'adapter': 'Adapter',
        'decorator': 'Decorator',
        'facade': 'Facade',
        'factory method': 'Factory Method',
        'factorymethod': 'Factory Method',
        'memento': 'Memento',
        'observer': 'Observer',
        'singleton': 'Singleton',
        'visitor': 'Visitor',
    }

    if norm in patterns:
        return patterns[norm]

    # CamelCase fallback (e.g., AbstractFactory -> abstract factory).
    camel = re.sub(r"(?<!^)(?=[A-Z])", " ", token).lower().strip()
    camel = re.sub(r"[\s_\-/]+", " ", camel).strip()
    if camel in patterns:
        return patterns[camel]

    return 'Unknown'


class MetricsCalculator:
    """Stateless helper that computes text-similarity metrics for summary pairs.

    All methods are static so the class acts as a namespace rather than requiring
    instantiation — callers can inject the class itself or a subclass for testing.
    """

    @staticmethod
    def cosine_similarity(text_a: str, text_b: str) -> float:
        """Compute TF-IDF cosine similarity between two text strings.

        Fits a fresh TfidfVectorizer on the pair to avoid vocabulary leakage
        from other pairs.  Returns 0.0 when either string is empty or contains
        only stop-words (sklearn raises ValueError in that case).
        """
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform([text_a, text_b])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
            return float(score)
        except ValueError:
            return 0.0

    @staticmethod
    def bert_scores(candidates: List[str], references: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute BERTScore precision, recall, and F1 for a batch of text pairs.

        Uses the default ``roberta-large`` model with English language settings.
        Baseline rescaling is disabled so raw contextual-embedding cosine distances
        are returned directly.  Raises RuntimeError (wrapping the underlying
        exception) on failure so callers can handle it uniformly.
        """
        try:
            precision, recall, f1 = bert_score(
                candidates,
                references,
                lang='en',
                rescale_with_baseline=False,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - propagate with context
            raise RuntimeError("BERTScore computation failed") from exc
        return precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy()


class SummaryDataLoader:
    """Reads and normalises the CSV inputs for human and generated summaries.

    Both loaders return a DataFrame with consistent column names so the
    evaluator can merge them without knowing the original column layout.
    """

    @staticmethod
    def load_human_summaries(csv_path: Path) -> pd.DataFrame:
        """Load the ground-truth human summary CSV.

        Expected columns: Project, File Name, Human Summary, URL, and
        optionally Design Pattern (used as the folder key for matching).
        Returns a DataFrame with columns: project, base_project, filename,
        folder, summary.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Human summaries file not found: {csv_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Human summaries file is empty: {csv_path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read human summaries: {csv_path}") from exc

        try:
            df.columns = df.columns.str.strip()
            required_cols = ['Project', 'File Name', 'Human Summary', 'URL']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required column(s) {missing} in {csv_path}")

            # Include Design Pattern column if available for folder matching
            cols_to_load = ['Project', 'File Name', 'Human Summary', 'URL']
            if 'Design Pattern' in df.columns:
                cols_to_load.append('Design Pattern')
                
            result = df[cols_to_load].copy()
            result.columns = ['base_project', 'filename', 'summary', 'url'] + (['design_pattern'] if 'Design Pattern' in df.columns else [])
            
            for column in ['base_project', 'filename', 'summary', 'url']:
                result[column] = result[column].astype(str).str.strip()

            result['project'] = result.apply(
                lambda row: extract_full_project_path_from_url(row['url'], row['base_project']),
                axis=1,
            )
            
            # Extract folder from Design Pattern column if available
            if 'design_pattern' in result.columns:
                result['folder'] = result['design_pattern'].astype(str).str.strip()
            else:
                result['folder'] = ''
            
            result = result[result['summary'].str.len() > 0]
            return result[['project', 'base_project', 'filename', 'folder', 'summary']]
        except Exception:
            raise

    @staticmethod
    def load_generated_summaries(csv_path: Path, summary_col_name: str) -> pd.DataFrame:
        """Load a generated-summary CSV produced by any summarisation method.

        Detects the project, folder, filename, and summary columns by matching
        against a set of known synonyms, so the same loader works for NLG,
        SWUM, and every LLM variant.  ``summary_col_name`` is the exact or
        case-insensitive name of the column that holds the summary text.
        Returns a DataFrame with columns: project, base_project, folder,
        filename, summary.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Generated summaries file not found: {csv_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Generated summaries file is empty: {csv_path}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read generated summaries: {csv_path}") from exc

        try:
            df.columns = df.columns.str.strip()
            project_col = folder_col = filename_col = summary_col = None

            for col in df.columns:
                col_lower = col.lower()
                if col_lower in {'project', 'project name', 'project_name', 'project path', 'project_path'}:
                    project_col = col
                elif col_lower in {'folder', 'folder name', 'folder_name', 'folder path', 'folder_path'}:
                    folder_col = col
                elif col_lower in {'filename', 'file name', 'file_name', 'class', 'class name'}:
                    filename_col = col
                elif col_lower == summary_col_name.lower() or col == summary_col_name:
                    summary_col = col

            if not all([project_col, filename_col, summary_col]):
                available = ', '.join(df.columns)
                raise ValueError(
                    f"Cannot find required columns in {csv_path}. Available columns: {available}"
                )

            columns = [project_col]
            if folder_col:
                columns.append(folder_col)
            columns.extend([filename_col, summary_col])

            result = df[columns].copy()
            rename_map = {project_col: 'project', filename_col: 'filename', summary_col: 'summary'}
            if folder_col:
                rename_map[folder_col] = 'folder'
            result = result.rename(columns=rename_map)

            result['project'] = result['project'].astype(str).str.strip()
            if 'folder' in result.columns:
                result['folder'] = result['folder'].astype(str).str.strip()
            else:
                result['folder'] = ''
            result['filename'] = result['filename'].astype(str).str.strip()
            result['summary'] = result['summary'].astype(str).str.strip()
            result['base_project'] = result['project'].apply(extract_base_project_name)
            result = result[result['summary'].str.len() > 0]
            return result
        except Exception:
            raise


@dataclass
class MethodEvaluationResult:
    """Bundles every artefact produced for a single summarisation method.

    Attributes:
        method: Short display name used as a CSV/plot label (e.g. 'LLM (Claude NC)').
        metrics: Corpus-level scalar scores (avg cosine, avg BERT F1, std, …).
        project_stats: Per-project aggregated scores DataFrame.
        pattern_stats: Per-design-pattern aggregated scores DataFrame.
        pattern_metrics: Macro/micro averages across design patterns.
        merged: The inner-joined DataFrame of (human, method) pairs with all
                per-class scores attached — used directly for violin plots.
    """

    method: str
    metrics: Dict[str, float]
    project_stats: pd.DataFrame
    pattern_stats: pd.DataFrame
    pattern_metrics: Dict[str, float]
    merged: pd.DataFrame


@dataclass
class EvaluationConfig:
    """Strongly-typed holder for all CLI inputs passed to the pipeline.

    Mandatory paths (human CSV, NLG, SWUM) are positional-style required
    fields.  Every LLM CSV is optional; absent ones are silently skipped by
    ``method_sources``.  NC (Narrative Context) variants follow the same
    naming convention as their base counterparts with an '_nc' suffix.
    """

    human_csv: Path
    nlg_csv: Path
    swum_csv: Path
    output_dir: Path
    llm_claude_csv: Optional[Path] = None
    llm_gemini_csv: Optional[Path] = None
    llm_gpt_csv: Optional[Path] = None
    llm_mistral_csv: Optional[Path] = None
    llm_qwen_csv: Optional[Path] = None
    dps_llm_csv: Optional[Path] = None        # Legacy Mixtral summaries
    dps_llm_nonconcise_csv: Optional[Path] = None  # Legacy non-concise alias
    llm_claude_nc_csv: Optional[Path] = None  # Claude with Narrative Context
    llm_gpt_nc_csv: Optional[Path] = None     # GPT with Narrative Context
    llm_mistral_nc_csv: Optional[Path] = None # Mistral with Narrative Context
    llm_qwen_nc_csv: Optional[Path] = None    # Qwen with Narrative Context

    def ensure_output_dir(self) -> None:
        """Create the output directory (and any parents) if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def method_sources(self) -> List[Tuple[str, Path]]:
        """Return an ordered list of (display-name, csv-path) tuples.

        NLG and SWUM are always first.  LLM variants (including NC) are
        appended only when their path is not None, preserving a deterministic
        order for violin-plot x-axis labels.
        """
        sources = [
            ('NLG', self.nlg_csv),
            ('SWUM', self.swum_csv),
        ]

        if self.llm_claude_csv is not None:
            sources.append(('LLM (Claude)', self.llm_claude_csv))
        if self.llm_gemini_csv is not None:
            sources.append(('LLM (Gemini)', self.llm_gemini_csv))
        if self.llm_gpt_csv is not None:
            sources.append(('LLM (GPT)', self.llm_gpt_csv))
        if self.llm_mistral_csv is not None:
            sources.append(('LLM (Mistral)', self.llm_mistral_csv))
        if self.llm_qwen_csv is not None:
            sources.append(('LLM (Qwen)', self.llm_qwen_csv))
        if self.dps_llm_csv is not None:
            sources.append(('LLM (Mixtral)', self.dps_llm_csv))
        if self.dps_llm_nonconcise_csv is not None:
            sources.append(('LLM (Non-Concise 50)', self.dps_llm_nonconcise_csv))
        # NC (Narrative Context) variants — evaluated separately in violin plots
        if self.llm_claude_nc_csv is not None:
            sources.append(('LLM (Claude NC)', self.llm_claude_nc_csv))
        if self.llm_gpt_nc_csv is not None:
            sources.append(('LLM (GPT NC)', self.llm_gpt_nc_csv))
        if self.llm_mistral_nc_csv is not None:
            sources.append(('LLM (Mistral NC)', self.llm_mistral_nc_csv))
        if self.llm_qwen_nc_csv is not None:
            sources.append(('LLM (Qwen NC)', self.llm_qwen_nc_csv))
        return sources


class SummaryEvaluator:
    """Scores generated summaries against human ground-truth for one method.

    The evaluator holds a fixed copy of the human-summary DataFrame and
    reuses the same ``MetricsCalculator`` instance for every method it
    evaluates, keeping model weights loaded between calls for speed.
    """

    def __init__(self, human_df: pd.DataFrame, output_dir: Path, metrics: MetricsCalculator) -> None:
        self.human_df = human_df.copy()
        self.output_dir = output_dir
        self.metrics = metrics

    def evaluate_method(
        self,
        method_name: str,
        method_df: pd.DataFrame,
        display_name: Optional[str] = None,
    ) -> Optional[MethodEvaluationResult]:
        """Score one set of generated summaries against the human ground truth.

        Normalises both DataFrames, builds a compound match key
        (base_project::folder::filename), performs a 1-to-1 inner join,
        then computes cosine similarity and BERTScore for every matched
        pair.  Saves three CSVs (class-level, project-level, pattern-level)
        and returns a ``MethodEvaluationResult`` or None if no pairs matched.
        """
        friendly_name = display_name or method_name
        print(f"\n{'='*60}")
        print(f"Evaluating {friendly_name} vs Human Summaries")
        print(f"{'='*60}")

        human_df = self.human_df.copy()
        method_df = method_df.copy()

        human_df['normalized_base_project'] = human_df['base_project'].apply(normalize_project_identifier)
        method_df['normalized_base_project'] = method_df['base_project'].apply(normalize_project_identifier)
        human_df['normalized_filename'] = human_df['filename'].apply(normalize_filename)
        method_df['normalized_filename'] = method_df['filename'].apply(normalize_filename)
        
        # Include folder information in match key for better duplicate handling
        if 'folder' in human_df.columns:
            human_df['normalized_folder'] = human_df['folder'].fillna('').apply(lambda x: normalize_project_identifier(str(x)) if x else '')
        else:
            human_df['normalized_folder'] = ''
            
        if 'folder' in method_df.columns:
            method_df['normalized_folder'] = method_df['folder'].fillna('').apply(lambda x: normalize_project_identifier(str(x)) if x else '')
        else:
            method_df['normalized_folder'] = ''
        
        human_df['match_key'] = human_df['normalized_base_project'] + '::' + human_df['normalized_folder'] + '::' + human_df['normalized_filename']
        method_df['match_key'] = method_df['normalized_base_project'] + '::' + method_df['normalized_folder'] + '::' + method_df['normalized_filename']
        human_df['dup_index'] = human_df.groupby('match_key').cumcount()
        method_df['dup_index'] = method_df.groupby('match_key').cumcount()

        print(f"  Human summaries: {len(human_df)}")
        print(f"  {friendly_name} summaries: {len(method_df)}")

        human_dupes = human_df[human_df.duplicated('match_key', keep=False)]
        method_dupes = method_df[method_df.duplicated('match_key', keep=False)]
        if not human_dupes.empty:
            print(f"  WARNING: Found {len(human_dupes)} duplicate entries in human summaries:")
            for key in human_dupes['match_key'].unique():
                print(f"    - {key}")
        if not method_dupes.empty:
            print(f"  WARNING: Found {len(method_dupes)} duplicate entries in {friendly_name} summaries:")
            for key in method_dupes['match_key'].unique():
                print(f"    - {key}")

        merged = human_df.merge(
            method_df,
            on=['match_key', 'dup_index'],
            how='inner',
            suffixes=('_human', '_method'),
        ).drop(columns=['dup_index'])

        if merged.empty:
            print(f"  ERROR: No matching entries found for {friendly_name}")
            print(f"  Sample human keys: {list(human_df['match_key'].head())}")
            print(f"  Sample {friendly_name} keys: {list(method_df['match_key'].head())}")
            return None

        print(f"  Matched: {len(merged)} (exact 1-to-1 pairs)")

        merged['cosine_similarity'] = merged.apply(
            lambda row: self.metrics.cosine_similarity(row['summary_human'], row['summary_method']),
            axis=1,
        )

        candidates = merged['summary_method'].tolist()
        references = merged['summary_human'].tolist()
        print(f"  Computing BERTScore for {len(candidates)} pairs (this may take 2-3 minutes)...")
        try:
            bert_p, bert_r, bert_f1 = self.metrics.bert_scores(candidates, references)
            print(f"  BERTScore computation complete")
        except RuntimeError as exc:
            print(f"  ERROR: {exc}")
            return None

        merged['bert_precision'] = bert_p
        merged['bert_recall'] = bert_r
        merged['bert_f1'] = bert_f1

        class_csv = self.output_dir / f'{method_name.lower()}_vs_human_class_scores.csv'
        output_df = pd.DataFrame(
            {
                'base_project': merged['base_project_human'],
                'project_path': merged['project_human'],
                'filename': merged['filename_human'],
                'summary_human': merged['summary_human'],
                'summary_method': merged['summary_method'],
                'cosine_similarity': merged['cosine_similarity'],
                'bert_precision': merged['bert_precision'],
                'bert_recall': merged['bert_recall'],
                'bert_f1': merged['bert_f1'],
            }
        )
        output_df.to_csv(class_csv, index=False)
        print(f"  Saved: {class_csv}")

        project_stats = merged.groupby('base_project_human').agg(
            classes=('filename_human', 'count'),
            avg_cosine=('cosine_similarity', 'mean'),
            avg_bert_precision=('bert_precision', 'mean'),
            avg_bert_recall=('bert_recall', 'mean'),
            avg_bert_f1=('bert_f1', 'mean'),
        ).reset_index()
        project_stats.rename(columns={'base_project_human': 'project'}, inplace=True)
        # Combined BERT+cosine score generation disabled per request.
        # project_stats['combined_score'] = (
        #     project_stats['avg_cosine'] + project_stats['avg_bert_f1']
        # ) / 2.0

        project_csv = self.output_dir / f'{method_name.lower()}_vs_human_project_scores.csv'
        # Keep deterministic ordering without the removed combined metric.
        project_stats.sort_values('avg_bert_f1', ascending=False).to_csv(project_csv, index=False)
        print(f"  Saved: {project_csv}")

        # Per-pattern aggregation extends existing outputs without replacing them.
        # Prefer human folder labels for grouping; fall back to method-side labels.
        merged['design_pattern'] = merged.apply(
            lambda row: canonicalize_design_pattern(
                row.get('folder_human', '') or row.get('folder_method', '')
            ),
            axis=1,
        )

        pattern_stats = merged.groupby('design_pattern').agg(
            matched_pairs=('filename_human', 'count'),
            classes=('filename_human', pd.Series.nunique),
            avg_cosine=('cosine_similarity', 'mean'),
            cosine_std=('cosine_similarity', lambda s: float(s.std(ddof=0))),
            avg_bert_precision=('bert_precision', 'mean'),
            avg_bert_recall=('bert_recall', 'mean'),
            avg_bert_f1=('bert_f1', 'mean'),
            bert_f1_std=('bert_f1', lambda s: float(s.std(ddof=0))),
        ).reset_index()
        # Combined BERT+cosine score generation disabled per request.
        # pattern_stats['combined_score'] = (
        #     pattern_stats['avg_cosine'] + pattern_stats['avg_bert_f1']
        # ) / 2.0
        pattern_stats = pattern_stats.sort_values('avg_bert_f1', ascending=False)

        pattern_metrics = {
            # Macro = unweighted average across patterns.
            'macro_avg_cosine': float(pattern_stats['avg_cosine'].mean()) if not pattern_stats.empty else 0.0,
            'macro_avg_bert_f1': float(pattern_stats['avg_bert_f1'].mean()) if not pattern_stats.empty else 0.0,
            # Micro = weighted by matched pairs (equivalent to corpus-level means).
            'micro_avg_cosine': float(merged['cosine_similarity'].mean()) if not merged.empty else 0.0,
            'micro_avg_bert_f1': float(merged['bert_f1'].mean()) if not merged.empty else 0.0,
            'patterns_evaluated': int(pattern_stats.shape[0]),
        }
        # Combined BERT+cosine score generation disabled per request.
        # pattern_metrics['macro_combined_score'] = (
        #     pattern_metrics['macro_avg_cosine'] + pattern_metrics['macro_avg_bert_f1']
        # ) / 2.0
        # pattern_metrics['micro_combined_score'] = (
        #     pattern_metrics['micro_avg_cosine'] + pattern_metrics['micro_avg_bert_f1']
        # ) / 2.0

        pattern_csv = self.output_dir / f'{method_name.lower()}_vs_human_pattern_scores.csv'
        pattern_stats.to_csv(pattern_csv, index=False)
        print(f"  Saved: {pattern_csv}")

        overall_metrics = {
            'method': method_name,
            'projects_evaluated': int(project_stats.shape[0]),
            'classes_evaluated': int(len(merged)),
            'avg_cosine': float(merged['cosine_similarity'].mean()),
            'avg_bert_precision': float(merged['bert_precision'].mean()),
            'avg_bert_recall': float(merged['bert_recall'].mean()),
            'avg_bert_f1': float(merged['bert_f1'].mean()),
            # 'combined_score': float(
            #     (merged['cosine_similarity'].mean() + merged['bert_f1'].mean()) / 2.0
            # ),
            'cosine_std': float(merged['cosine_similarity'].std(ddof=0)),
            'bert_f1_std': float(merged['bert_f1'].std(ddof=0)),
        }

        print(f"\n{friendly_name} Results:")
        print(f"  Classes evaluated: {overall_metrics['classes_evaluated']}")
        print(f"  Avg Cosine Similarity: {overall_metrics['avg_cosine']:.4f}")
        print(f"  Avg BERT Precision: {overall_metrics['avg_bert_precision']:.4f}")
        print(f"  Avg BERT Recall: {overall_metrics['avg_bert_recall']:.4f}")
        print(f"  Avg BERT F1: {overall_metrics['avg_bert_f1']:.4f}")
        # print(f"  Combined Score: {overall_metrics['combined_score']:.4f}")

        return MethodEvaluationResult(
            method=method_name,
            metrics=overall_metrics,
            project_stats=project_stats,
            pattern_stats=pattern_stats,
            pattern_metrics=pattern_metrics,
            merged=merged,
        )


_NC_SUFFIX = ' NC'
_NC_METHODS = {'LLM (Claude NC)', 'LLM (GPT NC)', 'LLM (Mistral NC)', 'LLM (Qwen NC)'}
_NON_NC_LLM_METHODS = {'LLM (Claude)', 'LLM (GPT)', 'LLM (Mistral)', 'LLM (Qwen)'}

_METHOD_COLORS = {
    'NLG': '#e74c3c',
    'SWUM': '#3498db',
    'LLM (Claude)': '#16a085',
    'LLM (Gemini)': '#8e44ad',
    'LLM (GPT)': '#d35400',
    'LLM (Mistral)': '#2c3e50',
    'LLM (Qwen)': '#27ae60',
    'LLM (Mixtral)': '#f39c12',
    'LLM (Non-Concise 50)': '#7f8c8d',
    # NC variants — lighter/complementary shades of the base model colour
    'LLM (Claude NC)': '#1abc9c',
    'LLM (GPT NC)': '#e67e22',
    'LLM (Mistral NC)': '#7f8c8d',
    'LLM (Qwen NC)': '#2ecc71',
}

# Short-label colour map used by plots that display model names without the 'LLM (…)' wrapper.
# NC and non-NC variants of the same model share a colour because their labels are identical.
_SHORT_LABEL_COLORS = {
    'NLG': '#e74c3c',
    'SWUM': '#3498db',
    'Claude': '#16a085',
    'GPT': '#d35400',
    'Mistral': '#2c3e50',
    'Qwen': '#27ae60',
    'Gemini': '#8e44ad',
    'Mixtral': '#f39c12',
}


class VisualizationManager:
    """Produces violin-plot figures for every combination of evaluation results.

    Four figures are generated when both NC and non-NC LLM results are present:
      1. ``concise_all_methods_violin.png``  — NLG, SWUM, and the 4 concise LLMs.
      2. ``nc_all_methods_violin.png``       — NLG, SWUM, and the 4 NC LLMs
                                              (same short model names; title marks NC).
      3. ``nc_llm_only_violin.png``          — 4 NC LLMs only, no baselines.
      4. ``concise_llm_only_violin.png``     — 4 concise LLMs only, no baselines.

    Each figure shows cosine similarity on the left and BERTScore F1 on the
    right, with a mean-marker (diamond) overlaid on every violin.
    """

    def __init__(self, metrics: MetricsCalculator) -> None:
        self.metrics = metrics

    def create_visualizations(
        self,
        results: List[MethodEvaluationResult],
        output_dir: Path,
    ) -> None:
        """Dispatch all four violin-plot figures for the full result set.

        Separates results into NLG/SWUM baselines, non-NC concise LLMs, and NC
        LLMs, then calls ``_create_labeled_violin`` four times with the
        appropriate groupings and titles.
        """
        if len(results) < 2:
            print("Insufficient data for comparison visualization (need at least 2 methods)")
            return

        nc_results = [r for r in results if r.method in _NC_METHODS]
        non_nc_results = [r for r in results if r.method in _NON_NC_LLM_METHODS]
        baseline_results = [r for r in results if r.method in {'NLG', 'SWUM'}]

        def _short(method: str) -> str:
            """Strip 'LLM (' prefix, trailing ')', and ' NC' suffix for display."""
            return method.replace('LLM (', '').rstrip(')').replace(' NC', '')

        # Plot 1: NLG + SWUM + 4 concise LLMs
        plot1 = baseline_results + non_nc_results
        if plot1:
            self._create_labeled_violin(
                plot1, output_dir,
                title='Concise Summaries Evaluation: Score Distributions',
                filename='concise_all_methods_violin.png',
                label_fn=_short,
            )

        # Plot 2: NLG + SWUM + 4 NC LLMs — same short model names, different title
        plot2 = baseline_results + nc_results
        if plot2:
            self._create_labeled_violin(
                plot2, output_dir,
                title='Non-Concise Summaries Evaluation: Score Distributions',
                filename='nc_all_methods_violin.png',
                label_fn=_short,
            )

        # Plot 3: 4 NC LLMs only (no baselines), same short names and title as plot 2
        if nc_results:
            self._create_labeled_violin(
                nc_results, output_dir,
                title='Non-Concise Summaries Evaluation: Score Distributions',
                filename='nc_llm_only_violin.png',
                label_fn=_short,
            )

        # Plot 4: 4 concise LLMs only (no baselines)
        if non_nc_results:
            self._create_labeled_violin(
                non_nc_results, output_dir,
                title='Concise Summaries Evaluation: Score Distributions',
                filename='concise_llm_only_violin.png',
                label_fn=_short,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_mean_markers(ax: plt.Axes, all_data: pd.DataFrame, method_order: List[str], metric: str) -> None:
        """Overlay a red diamond at the mean value on each violin."""
        for idx, method in enumerate(method_order):
            mean_val = all_data.loc[all_data['method'] == method, metric].mean()
            ax.plot(
                idx,
                mean_val,
                marker='D',
                markersize=8,
                color='darkred',
                zorder=3,
                label='Mean' if idx == 0 else '',
            )
        ax.legend(loc='upper left')

    @staticmethod
    def _violin_subplot(
        ax: plt.Axes,
        data: pd.DataFrame,
        metric: str,
        method_order: List[str],
        title: str,
        ylabel: str,
        palette: Optional[dict],
    ) -> None:
        """Draw a single violin plot panel on ``ax`` for the given metric."""
        sns.violinplot(
            data=data,
            x='method',
            y=metric,
            ax=ax,
            order=method_order,
            palette=palette,
            hue='method',
            legend=False,
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        VisualizationManager._add_mean_markers(ax, data, method_order, metric)

    def _create_labeled_violin(
        self,
        results: List[MethodEvaluationResult],
        output_dir: Path,
        title: str,
        filename: str,
        label_fn,
    ) -> None:
        """Save a 1×2 violin figure with custom display labels derived from ``label_fn``.

        ``label_fn`` maps an internal method name (e.g. 'LLM (Claude NC)') to
        the short x-axis label (e.g. 'Claude').  NC and non-NC variants of the
        same model share a colour via ``_SHORT_LABEL_COLORS`` when their labels
        are identical.
        """
        rows, order = [], []
        for r in results:
            if not r.merged.empty:
                subset = r.merged[['cosine_similarity', 'bert_f1']].copy()
                label = label_fn(r.method)
                subset['method'] = label
                rows.append(subset)
                if label not in order:
                    order.append(label)

        if not rows:
            print(f"Skipping {filename}: no data available")
            return

        data = pd.concat(rows, ignore_index=True)
        palette = {lbl: _SHORT_LABEL_COLORS[lbl] for lbl in order if lbl in _SHORT_LABEL_COLORS}

        n_methods = len(order)
        fig, axes = plt.subplots(1, 2, figsize=(max(12, n_methods * 2.5), 6))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        self._violin_subplot(
            axes[0], data, 'cosine_similarity', order,
            'Cosine Similarity', 'Cosine Similarity Score', palette or None,
        )
        self._violin_subplot(
            axes[1], data, 'bert_f1', order,
            'BERTScore F1', 'BERTScore F1 Score', palette or None,
        )

        plt.tight_layout()
        out_file = output_dir / filename
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_file}")
        plt.close()


class SummaryEvaluationPipeline:
    """Orchestrates the complete evaluation workflow from CSV loading to output.

    Responsibilities in order:
      1. Load human ground-truth summaries.
      2. Iterate over every configured method (NLG, SWUM, LLM variants, NC
         variants), load their summaries, and score them via ``SummaryEvaluator``.
      3. Persist per-class, per-project, and per-pattern score CSVs.
      4. Write an overall comparison CSV and a human-readable summary text file.
      5. Delegate violin-plot generation to ``VisualizationManager``.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.loader = SummaryDataLoader()
        self.metrics = MetricsCalculator()
        self.visualizer = VisualizationManager(self.metrics)

    def run(self) -> None:
        """Execute the full evaluation pipeline end to end."""
        print(f"\n{'='*60}")
        print("Summary Evaluation Against Human Summaries")
        print(f"{'='*60}")
        print("\nInitializing (loading ML models - this takes 30-60 seconds)...")

        self.config.ensure_output_dir()
        human_df = self._load_human_data()
        if human_df is None:
            return
        print(f"Loaded {len(human_df)} human summaries")

        evaluator = SummaryEvaluator(human_df, self.config.output_dir, self.metrics)
        results: List[MethodEvaluationResult] = []

        for method_name, csv_path in self.config.method_sources():
            display_name = method_name

            if not csv_path.exists():
                print(f"WARNING: {display_name} CSV not found: {csv_path}")
                continue

            try:
                method_df = self.loader.load_generated_summaries(csv_path, 'Summary')
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR loading {display_name} summaries: {exc}")
                continue

            print(f"\nLoading {display_name} summaries from: {csv_path}")
            print(f"Loaded {len(method_df)} {display_name} summaries")

            result = evaluator.evaluate_method(method_name, method_df, display_name)
            if result is None:
                continue

            results.append(result)

        if not results:
            print("\nNo evaluation results generated.")
            return

        self._save_overall_comparison(results)
        self._save_pattern_overall_comparison(results)
        print("\nCreating violin plot visualizations (30 seconds)...")
        self.visualizer.create_visualizations(results, self.config.output_dir)
        print("Visualizations complete")
        self._write_summary_files(results)
        self._print_completion(results)

    def _load_human_data(self) -> Optional[pd.DataFrame]:
        """Load and return the human-summary DataFrame, or None on failure."""
        print(f"\nLoading human summaries from: {self.config.human_csv}")
        try:
            return self.loader.load_human_summaries(self.config.human_csv)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR loading human summaries: {exc}")
            return None

    def _save_overall_comparison(self, results: List[MethodEvaluationResult]) -> None:
        """Write one row per method with corpus-level scores to overall_comparison.csv."""
        overall_df = pd.DataFrame(result.metrics for result in results)
        overall_csv = self.config.output_dir / 'overall_comparison.csv'
        overall_df.to_csv(overall_csv, index=False)
        print(f"\nSaved overall comparison: {overall_csv}")

    def _save_pattern_overall_comparison(self, results: List[MethodEvaluationResult]) -> None:
        """Persist a cross-method per-design-pattern table for side-by-side comparison."""
        all_pattern_rows: List[pd.DataFrame] = []
        for result in results:
            if result.pattern_stats.empty:
                continue
            df = result.pattern_stats.copy()
            df.insert(0, 'method', result.method)
            all_pattern_rows.append(df)

        if not all_pattern_rows:
            print("No per-pattern rows generated; skipping pattern_overall_comparison.csv")
            return

        pattern_overall = pd.concat(all_pattern_rows, ignore_index=True)
        out_csv = self.config.output_dir / 'pattern_overall_comparison.csv'
        pattern_overall.to_csv(out_csv, index=False)
        print(f"Saved per-pattern comparison: {out_csv}")

    def _write_summary_files(self, results: List[MethodEvaluationResult]) -> None:
        """Write evaluation_summary.txt (overwrite) and append to results.txt.

        ``evaluation_summary.txt`` — concise per-method scores for quick review.
        ``results.txt``            — append-mode log with timestamped corpus-level
                                     and per-design-pattern breakdowns.
        """
        summary_file = self.config.output_dir / 'evaluation_summary.txt'
        results_file = self.config.output_dir / 'results.txt'

        try:
            with open(summary_file, 'w', encoding='utf-8') as handle:
                handle.write("=" * 60 + "\n")
                handle.write("EVALUATION SUMMARY: Generated vs Human Summaries\n")
                handle.write("=" * 60 + "\n\n")
                for result in results:
                    method_info_map = {
                        'LLM (Mixtral)': ' (Mixtral-8x22B)',
                        'LLM (Claude)': ' (Claude Sonnet 4.6)',
                        'LLM (Gemini)': ' (Gemini 3.5 Flash)',
                        'LLM (GPT)': ' (GPT-5.4 Mini)',
                        'LLM (Mistral)': ' (Mistral Small 2603)',
                        'LLM (Qwen)': ' (Qwen3.7)',
                    }
                    method_info = method_info_map.get(result.method, "")
                    metrics = result.metrics
                    handle.write(f"\n{result.method}{method_info}:\n")
                    handle.write(f"  Classes Evaluated: {metrics['classes_evaluated']}\n")
                    handle.write(
                        f"  Avg Cosine Similarity: {metrics['avg_cosine']:.4f} (±{metrics['cosine_std']:.4f})\n"
                    )
                    handle.write(f"  Avg BERT Precision: {metrics['avg_bert_precision']:.4f}\n")
                    handle.write(f"  Avg BERT Recall: {metrics['avg_bert_recall']:.4f}\n")
                    handle.write(
                        f"  Avg BERT F1: {metrics['avg_bert_f1']:.4f} (±{metrics['bert_f1_std']:.4f})\n"
                    )
                    # handle.write(f"  Combined Score: {metrics['combined_score']:.4f}\n")

            formatted_overall = pd.DataFrame(result.metrics for result in results)
            float_cols = [
                'avg_cosine',
                'avg_bert_precision',
                'avg_bert_recall',
                'avg_bert_f1',
                'cosine_std',
                'bert_f1_std',
            ]
            for col in float_cols:
                if col in formatted_overall.columns:
                    formatted_overall[col] = formatted_overall[col].map(lambda x: f"{x:.4f}")

            # Append mode preserves all previous analyses already present.
            with open(results_file, 'a', encoding='utf-8') as handle:
                handle.write("\n" + "=" * 60 + "\n")
                handle.write("OVERALL ANALYSIS\n")
                handle.write("=" * 60 + "\n")
                handle.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                handle.write(formatted_overall.to_string(index=False))
                handle.write("\n\nDetailed Metrics by Method:\n")
                for result in results:
                    method_info_map = {
                        'LLM (Mixtral)': ' (Mixtral-8x22B)',
                        'LLM (Claude)': ' (Claude Sonnet 4.6)',
                        'LLM (Gemini)': ' (Gemini 3.5 Flash)',
                        'LLM (GPT)': ' (GPT-5.4 Mini)',
                        'LLM (Mistral)': ' (Mistral Small 2603)',
                        'LLM (Qwen)': ' (Qwen3.7)',
                    }
                    method_info = method_info_map.get(result.method, "")
                    metrics = result.metrics
                    handle.write(f"\n{result.method}{method_info}\n")
                    handle.write(f"  Projects Evaluated: {metrics['projects_evaluated']}\n")
                    handle.write(f"  Classes Evaluated: {metrics['classes_evaluated']}\n")
                    handle.write(f"  Avg Cosine Similarity: {metrics['avg_cosine']:.4f}\n")
                    handle.write(f"  Avg BERT Precision: {metrics['avg_bert_precision']:.4f}\n")
                    handle.write(f"  Avg BERT Recall: {metrics['avg_bert_recall']:.4f}\n")
                    handle.write(f"  Avg BERT F1: {metrics['avg_bert_f1']:.4f}\n")
                    # handle.write(f"  Combined Score: {metrics['combined_score']:.4f}\n")

                # Dedicated section for the new per-design-pattern requirement.
                handle.write("\n" + "=" * 60 + "\n")
                handle.write("PER-DESIGN-PATTERN ANALYSIS\n")
                handle.write("=" * 60 + "\n")
                handle.write(
                    "Computed from matched class pairs, grouped by canonical design pattern labels.\n"
                )
                for result in results:
                    handle.write(f"\n{result.method}\n")
                    pmetrics = result.pattern_metrics
                    handle.write(f"  Patterns Evaluated: {pmetrics['patterns_evaluated']}\n")
                    handle.write(f"  Macro Avg Cosine: {pmetrics['macro_avg_cosine']:.4f}\n")
                    handle.write(f"  Macro Avg BERT F1: {pmetrics['macro_avg_bert_f1']:.4f}\n")
                    # handle.write(f"  Macro Combined: {pmetrics['macro_combined_score']:.4f}\n")
                    handle.write(f"  Micro Avg Cosine: {pmetrics['micro_avg_cosine']:.4f}\n")
                    handle.write(f"  Micro Avg BERT F1: {pmetrics['micro_avg_bert_f1']:.4f}\n")
                    # handle.write(f"  Micro Combined: {pmetrics['micro_combined_score']:.4f}\n")

                    if result.pattern_stats.empty:
                        handle.write("  Pattern table: no rows\n")
                        continue

                    cols = [
                        'design_pattern',
                        'matched_pairs',
                        'avg_cosine',
                        'avg_bert_f1',
                    ]
                    pattern_table = result.pattern_stats[cols].copy().sort_values(
                        'avg_bert_f1', ascending=False
                    )
                    for col in ('avg_cosine', 'avg_bert_f1'):
                        pattern_table[col] = pattern_table[col].map(lambda x: f"{x:.4f}")
                    handle.write(pattern_table.to_string(index=False))
                    handle.write("\n")
        except OSError as exc:
            print(f"ERROR writing summary files: {exc}")
            return

        print(f"Saved summary: {summary_file}")
        print(f"Saved overall analysis: {results_file}")

    def _print_completion(self, results: List[MethodEvaluationResult]) -> None:
        """Print a final summary listing every output file that was generated."""
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {self.config.output_dir}")
        print("\nGenerated files:")
        print("  - overall_comparison.csv")
        print("  - evaluation_summary.txt")
        print("  - pattern_overall_comparison.csv")
        has_nc = any(r.method in _NC_METHODS for r in results)
        has_non_nc = any(r.method in _NON_NC_LLM_METHODS for r in results)
        has_baselines = any(r.method in {'NLG', 'SWUM'} for r in results)
        if has_baselines and has_non_nc:
            print("  - concise_all_methods_violin.png")
        if has_baselines and has_nc:
            print("  - nc_all_methods_violin.png")
        if has_nc:
            print("  - nc_llm_only_violin.png")
        if has_non_nc:
            print("  - concise_llm_only_violin.png")
        for result in results:
            method = result.method.lower()
            print(f"  - {method}_vs_human_class_scores.csv")
            print(f"  - {method}_vs_human_project_scores.csv")
            print(f"  - {method}_vs_human_pattern_scores.csv")


def parse_arguments(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries against human summaries.")
    parser.add_argument(
        '--human-csv',
        type=Path,
        default=Path('input/DPS_Human_Summaries.csv'),
        help='Path to human summaries CSV file',
    )
    parser.add_argument(
        '--nlg-csv',
        type=Path,
        default=Path('output/summary-output/nlg_summaries.csv'),
        help='Path to NLG summaries CSV file',
    )
    parser.add_argument(
        '--swum-csv',
        type=Path,
        default=Path('output/summary-output/swum_summaries.csv'),
        help='Path to SWUM summaries CSV file',
    )
    parser.add_argument(
        '--dps-llm-csv',
        type=Path,
        default=None,
        help='Optional path to legacy LLM (Mixtral) summaries CSV file',
    )
    parser.add_argument(
        '--llm-claude-csv',
        type=Path,
        default=Path('output/summary-output/LLM_CLAUDE_SUMMARY.csv'),
        help='Path to Claude LLM summaries CSV file',
    )
    parser.add_argument(
        '--llm-gemini-csv',
        type=Path,
        default=None,
        help='Optional path to Gemini LLM summaries CSV file',
    )
    parser.add_argument(
        '--llm-gpt-csv',
        type=Path,
        default=Path('output/summary-output/LLM_GPT_SUMMARY.csv'),
        help='Path to GPT LLM summaries CSV file',
    )
    parser.add_argument(
        '--llm-mistral-csv',
        type=Path,
        default=Path('output/summary-output/LLM_MISTRAL_SUMMARY.csv'),
        help='Path to Mistral LLM summaries CSV file',
    )
    parser.add_argument(
        '--llm-qwen-csv',
        type=Path,
        default=Path('output/summary-output/LLM_QWEN_SUMMARY.csv'),
        help='Path to Qwen LLM summaries CSV file',
    )
    parser.add_argument(
        '--dps-llm-nonconcise-csv',
        type=Path,
        default=None,
        help='Optional path to LLM non-concise summaries CSV file',
    )
    parser.add_argument(
        '--llm-claude-nc-csv',
        type=Path,
        default=Path('output/summary-output/LLM_CLAUDE_NC_SUMMARY.csv'),
        help='Path to Claude NC summaries CSV file',
    )
    parser.add_argument(
        '--llm-gpt-nc-csv',
        type=Path,
        default=Path('output/summary-output/LLM_GPT_NC_SUMMARY.csv'),
        help='Path to GPT NC summaries CSV file',
    )
    parser.add_argument(
        '--llm-mistral-nc-csv',
        type=Path,
        default=Path('output/summary-output/LLM_MISTRAL_NC_SUMMARY.csv'),
        help='Path to Mistral NC summaries CSV file',
    )
    parser.add_argument(
        '--llm-qwen-nc-csv',
        type=Path,
        default=Path('output/summary-output/LLM_QWEN_NC_SUMMARY.csv'),
        help='Path to Qwen NC summaries CSV file',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation-results'),
        help='Output directory for results',
    )
    return parser.parse_args([] if argv is None else argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)
    config = EvaluationConfig(
        human_csv=args.human_csv,
        nlg_csv=args.nlg_csv,
        swum_csv=args.swum_csv,
        output_dir=args.output_dir,
        llm_claude_csv=args.llm_claude_csv,
        llm_gemini_csv=args.llm_gemini_csv,
        llm_gpt_csv=args.llm_gpt_csv,
        llm_mistral_csv=args.llm_mistral_csv,
        llm_qwen_csv=args.llm_qwen_csv,
        dps_llm_csv=args.dps_llm_csv,
        dps_llm_nonconcise_csv=args.dps_llm_nonconcise_csv,
        llm_claude_nc_csv=args.llm_claude_nc_csv,
        llm_gpt_nc_csv=args.llm_gpt_nc_csv,
        llm_mistral_nc_csv=args.llm_mistral_nc_csv,
        llm_qwen_nc_csv=args.llm_qwen_nc_csv,
    )
    pipeline = SummaryEvaluationPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
