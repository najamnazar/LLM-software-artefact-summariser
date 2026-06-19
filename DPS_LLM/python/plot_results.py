#!/usr/bin/env python3
"""
Regenerate the four violin plots from previously computed class-score CSVs.

Run from the DPS_LLM/ directory:
    python python/plot_results.py

No BERTScore computation — just loads existing *_vs_human_class_scores.csv
files and calls the VisualizationManager to produce the four plots.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_summaries import (
    MetricsCalculator,
    MethodEvaluationResult,
    VisualizationManager,
)

RESULTS_DIR = Path('evaluation-results')

# Internal method name → class-scores CSV filename (as saved by the pipeline)
METHOD_CSV_MAP = {
    'NLG':             'nlg_vs_human_class_scores.csv',
    'SWUM':            'swum_vs_human_class_scores.csv',
    'LLM (Claude)':    'llm (claude)_vs_human_class_scores.csv',
    'LLM (GPT)':       'llm (gpt)_vs_human_class_scores.csv',
    'LLM (Mistral)':   'llm (mistral)_vs_human_class_scores.csv',
    'LLM (Qwen)':      'llm (qwen)_vs_human_class_scores.csv',
    'LLM (Claude NC)': 'llm (claude nc)_vs_human_class_scores.csv',
    'LLM (GPT NC)':    'llm (gpt nc)_vs_human_class_scores.csv',
    'LLM (Mistral NC)':'llm (mistral nc)_vs_human_class_scores.csv',
    'LLM (Qwen NC)':   'llm (qwen nc)_vs_human_class_scores.csv',
}


def load_results(results_dir: Path) -> list:
    results = []
    for method, csv_name in METHOD_CSV_MAP.items():
        csv_path = results_dir / csv_name
        if not csv_path.exists():
            print(f"  Skipping {method}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        results.append(MethodEvaluationResult(
            method=method,
            metrics={},
            project_stats=pd.DataFrame(),
            pattern_stats=pd.DataFrame(),
            pattern_metrics={},
            merged=df,
        ))
        print(f"  Loaded {len(df):>3} rows  →  {method}")
    return results


def main() -> None:
    print(f"Loading class-score CSVs from: {RESULTS_DIR.resolve()}")
    results = load_results(RESULTS_DIR)

    if not results:
        print("No result CSVs found. Run evaluate_summaries.py first.")
        sys.exit(1)

    print(f"\nGenerating violin plots for {len(results)} methods...")
    viz = VisualizationManager(MetricsCalculator())
    viz.create_visualizations(results, RESULTS_DIR)
    print("\nDone.")


if __name__ == '__main__':
    main()
