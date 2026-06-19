"""
rank_model_comparisons.py - One-pass ranking across LLM model comparisons

This script runs four single-pass comparisons using the shared multi-criteria
ranking logic from rank_summaries.py:

- NLG vs Claude vs SWUM
- NLG vs Qwen vs SWUM
- NLG vs GPT vs SWUM
- NLG vs Mistral vs SWUM

Ranking criteria (via resources/prompts.json -> summary_ranking):
1. Accuracy - How factually correct and faithful the summary is to the source code.
2. Conciseness - How clearly the summary conveys key points without unnecessary detail.
3. Adequacy - How completely the summary covers the important functionality and intent.
4. Code Context - How well the summary reflects surrounding implementation details and relationships.
5. Design Pattern Recognition - How accurately the summary identifies and explains relevant design patterns.

Ranking model is resolved from:
- CLI positional override (optional)
- .env RANK_SUMMARIES_MODEL (default)
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

# Reuse ranking engine and matching logic from the shared script when available.
# Fall back to a local implementation if the sibling module is not present in this workspace.
sys.path.insert(0, str(Path(__file__).parent))
try:
    _shared_rank_summaries = importlib.import_module('rank_summaries')
    MultiCriteriaRanker = _shared_rank_summaries.MultiCriteriaRanker
    build_match_key = _shared_rank_summaries.build_match_key
    resolve_ranking_model = _shared_rank_summaries.resolve_ranking_model
except ModuleNotFoundError:
    import re

    def _strip_extension(filename: str) -> str:
        if not isinstance(filename, str):
            return ""
        return re.sub(r"\.(java|txt|md)$", "", filename.strip(), flags=re.IGNORECASE)

    def _normalize_component(value: str) -> str:
        if not isinstance(value, str):
            return ""
        value = _strip_extension(value)
        value = value.lower().strip()
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^a-z0-9]", "", value)

    def build_match_key(project: str, filename: str) -> str:
        return f"{_normalize_component(project)}::{_normalize_component(filename)}"

    def _first_non_blank(*values: str | None) -> str | None:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def resolve_ranking_model(model_override: str | None = None) -> str:
        model = _first_non_blank(model_override, os.getenv('RANK_SUMMARIES_MODEL'))
        if not model:
            raise ValueError('RANK_SUMMARIES_MODEL not found in .env file')
        return model

    class MultiCriteriaRanker:
        CRITERIA = {
            'accuracy': 'accuracy',
            'conciseness': 'conciseness',
            'adequacy': 'adequacy',
            'code_context': 'context',
            'design_patterns': 'pattern',
        }

        def __init__(self, api_key: str, api_url: str, model: str, prompts: dict[str, str], max_tokens: int) -> None:
            self.prompts = prompts
            self.api_key = api_key
            self.api_url = api_url
            self.model = model
            self.max_tokens = max_tokens

        def rank_single_criterion(self, human_summary, summary_a, summary_b, summary_c, criterion_name, criterion_key):
            template = self.prompts.get(criterion_name)
            if not template:
                template = 'Rank summaries 1, 2, 3 from best to worst. Output only the ranking.'
            prompt = template.format(
                human_summary=human_summary,
                summary_a=summary_a,
                summary_b=summary_b,
                summary_c=summary_c,
            )

            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }

            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.0,
            }
            if self.max_tokens is not None:
                data['max_tokens'] = self.max_tokens

            print(f'Calling ranking model: {self.model}')

            attempts = 0
            last_error = None
            while attempts < 3:
                attempts += 1
                try:
                    response = requests.post(self.api_url, headers=headers, json=data, timeout=45)
                    response.raise_for_status()
                    result = response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    rankings = self._parse_ranking_output(content)
                    if rankings is None:
                        print('invalid parse; skipping criterion')
                        return None
                    return rankings
                except Exception as exc:
                    last_error = exc
            print(f"    ERROR calling API: {str(last_error)}")
            return None

        def _parse_ranking_output(self, content):
            text = (content or '').strip()
            if not text:
                return None

            numbers = re.findall(r"[123]", text)
            if len(numbers) >= 3:
                ranks = numbers[:3]
                if set(ranks) == {'1', '2', '3'}:
                    position_mapping = {ranks[0]: '1', ranks[1]: '2', ranks[2]: '3'}
                    position_mapping['reasoning'] = text
                    return position_mapping

            ordered = re.findall(r"1(?:st)?\D*([123]).*?2(?:nd)?\D*([123]).*?3(?:rd)?\D*([123])", text, flags=re.IGNORECASE | re.DOTALL)
            if ordered:
                a, b, c = ordered[0]
                if set([a, b, c]) == {'1', '2', '3'}:
                    position_mapping = {'1': a, '2': b, '3': c, 'reasoning': text}
                    return position_mapping

            return None

        def rank_summaries_all_criteria(self, human_summary, summary_a, summary_b, summary_c, file_name, project_name):
            print(f"\n  Ranking: {file_name} (Project: {project_name})")

            results = {
                'project': project_name,
                'file': file_name,
                'human_summary': human_summary,
                'summary_a': summary_a,
                'summary_b': summary_b,
                'summary_c': summary_c,
            }

            total_points = {'A': 0, 'B': 0, 'C': 0}

            for idx, (criterion_name, criterion_key) in enumerate(self.CRITERIA.items(), 1):
                print(f"    [{idx}/5] Evaluating {criterion_name}...", end=' ')

                ranking = self.rank_single_criterion(
                    human_summary, summary_a, summary_b, summary_c,
                    criterion_name, criterion_key,
                )

                if ranking is None:
                    results[f'{criterion_name}_rank_1st'] = ''
                    results[f'{criterion_name}_rank_2nd'] = ''
                    results[f'{criterion_name}_rank_3rd'] = ''
                    results[f'{criterion_name}_reasoning'] = 'Invalid or error response; criterion skipped'
                    print('skipped')
                    continue

                first_place = ranking.get('1')
                second_place = ranking.get('2')
                third_place = ranking.get('3')

                results[f'{criterion_name}_rank_1st'] = first_place
                results[f'{criterion_name}_rank_2nd'] = second_place
                results[f'{criterion_name}_rank_3rd'] = third_place
                results[f'{criterion_name}_reasoning'] = ranking.get('reasoning', '')

                if first_place == '1':
                    total_points['A'] += 3
                elif first_place == '2':
                    total_points['B'] += 3
                elif first_place == '3':
                    total_points['C'] += 3

                if second_place == '1':
                    total_points['A'] += 2
                elif second_place == '2':
                    total_points['B'] += 2
                elif second_place == '3':
                    total_points['C'] += 2

                if third_place == '1':
                    total_points['A'] += 1
                elif third_place == '2':
                    total_points['B'] += 1
                elif third_place == '3':
                    total_points['C'] += 1

                print(f"1st={first_place}, 2nd={second_place}, 3rd={third_place}")

            results['total_points_a'] = total_points['A']
            results['total_points_b'] = total_points['B']
            results['total_points_c'] = total_points['C']

            results['avg_points_a'] = round(total_points['A'] / 5, 2)
            results['avg_points_b'] = round(total_points['B'] / 5, 2)
            results['avg_points_c'] = round(total_points['C'] / 5, 2)

            max_points = max(total_points.values())
            winners = [k for k, v in total_points.items() if v == max_points]
            results['winner'] = ', '.join(winners) if len(winners) > 1 else winners[0]

            print(f"    Total Points: A={total_points['A']}, B={total_points['B']}, C={total_points['C']}")
            print(f"    Winner: {results['winner']}")

            return results


class ModelComparisonRankingPipeline:
    """Orchestrates one-pass NLG/model/SWUM ranking comparisons."""

    CRITERIA = ['accuracy', 'conciseness', 'adequacy', 'code_context', 'design_patterns']

    COMPARISONS = [
        ('CLAUDE', 'LLM_CLAUDE_SUMMARY.csv'),
        ('QWEN', 'LLM_QWEN_SUMMARY.csv'),
        ('GPT', 'LLM_GPT_SUMMARY.csv'),
        ('MISTRAL', 'LLM_MISTRAL_SUMMARY.csv'),
    ]

    def __init__(self, model_override: str | None = None, limit: int | None = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.limit = limit

        self.output_dir = (base_dir / 'output' / 'summary-output').resolve()
        self.input_dir = (base_dir / 'input').resolve()
        self.results_dir = (base_dir / 'evaluation-results').resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        env_path = base_dir / '.env'
        load_dotenv(env_path)

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError('OPENROUTER_API_KEY not found in .env file')

        api_url = os.getenv('RANK_SUMMARIES_API_URL')
        if not api_url:
            api_url = os.getenv('OPENROUTER_API_URL')
        if not api_url:
            raise ValueError('Set OPENROUTER_API_URL (or RANK_SUMMARIES_API_URL) in .env before running model comparison ranking')

        ranking_model = resolve_ranking_model(model_override)
        print(f'Using ranking model: {ranking_model}')

        # Require explicit .env configuration so ranking behavior is environment-driven
        # and never silently falls back to an in-code default token budget.
        max_tokens_raw = os.getenv('RANK_SUMMARIES_MAX_TOKENS')
        if not max_tokens_raw:
            raise ValueError('RANK_SUMMARIES_MAX_TOKENS not found in .env file')
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ValueError('RANK_SUMMARIES_MAX_TOKENS must be an integer') from exc

        prompts_path = self.base_dir / 'resources' / 'prompts.json'
        if not prompts_path.exists():
            raise FileNotFoundError(f'Prompt file not found: {prompts_path}')

        with open(prompts_path, 'r', encoding='utf-8') as prompt_file:
            prompts_data = json.load(prompt_file)
        if not isinstance(prompts_data, dict):
            raise ValueError('prompts.json must contain a top-level JSON object')

        ranking_prompts = prompts_data.get('summary_ranking')
        if not isinstance(ranking_prompts, dict):
            raise ValueError('prompts.json is missing the summary_ranking section required by rank_model_comparisons.py')

        self.ranker = MultiCriteriaRanker(
            api_key=api_key,
            api_url=api_url,
            model=ranking_model,
            prompts=ranking_prompts,
            max_tokens=max_tokens,
        )

    def load_corpus(self, filename: str) -> pd.DataFrame:
        """Load and normalize a summary CSV from output/summary-output."""
        csv_path = self.output_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f'Summary file not found: {csv_path}')

        df = pd.read_csv(csv_path)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        if 'project_name' not in df.columns:
            if 'project' in df.columns:
                df['project_name'] = df['project']
            elif 'projectname' in df.columns:
                df['project_name'] = df['projectname']
            else:
                raise ValueError(f'Missing project column in {filename}')

        if 'file_name' not in df.columns:
            raise ValueError(f'Missing file_name column in {filename}')
        if 'summary' not in df.columns:
            raise ValueError(f'Missing summary column in {filename}')

        df['summary'] = df['summary'].astype(str).str.strip()
        df['match_key'] = df.apply(
            lambda row: build_match_key(row['project_name'], row['file_name']),
            axis=1,
        )
        return df

    def load_human_summaries(self) -> pd.DataFrame:
        """Load and normalize human summary references."""
        human_path = self.input_dir / 'DPS_Human_Summaries.csv'
        if not human_path.exists():
            raise FileNotFoundError(f'Human summaries file not found: {human_path}')

        df = pd.read_csv(human_path)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        required = {'project', 'file_name', 'human_summary'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Missing required columns in DPS_Human_Summaries.csv: {sorted(missing)}')

        df['human_summary'] = df['human_summary'].astype(str).str.strip()
        df['match_key'] = df.apply(
            lambda row: build_match_key(row['project'], row['file_name']),
            axis=1,
        )

        if self.limit is not None:
            return df.head(self.limit).copy()
        return df

    def rank_comparison(
        self,
        comparison_name: str,
        llm_filename: str,
        df_nlg: pd.DataFrame,
        df_swum: pd.DataFrame,
        df_human: pd.DataFrame,
    ) -> List[Dict]:
        """Rank one comparison set: NLG (A) vs model (B) vs SWUM (C)."""
        print(f"\n{'=' * 70}")
        print(f'COMPARISON: NLG vs {comparison_name} vs SWUM')
        print(f"{'=' * 70}")

        df_llm = self.load_corpus(llm_filename)
        print(f'  Loaded {llm_filename}: {len(df_llm)} entries')

        summary_map_nlg = df_nlg.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        summary_map_llm = df_llm.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        summary_map_swum = df_swum.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()

        results: List[Dict] = []
        ranked_count = 0
        skipped_count = 0

        total_rows = len(df_human)
        for idx, row in enumerate(df_human.itertuples(index=False), start=1):
            project_name = row.project
            file_name = row.file_name
            human_summary = str(row.human_summary).strip()
            match_key = row.match_key

            summary_a = summary_map_nlg.get(match_key)
            summary_b = summary_map_llm.get(match_key)
            summary_c = summary_map_swum.get(match_key)

            missing_methods = []
            if not summary_a:
                missing_methods.append('NLG')
            if not summary_b:
                missing_methods.append(comparison_name)
            if not summary_c:
                missing_methods.append('SWUM')

            if missing_methods:
                skipped_count += 1
                print(f"  [{idx}/{total_rows}] Skipping {file_name} (missing: {', '.join(missing_methods)})")
                result = {
                    'comparison': comparison_name,
                    'llm_source_file': llm_filename,
                    'project': project_name,
                    'file': file_name,
                    'match_key': match_key,
                    'status': 'skipped',
                    'missing_methods': ', '.join(missing_methods),
                }
                for criterion in self.CRITERIA:
                    result[f'{criterion}_rank_1st'] = None
                    result[f'{criterion}_rank_2nd'] = None
                    result[f'{criterion}_rank_3rd'] = None
                result['total_points_a'] = None
                result['total_points_b'] = None
                result['total_points_c'] = None
                results.append(result)
                continue

            print(f'  [{idx}/{total_rows}] Ranking {file_name} (Project: {project_name})')

            ranking_result = self.ranker.rank_summaries_all_criteria(
                human_summary,
                summary_a,
                summary_b,
                summary_c,
                file_name,
                project_name,
            )

            ranking_result['comparison'] = comparison_name
            ranking_result['llm_source_file'] = llm_filename
            ranking_result['status'] = 'ranked'
            ranking_result['missing_methods'] = ''
            ranking_result['match_key'] = match_key

            results.append(ranking_result)
            ranked_count += 1

        print(f'\nComparison {comparison_name} complete: Ranked={ranked_count}, Skipped={skipped_count}')
        return results

    def compute_comparison_stats(self, results: List[Dict]) -> Dict:
        """Compute summary statistics for one comparison."""
        ranked = [r for r in results if r['status'] == 'ranked']
        df = pd.DataFrame(ranked)

        comparison_name = results[0]['comparison'] if results else 'UNKNOWN'
        llm_source_file = results[0]['llm_source_file'] if results else ''

        if df.empty:
            return {
                'comparison': comparison_name,
                'llm_source_file': llm_source_file,
                'ranked_count': 0,
                'skipped_count': len(results),
            }

        stats: Dict[str, object] = {
            'comparison': comparison_name,
            'llm_source_file': llm_source_file,
            'ranked_count': len(df),
            'skipped_count': len([r for r in results if r['status'] != 'ranked']),
        }

        for criterion in self.CRITERIA:
            first_col = f'{criterion}_rank_1st'
            second_col = f'{criterion}_rank_2nd'
            third_col = f'{criterion}_rank_3rd'

            first_counts = df[first_col].value_counts()
            second_counts = df[second_col].value_counts()
            third_counts = df[third_col].value_counts()

            for corpus_num, corpus_name in [('1', 'a'), ('2', 'b'), ('3', 'c')]:
                stats[f'{criterion}_{corpus_name}_1st'] = first_counts.get(corpus_num, 0)
                stats[f'{criterion}_{corpus_name}_2nd'] = second_counts.get(corpus_num, 0)
                stats[f'{criterion}_{corpus_name}_3rd'] = third_counts.get(corpus_num, 0)

        stats['avg_points_a'] = df['total_points_a'].mean()
        stats['avg_points_b'] = df['total_points_b'].mean()
        stats['avg_points_c'] = df['total_points_c'].mean()
        return stats

    def compute_aggregate_stats(self, all_stats: List[Dict]) -> Dict:
        """Compute min/max/avg aggregates across all four comparisons."""
        df = pd.DataFrame(all_stats)
        aggregate: Dict[str, object] = {
            'total_ranked': int(df['ranked_count'].sum()),
            'total_skipped': int(df['skipped_count'].sum()),
        }

        for criterion in self.CRITERIA:
            for corpus in ['a', 'b', 'c']:
                for position in ['1st', '2nd', '3rd']:
                    col = f'{criterion}_{corpus}_{position}'
                    if col in df.columns:
                        aggregate[f'{col}_min'] = float(df[col].min())
                        aggregate[f'{col}_max'] = float(df[col].max())
                        aggregate[f'{col}_avg'] = float(df[col].mean())

        for corpus in ['a', 'b', 'c']:
            col = f'avg_points_{corpus}'
            aggregate[f'{col}_min'] = float(df[col].min())
            aggregate[f'{col}_max'] = float(df[col].max())
            aggregate[f'{col}_avg'] = float(df[col].mean())

        return aggregate

    def write_report(self, all_stats: List[Dict], aggregate_stats: Dict):
        """Write a human-readable report dedicated to model comparisons."""
        report_path = self.results_dir / 'model_comparisons_ranking_report.txt'

        lines: List[str] = []
        lines.append('=' * 70)
        lines.append('MODEL COMPARISON RANKING RESULTS (NLG vs MODEL vs SWUM)')
        lines.append('=' * 70)
        lines.append('')

        lines.append('Per-Comparison Summary:')
        lines.append('-' * 70)
        for stats in all_stats:
            comparison = stats['comparison']
            lines.append(f"\nComparison: NLG vs {comparison} vs SWUM")
            lines.append(f"  Source file (B): {stats['llm_source_file']}")
            lines.append(f"  Ranked: {stats['ranked_count']}, Skipped: {stats['skipped_count']}")
            lines.append(
                f"  Average Points: A(NLG)={stats['avg_points_a']:.2f}, "
                f"B({comparison})={stats['avg_points_b']:.2f}, C(SWUM)={stats['avg_points_c']:.2f}"
            )

            lines.append('  Ranking by Criterion:')
            for criterion in self.CRITERIA:
                lines.append(f'    {criterion.upper()}:')
                for corpus in ['a', 'b', 'c']:
                    first = stats.get(f'{criterion}_{corpus}_1st', 0)
                    second = stats.get(f'{criterion}_{corpus}_2nd', 0)
                    third = stats.get(f'{criterion}_{corpus}_3rd', 0)
                    label = 'NLG' if corpus == 'a' else comparison if corpus == 'b' else 'SWUM'
                    lines.append(f'      {label}: {first} first, {second} second, {third} third')

        lines.append('\n' + '-' * 70)
        lines.append('Aggregate Statistics (Min/Max/Avg across all 4 comparisons):')
        lines.append('-' * 70)
        lines.append('')

        lines.append('Overall Average Points:')
        lines.append(
            f"  A (NLG): min={aggregate_stats['avg_points_a_min']:.2f}, "
            f"max={aggregate_stats['avg_points_a_max']:.2f}, avg={aggregate_stats['avg_points_a_avg']:.2f}"
        )
        lines.append(
            f"  B (Model): min={aggregate_stats['avg_points_b_min']:.2f}, "
            f"max={aggregate_stats['avg_points_b_max']:.2f}, avg={aggregate_stats['avg_points_b_avg']:.2f}"
        )
        lines.append(
            f"  C (SWUM): min={aggregate_stats['avg_points_c_min']:.2f}, "
            f"max={aggregate_stats['avg_points_c_max']:.2f}, avg={aggregate_stats['avg_points_c_avg']:.2f}"
        )

        lines.append('\n' + '=' * 70)

        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write('\n'.join(lines) + '\n')

        print(f'\nReport saved to: {report_path}')

    def run(self) -> None:
        """Run all four one-pass model comparisons."""
        print('=' * 70)
        print('MODEL COMPARISON RANKING PIPELINE')
        print('=' * 70)

        print('\nLoading fixed corpora...')
        df_nlg = self.load_corpus('nlg_summaries.csv')
        df_swum = self.load_corpus('swum_summaries.csv')
        df_human = self.load_human_summaries()

        print(f'  Loaded nlg_summaries.csv: {len(df_nlg)} entries')
        print(f'  Loaded swum_summaries.csv: {len(df_swum)} entries')
        print(f'  Loaded Human Summaries: {len(df_human)} entries')
        if self.limit is not None:
            print(f'  Limit enabled: processing first {len(df_human)} human summaries')

        all_results: List[List[Dict]] = []
        all_stats: List[Dict] = []

        for comparison_name, llm_filename in self.COMPARISONS:
            results = self.rank_comparison(comparison_name, llm_filename, df_nlg, df_swum, df_human)
            all_results.append(results)
            stats = self.compute_comparison_stats(results)
            all_stats.append(stats)

        aggregate_stats = self.compute_aggregate_stats(all_stats)

        all_results_flat: List[Dict] = []
        for comparison_results in all_results:
            all_results_flat.extend(comparison_results)

        detail_df = pd.DataFrame(all_results_flat)
        detail_csv_path = self.results_dir / 'model_comparisons_ranking_detail.csv'
        detail_df.to_csv(detail_csv_path, index=False)
        print(f'\nDetailed results saved to: {detail_csv_path}')

        summary_df = pd.DataFrame(all_stats)
        summary_csv_path = self.results_dir / 'model_comparisons_ranking_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f'Summary statistics saved to: {summary_csv_path}')

        self.write_report(all_stats, aggregate_stats)

        print('\n' + '=' * 70)
        print('MODEL COMPARISON RANKING COMPLETE')
        print('=' * 70)


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse optional model override and sample limit from command line."""
    parser = argparse.ArgumentParser(description='Run NLG/model/SWUM ranking comparisons.')
    parser.add_argument(
        'model',
        nargs='?',
        help='Optional ranking model identifier to use instead of RANK_SUMMARIES_MODEL',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional number of human summaries to rank for a quick validation run',
    )
    return parser.parse_args([] if argv is None else argv)


def main(argv: list[str] | None = None):
    """Program entry point for CLI/script execution."""
    try:
        args = parse_arguments(argv)
        pipeline = ModelComparisonRankingPipeline(model_override=args.model, limit=args.limit)
        pipeline.run()
    except Exception as exc:
        print(f'\nERROR: {str(exc)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])