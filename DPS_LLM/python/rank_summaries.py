"""
rank_summaries.py - Multi-criteria ranking system for code summaries

This script runs four one-pass comparisons using the multi-criteria ranking engine:

- NLG vs Claude vs SWUM
- NLG vs Qwen vs SWUM
- NLG vs GPT vs SWUM
- NLG vs Mistral vs SWUM

Each comparison evaluates all human-referenced summaries against 5 criteria:
1. Accuracy - How factually correct and faithful the summary is to the source code.
2. Conciseness - How clearly the summary conveys key points without unnecessary detail.
3. Adequacy - How completely the summary covers the important functionality and intent.
4. Code Context - How well the summary reflects surrounding implementation details and relationships.
5. Design Pattern Recognition - How accurately the summary identifies and explains relevant design patterns.

Each criterion is evaluated via LLM (configured via RANK_SUMMARIES_MODEL in .env), ranking
NLG (slot A), each LLM model (slot B), and SWUM (slot C) from most to least relevant.
"""

import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv


def _strip_extension(filename: str) -> str:
    """Remove common source-file extensions when building comparison keys."""
    if not isinstance(filename, str):
        return ""
    return re.sub(r"\.(java|txt|md)$", "", filename.strip(), flags=re.IGNORECASE)


def _normalize_component(value: str) -> str:
    """Normalize project or filename segments for reliable cross-file matching."""
    if not isinstance(value, str):
        return ""
    value = _strip_extension(value)
    value = value.lower().strip()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def build_match_key(project: str, filename: str) -> str:
    """Build a canonical match key that tolerates naming and formatting differences."""
    return f"{_normalize_component(project)}::{_normalize_component(filename)}"


def _first_non_blank(*values: str | None) -> str | None:
    """Return the first non-empty string from a list of candidates."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def resolve_ranking_model(model_override: str | None = None) -> str:
    """Resolve the ranking model from a CLI override or from the .env file."""
    # Prefer an explicit command-line model override, then fall back to the .env value.
    model = _first_non_blank(model_override, os.getenv('RANK_SUMMARIES_MODEL'))
    if not model:
        raise ValueError("RANK_SUMMARIES_MODEL not found in .env file")
    return model


class MultiCriteriaRanker:
    """Ranks summaries using multiple criteria via LLM API."""

    CRITERIA = {
        'accuracy': 'accuracy',
        'conciseness': 'conciseness',
        'adequacy': 'adequacy',
        'code_context': 'context',
        'design_patterns': 'pattern'
    }

    def __init__(self, api_key: str, api_url: str, model: str, prompts: dict[str, str], max_tokens: int) -> None:
        """Initialise ranker with API configuration and prompt templates."""
        # Prompts are provided externally via JSON so updates do not require code edits.
        self.prompts = prompts
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens

    def rank_single_criterion(self, human_summary, summary_a, summary_b, summary_c,
                             criterion_name, criterion_key):
        """
        Rank three summaries on a single criterion using LLM.

        Returns:
            dict: Rankings for each method (1=best, 3=worst) and reasoning
        """
        # Build criterion-specific prompt using dedicated methods
        template = self.prompts.get(criterion_name)
        if not template:
            template = "Rank summaries 1, 2, 3 from best to worst. Output only the ranking."
        prompt = template.format(
            human_summary=human_summary,
            summary_a=summary_a,
            summary_b=summary_b,
            summary_c=summary_c,
        )


        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        # Log the selected ranking model before each request so one-sample runs can confirm the target model.
        print(f"Calling ranking model: {self.model}")

        # Simple retry loop to reduce transient failures
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
                    # Invalid/ambiguous output; don't bias results
                    print("invalid parse; skipping criterion")
                    return None
                return rankings
            except Exception as e:
                last_error = e
        print(f"    ERROR calling API: {str(last_error)}")
        return None

    def _parse_ranking_output(self, content):
        """
        Parse LLM output to extract ranking robustly.
        Returns a dict mapping positions to summary ids (e.g., {"1": "2", "2": "1", "3": "3"})
        where keys are rank positions (1=best) and values are summary numbers (1=A, 2=B, 3=C).
        Returns None if the output is invalid/ambiguous.

        Ranking interpretation (Borda count):
          The LLM is asked to rank three summaries from best (1) to worst (3) across a
          given criterion. The parser extracts an ORDERED PLACEMENT list — the first number
          is the summary in 1st place, the second number is the summary in 2nd place, etc.
          These placements are later converted to Borda points (1st=3, 2nd=2, 3rd=1) and
          summed across all 5 criteria to produce a total score per method. The method with
          the highest total score is declared the winner.
        """
        import re
        text = (content or "").strip()
        if not text:
            return None

        # Case 1: LLM outputs a plain ordered placement list, e.g. "2, 1, 3" meaning
        # 1st place = summary 2, 2nd place = summary 1, 3rd place = summary 3.
        # The prompts ask "rank from best (1) to worst (3)", so LLMs consistently produce
        # this ordered-placement format for short outputs (max_tokens=75).
        #
        # FIX: Previously this branch incorrectly treated the numbers as rank-per-summary
        # (i.e. "summary 1 gets rank 2, summary 2 gets rank 1, ..."), which produced the
        # wrong winner for 2 of the 6 possible permutations ("2,3,1" and "3,1,2").
        # The corrected mapping directly assigns: position "1" → ranks[0], etc.
        numbers = re.findall(r"[123]", text)
        if len(numbers) >= 3:
            ranks = numbers[:3]
            # Validate it is a permutation of 1, 2, 3
            if set(ranks) == {"1", "2", "3"}:
                # OLD (rank-per-summary — incorrect for ordered-placement LLM output):
                # position_mapping = {ranks[0]: "1", ranks[1]: "2", ranks[2]: "3"}
                # FIX: treat as ordered placement — first number is the summary in 1st place
                position_mapping = {"1": ranks[0], "2": ranks[1], "3": ranks[2]}
                position_mapping["reasoning"] = text
                return position_mapping

        # Case 2: Verbose ordered list, e.g. "1st: Summary 2, 2nd: Summary 1, 3rd: Summary 3"
        # Already uses ordered-placement logic — no change needed here.
        ordered = re.findall(r"1(?:st)?\D*([123]).*?2(?:nd)?\D*([123]).*?3(?:rd)?\D*([123])", text, flags=re.IGNORECASE | re.DOTALL)
        if ordered:
            a, b, c = ordered[0]
            if set([a, b, c]) == {"1", "2", "3"}:
                position_mapping = {"1": a, "2": b, "3": c, "reasoning": text}
                return position_mapping

        # Unable to parse confidently
        return None

    def rank_summaries_all_criteria(self, human_summary, summary_a, summary_b, summary_c,
                                    file_name, project_name, labels=None):
        """
        Rank summaries on all 5 criteria.

        Returns:
            dict: Results containing rankings for each criterion and aggregate statistics
        """
        if labels is None:
            labels = {'A': 'A', 'B': 'B', 'C': 'C'}
        print(f"\n  Ranking: {file_name} (Project: {project_name})")

        results = {
            'project': project_name,
            'file': file_name,
            'human_summary': human_summary,
            'summary_a': summary_a,
            'summary_b': summary_b,
            'summary_c': summary_c
        }

        # Borda count scoring: each criterion is evaluated independently by the LLM.
        # Placement is converted to points — 1st=3, 2nd=2, 3rd=1 — then summed across
        # all scored criteria. The method with the highest total wins.
        total_points = {'A': 0, 'B': 0, 'C': 0}
        # FIX: track how many criteria were actually scored so the average is not deflated
        # by API errors or parse failures. Previously the denominator was hardcoded to 5
        # regardless of how many criteria were skipped.
        scored_criteria = 0

        for idx, (criterion_name, criterion_key) in enumerate(self.CRITERIA.items(), 1):
            print(f"    [{idx}/5] Evaluating {criterion_name}...", end=' ')

            ranking = self.rank_single_criterion(
                human_summary, summary_a, summary_b, summary_c,
                criterion_name, criterion_key
            )

            # ranking contains: {"1": "2", "2": "1", "3": "3", "reasoning": "..."}
            # keys are rank positions (1=best), values are summary numbers (1=A, 2=B, 3=C)

            if ranking is None:
                # Record blanks for this criterion and skip point allocation
                results[f'{criterion_name}_rank_1st'] = ''
                results[f'{criterion_name}_rank_2nd'] = ''
                results[f'{criterion_name}_rank_3rd'] = ''
                results[f'{criterion_name}_reasoning'] = 'Invalid or error response; criterion skipped'
                print("skipped")
                continue

            scored_criteria += 1

            first_place = ranking.get('1')  # Which summary (1, 2, or 3) is 1st
            second_place = ranking.get('2')
            third_place = ranking.get('3')

            # Store rankings for this criterion
            results[f'{criterion_name}_rank_1st'] = first_place
            results[f'{criterion_name}_rank_2nd'] = second_place
            results[f'{criterion_name}_rank_3rd'] = third_place
            results[f'{criterion_name}_reasoning'] = ranking.get('reasoning', '')

            # Award points based on ranking
            # 1st place gets 3 points, 2nd gets 2, 3rd gets 1
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

        # Calculate aggregate statistics
        results['total_points_a'] = total_points['A']
        results['total_points_b'] = total_points['B']
        results['total_points_c'] = total_points['C']
        results['scored_criteria'] = scored_criteria

        # FIX: divide by the number of criteria actually scored, not a hardcoded 5.
        # If all 5 criteria are scored this is equivalent; if any were skipped due to
        # API errors the average correctly reflects only the criteria that produced results.
        # Guard against zero division when all criteria failed.
        avg_divisor = scored_criteria if scored_criteria > 0 else 1
        # results['avg_points_a'] = round(total_points['A'] / 5, 2)
        # results['avg_points_b'] = round(total_points['B'] / 5, 2)
        # results['avg_points_c'] = round(total_points['C'] / 5, 2)
        results['avg_points_a'] = round(total_points['A'] / avg_divisor, 2)
        results['avg_points_b'] = round(total_points['B'] / avg_divisor, 2)
        results['avg_points_c'] = round(total_points['C'] / avg_divisor, 2)

        # Determine winner
        max_points = max(total_points.values())
        winners = [k for k, v in total_points.items() if v == max_points]
        # results['winner'] = ', '.join(winners) if len(winners) > 1 else winners[0]
        winner_labels = [labels[w] for w in winners]
        results['winner'] = ', '.join(winner_labels) if len(winner_labels) > 1 else winner_labels[0]

        # print(f"    Total Points: A={total_points['A']}, B={total_points['B']}, C={total_points['C']}")
        print(f"    Total Points: {labels['A']}={total_points['A']}, {labels['B']}={total_points['B']}, {labels['C']}={total_points['C']}")
        print(f"    Winner: {results['winner']}")

        return results


class SummaryRankingPipeline:
    """Runs NLG vs each LLM model vs SWUM comparisons across all human summaries.

    Four comparisons are executed — one per LLM model — with NLG fixed as slot A
    and SWUM fixed as slot C.  Each comparison processes all available human summaries.
    """

    CRITERIA = ['accuracy', 'conciseness', 'adequacy', 'code_context', 'design_patterns']

    # Slot B is the variable LLM model; A=NLG and C=SWUM are always fixed.
    COMPARISONS = [
        ('CLAUDE',  'LLM_CLAUDE_SUMMARY.csv'),
        ('QWEN',    'LLM_QWEN_SUMMARY.csv'),
        ('GPT',     'LLM_GPT_SUMMARY.csv'),
        ('MISTRAL', 'LLM_MISTRAL_SUMMARY.csv'),
    ]
    NLG_CSV  = 'nlg_summaries.csv'
    SWUM_CSV = 'swum_summaries.csv'

    def __init__(self, model_override: str | None = None, limit: int | None = None) -> None:
        """Initialise pipeline, loading config and credentials from the .env file."""
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.limit = limit

        self.output_dir  = (base_dir / 'output' / 'summary-output').resolve()
        self.input_dir   = (base_dir / 'input').resolve()
        self.results_dir = (base_dir / 'evaluation-results').resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        env_path = base_dir.parent / '.env'
        load_dotenv(env_path)

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError('OPENROUTER_API_KEY not found in .env file')

        api_url = os.getenv('RANK_SUMMARIES_API_URL') or os.getenv('OPENROUTER_API_URL')
        if not api_url:
            raise ValueError(
                'Set OPENROUTER_API_URL (or RANK_SUMMARIES_API_URL) in .env before running the ranking pipeline'
            )

        model = resolve_ranking_model(model_override)
        print(f'Using ranking model: {model}')

        # Require explicit .env configuration so ranking behaviour is environment-driven
        # and never silently falls back to an in-code default token budget.
        max_tokens_raw = os.getenv('RANK_SUMMARIES_MAX_TOKENS')
        if not max_tokens_raw:
            raise ValueError('RANK_SUMMARIES_MAX_TOKENS not found in .env file')
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ValueError('RANK_SUMMARIES_MAX_TOKENS must be an integer') from exc

        prompts_path = base_dir.parent / 'resources' / 'prompts.json'
        if not prompts_path.exists():
            raise FileNotFoundError(f'Prompt file not found: {prompts_path}')

        with open(prompts_path, 'r', encoding='utf-8') as fh:
            prompts_data = json.load(fh)
        if not isinstance(prompts_data, dict):
            raise ValueError('prompts.json must contain a top-level JSON object')

        ranking_prompts = prompts_data.get('dps_llm', {}).get('summary_ranking')
        if not isinstance(ranking_prompts, dict):
            raise ValueError(
                'prompts.json is missing the dps_llm.summary_ranking section required by rank_summaries.py'
            )

        self.ranker = MultiCriteriaRanker(
            api_key=api_key,
            api_url=api_url,
            model=model,
            prompts=ranking_prompts,
            max_tokens=max_tokens,
        )

    def load_corpus(self, filename: str) -> pd.DataFrame:
        """Load and normalise a summary CSV from output/summary-output."""
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
        """Load and normalise human summary references from input/DPS_Human_Summaries.csv."""
        human_path = self.input_dir / 'DPS_Human_Summaries.csv'
        if not human_path.exists():
            raise FileNotFoundError(f'Human summaries file not found: {human_path}')

        df = pd.read_csv(human_path)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        required = {'project', 'file_name', 'human_summary'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f'Missing required columns in DPS_Human_Summaries.csv: {sorted(missing)}'
            )

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
    ) -> list[dict]:
        """Rank one comparison set: NLG (slot A) vs model (slot B) vs SWUM (slot C)."""
        print(f"\n{'=' * 70}")
        print(f'COMPARISON: NLG vs {comparison_name} vs SWUM')
        print(f"{'=' * 70}")

        df_llm = self.load_corpus(llm_filename)
        print(f'  Loaded {llm_filename}: {len(df_llm)} entries')

        summary_map_nlg  = df_nlg.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        summary_map_llm  = df_llm.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()
        summary_map_swum = df_swum.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict()

        results: list[dict] = []
        ranked_count  = 0
        skipped_count = 0
        total_rows    = len(df_human)

        for idx, row in enumerate(df_human.itertuples(index=False), start=1):
            project_name   = row.project
            file_name      = row.file_name
            human_summary  = str(row.human_summary).strip()
            match_key      = row.match_key

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
                result: dict = {
                    'comparison':       comparison_name,
                    'llm_source_file':  llm_filename,
                    'project':          project_name,
                    'file':             file_name,
                    'match_key':        match_key,
                    'status':           'skipped',
                    'missing_methods':  ', '.join(missing_methods),
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
                # labels=None,  # was: generic A/B/C labels
                labels={'A': 'NLG', 'B': comparison_name, 'C': 'SWUM'},
            )

            ranking_result['comparison']      = comparison_name
            ranking_result['llm_source_file'] = llm_filename
            ranking_result['status']          = 'ranked'
            ranking_result['missing_methods'] = ''
            ranking_result['match_key']       = match_key

            results.append(ranking_result)
            ranked_count += 1

        print(f'\nComparison {comparison_name} complete: Ranked={ranked_count}, Skipped={skipped_count}')
        return results

    def compute_comparison_stats(self, results: list[dict]) -> dict:
        """Compute summary statistics for a single comparison."""
        ranked = [r for r in results if r.get('status') == 'ranked']
        df = pd.DataFrame(ranked)

        comparison_name = results[0]['comparison'] if results else 'UNKNOWN'
        llm_source_file = results[0]['llm_source_file'] if results else ''

        if df.empty:
            return {
                'comparison':      comparison_name,
                'llm_source_file': llm_source_file,
                'ranked_count':    0,
                'skipped_count':   len(results),
            }

        stats: dict = {
            'comparison':      comparison_name,
            'llm_source_file': llm_source_file,
            'ranked_count':    len(df),
            'skipped_count':   len([r for r in results if r.get('status') != 'ranked']),
        }

        for criterion in self.CRITERIA:
            first_counts  = df[f'{criterion}_rank_1st'].value_counts()
            second_counts = df[f'{criterion}_rank_2nd'].value_counts()
            third_counts  = df[f'{criterion}_rank_3rd'].value_counts()

            for corpus_num, corpus_key in [('1', 'a'), ('2', 'b'), ('3', 'c')]:
                stats[f'{criterion}_{corpus_key}_1st'] = first_counts.get(corpus_num, 0)
                stats[f'{criterion}_{corpus_key}_2nd'] = second_counts.get(corpus_num, 0)
                stats[f'{criterion}_{corpus_key}_3rd'] = third_counts.get(corpus_num, 0)

        stats['avg_points_a'] = df['total_points_a'].mean()
        stats['avg_points_b'] = df['total_points_b'].mean()
        stats['avg_points_c'] = df['total_points_c'].mean()
        return stats

    def compute_aggregate_stats(self, all_stats: list[dict]) -> dict:
        """Compute min/max/avg aggregates across all four comparisons."""
        df = pd.DataFrame(all_stats)
        aggregate: dict = {
            'total_ranked':  int(df['ranked_count'].sum()),
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

    def write_report(self, all_stats: list[dict], aggregate_stats: dict) -> None:
        """Write a human-readable text report summarising all four comparisons."""
        report_path = self.results_dir / 'model_comparisons_ranking_report.txt'

        lines: list[str] = [
            '=' * 70,
            'MODEL COMPARISON RANKING RESULTS (NLG vs MODEL vs SWUM)',
            '=' * 70,
            '',
            'Per-Comparison Summary:',
            '-' * 70,
        ]

        for stats in all_stats:
            comparison = stats['comparison']
            lines.append(f"\nComparison: NLG vs {comparison} vs SWUM")
            lines.append(f"  Source file (B): {stats['llm_source_file']}")
            lines.append(f"  Ranked: {stats['ranked_count']}, Skipped: {stats['skipped_count']}")
            lines.append(
                f"  Average Points: A(NLG)={stats['avg_points_a']:.2f}, "
                f"B({comparison})={stats['avg_points_b']:.2f}, "
                f"C(SWUM)={stats['avg_points_c']:.2f}"
            )
            lines.append('  Ranking by Criterion:')
            for criterion in self.CRITERIA:
                lines.append(f'    {criterion.upper()}:')
                for corpus in ['a', 'b', 'c']:
                    first  = stats.get(f'{criterion}_{corpus}_1st', 0)
                    second = stats.get(f'{criterion}_{corpus}_2nd', 0)
                    third  = stats.get(f'{criterion}_{corpus}_3rd', 0)
                    label  = 'NLG' if corpus == 'a' else (comparison if corpus == 'b' else 'SWUM')
                    lines.append(f'      {label}: {first} first, {second} second, {third} third')

        lines += [
            '\n' + '-' * 70,
            'Aggregate Statistics (Min/Max/Avg across all 4 comparisons):',
            '-' * 70,
            '',
            'Overall Average Points:',
            (
                f"  A (NLG):   min={aggregate_stats['avg_points_a_min']:.2f}, "
                f"max={aggregate_stats['avg_points_a_max']:.2f}, "
                f"avg={aggregate_stats['avg_points_a_avg']:.2f}"
            ),
            (
                f"  B (Model): min={aggregate_stats['avg_points_b_min']:.2f}, "
                f"max={aggregate_stats['avg_points_b_max']:.2f}, "
                f"avg={aggregate_stats['avg_points_b_avg']:.2f}"
            ),
            (
                f"  C (SWUM):  min={aggregate_stats['avg_points_c_min']:.2f}, "
                f"max={aggregate_stats['avg_points_c_max']:.2f}, "
                f"avg={aggregate_stats['avg_points_c_avg']:.2f}"
            ),
            '\n' + '=' * 70,
        ]

        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')

        print(f'\nReport saved to: {report_path}')

    def run(self) -> None:
        """Execute all four NLG vs LLM vs SWUM comparisons and write output files."""
        print('=' * 70)
        print('MODEL COMPARISON RANKING PIPELINE')
        print('=' * 70)

        print('\nLoading fixed corpora (NLG and SWUM)...')
        df_nlg   = self.load_corpus(self.NLG_CSV)
        df_swum  = self.load_corpus(self.SWUM_CSV)
        df_human = self.load_human_summaries()

        print(f'  Loaded {self.NLG_CSV}: {len(df_nlg)} entries')
        print(f'  Loaded {self.SWUM_CSV}: {len(df_swum)} entries')
        print(f'  Loaded Human Summaries: {len(df_human)} entries')
        if self.limit is not None:
            print(f'  Limit enabled: processing first {len(df_human)} human summaries')

        total_api_calls = len(df_human) * len(self.CRITERIA) * len(self.COMPARISONS)
        print(f'\nUp to {total_api_calls} API calls across {len(self.COMPARISONS)} comparisons '
              f'({len(df_human)} summaries × {len(self.CRITERIA)} criteria × {len(self.COMPARISONS)} models)')

        all_results: list[list[dict]] = []
        all_stats:   list[dict]       = []

        for comparison_name, llm_filename in self.COMPARISONS:
            results = self.rank_comparison(
                comparison_name, llm_filename, df_nlg, df_swum, df_human
            )
            all_results.append(results)
            all_stats.append(self.compute_comparison_stats(results))

        aggregate_stats = self.compute_aggregate_stats(all_stats)

        # Flatten all comparison results into one detail CSV
        all_results_flat = [r for comparison_results in all_results for r in comparison_results]

        detail_df       = pd.DataFrame(all_results_flat)
        detail_csv_path = self.results_dir / 'model_comparisons_ranking_detail.csv'
        detail_df.to_csv(detail_csv_path, index=False)
        print(f'\nDetailed results saved to: {detail_csv_path}')

        summary_df       = pd.DataFrame(all_stats)
        summary_csv_path = self.results_dir / 'model_comparisons_ranking_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f'Summary statistics saved to: {summary_csv_path}')

        self.write_report(all_stats, aggregate_stats)

        print('\n' + '=' * 70)
        print('MODEL COMPARISON RANKING COMPLETE')
        print('=' * 70)


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the optional model override and sample limit from the command line."""
    parser = argparse.ArgumentParser(
        description='Rank summaries across NLG/model/SWUM comparisons.'
    )
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


class _Tee(io.TextIOBase):
    """Write to multiple streams simultaneously (console + log file)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    try:
        args     = parse_arguments(argv)
        pipeline = SummaryRankingPipeline(model_override=args.model, limit=args.limit)

        console_file = pipeline.results_dir / 'ranking_console_output.txt'
        with open(console_file, 'w', encoding='utf-8') as fh:
            with redirect_stdout(_Tee(sys.stdout, fh)):
                pipeline.run()

    except Exception as e:
        print(f'\nERROR: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
