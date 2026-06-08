"""
rank_summaries.py - Multi-criteria ranking system for code summaries

This script ranks summaries from three different methods (A, B, C) based on 5 criteria:
1. Accuracy - How factually correct and faithful the summary is to the source code.
2. Conciseness - How clearly the summary conveys key points without unnecessary detail.
3. Adequacy - How completely the summary covers the important functionality and intent.
4. Code Context - How well the summary reflects surrounding implementation details and relationships.
5. Design Pattern Recognition - How accurately the summary identifies and explains relevant design patterns.

Each criterion is evaluated via LLM (Llama) rankings from most relevant (1) to least relevant (3).
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
        Returns a dict mapping positions to summary ids (e.g., {"1": "1", "2": "2", "3": "3"})
        or None if the output is invalid/ambiguous.
        """
        import re
        text = (content or "").strip()
        if not text:
            return None

        # Case 1: Model returns ranks-per-summary (rank for A,B,C)
        numbers = re.findall(r"[123]", text)
        if len(numbers) >= 3:
            ranks = numbers[:3]
            # Validate permutation of 1,2,3
            if set(ranks) == {"1", "2", "3"}:
                # Invert to positions: which summary is 1st/2nd/3rd
                position_mapping = {ranks[0]: "1", ranks[1]: "2", ranks[2]: "3"}
                position_mapping["reasoning"] = text
                return position_mapping

        # Case 2: Ordered list like "1st: Summary 2, 2nd: Summary 1, 3rd: Summary 3"
        ordered = re.findall(r"1(?:st)?\D*([123]).*?2(?:nd)?\D*([123]).*?3(?:rd)?\D*([123])", text, flags=re.IGNORECASE | re.DOTALL)
        if ordered:
            a, b, c = ordered[0]
            if set([a, b, c]) == {"1", "2", "3"}:
                position_mapping = {"1": a, "2": b, "3": c, "reasoning": text}
                return position_mapping

        # Unable to parse confidently
        return None
    
    def rank_summaries_all_criteria(self, human_summary, summary_a, summary_b, summary_c, 
                                    file_name, project_name):
        """
        Rank summaries on all 5 criteria.
        
        Returns:
            dict: Results containing rankings for each criterion and aggregate statistics
        """
        print(f"\n  Ranking: {file_name} (Project: {project_name})")
        
        results = {
            'project': project_name,
            'file': file_name,
            'human_summary': human_summary,
            'summary_a': summary_a,
            'summary_b': summary_b,
            'summary_c': summary_c
        }
        
        # Track points: 1st place = 3 points, 2nd = 2 points, 3rd = 1 point
        total_points = {'A': 0, 'B': 0, 'C': 0}
        
        for idx, (criterion_name, criterion_key) in enumerate(self.CRITERIA.items(), 1):
            print(f"    [{idx}/5] Evaluating {criterion_name}...", end=' ')
            
            ranking = self.rank_single_criterion(
                human_summary, summary_a, summary_b, summary_c,
                criterion_name, criterion_key
            )
            
            # ranking contains: {"1": "1", "2": "2", "3": "3", "reasoning": "..."}
            # where the values indicate which summary (1=A, 2=B, 3=C) got that position
            
            if ranking is None:
                # Record blanks for this criterion and skip point allocation
                results[f'{criterion_name}_rank_1st'] = ''
                results[f'{criterion_name}_rank_2nd'] = ''
                results[f'{criterion_name}_rank_3rd'] = ''
                results[f'{criterion_name}_reasoning'] = 'Invalid or error response; criterion skipped'
                print("skipped")
                continue

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
        
        results['avg_points_a'] = round(total_points['A'] / 5, 2)
        results['avg_points_b'] = round(total_points['B'] / 5, 2)
        results['avg_points_c'] = round(total_points['C'] / 5, 2)
        
        # Determine winner
        max_points = max(total_points.values())
        winners = [k for k, v in total_points.items() if v == max_points]
        results['winner'] = ', '.join(winners) if len(winners) > 1 else winners[0]
        
        print(f"    Total Points: A={total_points['A']}, B={total_points['B']}, C={total_points['C']}")
        print(f"    Winner: {results['winner']}")
        
        return results


class SummaryRankingPipeline:
    """Main pipeline for ranking summaries from A.csv, B.csv, C.csv."""
    
    def __init__(self, output_dir='output/summary-output', 
                 input_dir='input', 
                 results_dir='evaluation-results',
                 model_override: str | None = None):
        """Initialize pipeline with directory paths."""
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir

        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (base_dir / self.output_dir).resolve()

        self.input_dir = Path(input_dir)
        if not self.input_dir.is_absolute():
            self.input_dir = (base_dir / self.input_dir).resolve()

        self.results_dir = Path(results_dir)
        if not self.results_dir.is_absolute():
            self.results_dir = (base_dir / self.results_dir).resolve()

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.aggregate_stats_text = ""
        
        # Load API key
        env_path = base_dir / '.env'
        load_dotenv(env_path)

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")

        api_url = os.getenv('RANK_SUMMARIES_API_URL')
        if not api_url:
            # Ranking shares the same OpenRouter endpoint; fall back to OPENROUTER_API_URL when no override is set.
            api_url = os.getenv('OPENROUTER_API_URL')
        if not api_url:
            raise ValueError("Set OPENROUTER_API_URL (or RANK_SUMMARIES_API_URL) in .env before running the ranking pipeline")

        # Resolve the ranking model from the command line first so one-off runs can target a specific model.
        model = resolve_ranking_model(model_override)
        print(f"Using ranking model: {model}")

        max_tokens_raw = os.getenv('RANK_SUMMARIES_MAX_TOKENS', '50')
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ValueError("RANK_SUMMARIES_MAX_TOKENS must be an integer") from exc

        prompts_path = base_dir / 'resources' / 'prompts.json'
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

        with open(prompts_path, 'r', encoding='utf-8') as prompt_file:
            prompts_data = json.load(prompt_file)
        if not isinstance(prompts_data, dict):
            raise ValueError("prompts.json must contain a top-level JSON object")

        # Summary ranking templates live under the "summary_ranking" key so LLM wording stays editable outside code.
        ranking_prompts = prompts_data.get('summary_ranking')
        if not isinstance(ranking_prompts, dict):
            raise ValueError("prompts.json is missing the summary_ranking section required by rank_summaries.py")

        self.ranker = MultiCriteriaRanker(
            api_key=api_key,
            api_url=api_url,
            model=model,
            prompts=ranking_prompts,
            max_tokens=max_tokens,
        )
        
    def load_summaries(self):
        """Load and normalise summaries from all data sources."""
        print("\nLoading summary files...")

        def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result.columns = [col.strip().lower().replace(' ', '_') for col in result.columns]
            return result

        # Load corpus files
        self.df_a = _prepare_dataframe(pd.read_csv(self.output_dir / 'A.csv'))
        self.df_b = _prepare_dataframe(pd.read_csv(self.output_dir / 'B.csv'))
        self.df_c = _prepare_dataframe(pd.read_csv(self.output_dir / 'C.csv'))

        print(f"  Loaded A.csv: {len(self.df_a)} entries")
        print(f"  Loaded B.csv: {len(self.df_b)} entries")
        print(f"  Loaded C.csv: {len(self.df_c)} entries")

        # Load human summaries
        self.df_human = _prepare_dataframe(pd.read_csv(self.input_dir / 'DPS_Human_Summaries.csv'))
        print(f"  Loaded Human Summaries: {len(self.df_human)} entries")

        # Ensure required columns are present
        required_method_cols = {'project_name', 'file_name', 'summary'}
        for label, df in [('A.csv', self.df_a), ('B.csv', self.df_b), ('C.csv', self.df_c)]:
            if 'project' in df.columns and 'project_name' not in df.columns:
                df['project_name'] = df['project']
            missing = required_method_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns {missing} in {label}")

        required_human_cols = {'project', 'file_name', 'human_summary'}
        missing_human = required_human_cols - set(self.df_human.columns)
        if missing_human:
            raise ValueError(f"Missing columns {missing_human} in DPS_Human_Summaries.csv")

        # Normalise summary text
        for df in [self.df_a, self.df_b, self.df_c]:
            df['summary'] = df['summary'].astype(str).str.strip()

        self.df_human['human_summary'] = self.df_human['human_summary'].astype(str).str.strip()

        # Build match keys for consistent alignment
        self.df_a['match_key'] = self.df_a.apply(lambda row: build_match_key(row['project_name'], row['file_name']), axis=1)
        self.df_b['match_key'] = self.df_b.apply(lambda row: build_match_key(row['project_name'], row['file_name']), axis=1)
        self.df_c['match_key'] = self.df_c.apply(lambda row: build_match_key(row['project_name'], row['file_name']), axis=1)
        self.df_human['match_key'] = self.df_human.apply(lambda row: build_match_key(row['project'], row['file_name']), axis=1)

        # Record key sets for reporting
        self.keys_a = set(self.df_a['match_key'])
        self.keys_b = set(self.df_b['match_key'])
        self.keys_c = set(self.df_c['match_key'])
        self.keys_human = set(self.df_human['match_key'])

        # Build quick-lookup maps (deduplicating on first occurrence)
        self.summary_maps = {
            'A': self.df_a.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict(),
            'B': self.df_b.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict(),
            'C': self.df_c.drop_duplicates('match_key').set_index('match_key')['summary'].to_dict(),
        }
        
    def get_all_common_samples(self):
        """
        Get all samples that exist in all three corpora (A, B, C) and human summaries.
        
        Returns:
            list: List of common keys
        """
        # Find intersection - keys present in all four datasets
        common_keys = self.keys_a & self.keys_b & self.keys_c & self.keys_human

        print(f"\nFound {len(common_keys)} summaries common to all datasets")
        print(f"  Corpus A unique keys: {len(self.keys_a)}")
        print(f"  Corpus B unique keys: {len(self.keys_b)}")
        print(f"  Corpus C unique keys: {len(self.keys_c)}")
        print(f"  Human unique keys: {len(self.keys_human)}")

        missing_from_methods = self.keys_human - (self.keys_a & self.keys_b & self.keys_c)
        missing_from_human = (self.keys_a & self.keys_b & self.keys_c) - self.keys_human

        print(f"  Human summaries without full method coverage: {len(missing_from_methods)}")
        print(f"  Method summaries missing human references: {len(missing_from_human)}")

        return list(common_keys)
    
    def rank_samples(self, limit=None):
        """Rank every human summary against all available generated summaries."""
        results = []
        total_items = len(self.df_human) if limit is None else min(limit, len(self.df_human))
        ranked_count = 0
        skipped_count = 0

        criteria = list(self.ranker.CRITERIA.keys())

        def _clean_summary(value: str) -> str | None:
            if value is None:
                return None
            if isinstance(value, float) and pd.isna(value):
                return None
            text = str(value).strip()
            return text if text else None

        # Limit iteration if specified
        df_subset = self.df_human.head(total_items) if limit is not None else self.df_human

        for idx, row in enumerate(df_subset.itertuples(index=False), start=1):
            project_name = row.project
            file_name = row.file_name
            human_summary = _clean_summary(row.human_summary)
            match_key = row.match_key

            summary_a = _clean_summary(self.summary_maps['A'].get(match_key))
            summary_b = _clean_summary(self.summary_maps['B'].get(match_key))
            summary_c = _clean_summary(self.summary_maps['C'].get(match_key))

            print(f"  [{idx}/{total_items}]", end=' ')

            missing_methods = [label for label, summary in [('A', summary_a), ('B', summary_b), ('C', summary_c)] if summary is None]

            if missing_methods:
                skipped_count += 1
                print(f"Skipping (missing summaries from {', '.join(missing_methods)})")

                result_record = {
                    'project': project_name,
                    'file': file_name,
                    'human_summary': human_summary or '',
                    'summary_a': summary_a or '',
                    'summary_b': summary_b or '',
                    'summary_c': summary_c or '',
                    'match_key': match_key,
                    'status': 'missing',
                    'missing_methods': ', '.join(missing_methods),
                }

                for criterion in criteria:
                    result_record[f'{criterion}_rank_1st'] = ''
                    result_record[f'{criterion}_rank_2nd'] = ''
                    result_record[f'{criterion}_rank_3rd'] = ''
                    result_record[f'{criterion}_reasoning'] = ''

                result_record['total_points_a'] = None
                result_record['total_points_b'] = None
                result_record['total_points_c'] = None
                result_record['avg_points_a'] = None
                result_record['avg_points_b'] = None
                result_record['avg_points_c'] = None
                result_record['winner'] = ''

                results.append(result_record)
                continue

            print(f"Ranking {file_name} (Project: {project_name})")

            ranking_result = self.ranker.rank_summaries_all_criteria(
                human_summary,
                summary_a,
                summary_b,
                summary_c,
                file_name,
                project_name,
            )

            ranking_result['status'] = 'ranked'
            ranking_result['missing_methods'] = ''
            ranking_result['match_key'] = match_key

            results.append(ranking_result)
            ranked_count += 1

        print(f"\nRanking complete. Ranked: {ranked_count}, Skipped: {skipped_count}")
        return results
    
    def run(self):
        """Execute the complete ranking pipeline."""
        print("="*70)
        print("MULTI-CRITERIA SUMMARY RANKING")
        print("="*70)
        
        # Load all data
        self.load_summaries()
        
        # Get all common samples
        print("\n" + "="*70)
        print("IDENTIFYING ALL COMMON SAMPLES")
        print("="*70)
        
        sample_keys = self.get_all_common_samples()

        total_human_rows = len(self.df_human)
        estimated_calls = total_human_rows * len(self.ranker.CRITERIA)
        estimated_seconds = estimated_calls * 5  # approx 5 seconds per API call

        print(f"\n" + "="*70)
        print(f"ATTEMPTING TO RANK {total_human_rows} HUMAN SUMMARIES")
        print("="*70)
        print(f"This will trigger up to {estimated_calls} API calls ({len(self.ranker.CRITERIA)} criteria per summary)")
        print(f"Estimated time: ~{estimated_seconds // 60} minutes")
        print("\nStarting ranking process...")

        # Rank all samples (including duplicates / differing order)
        results = self.rank_samples()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_file = self.results_dir / 'multi_criteria_rankings.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"\n" + "="*70)
        print("RANKING COMPLETE")
        print("="*70)
        print(f"Results saved to: {output_file}")
        print(f"Total records processed: {len(results)}")
        
        # Calculate aggregate statistics
        aggregate_text = self._print_aggregate_statistics(results_df)
        self.aggregate_stats_text = aggregate_text
        
        return results_df
    
    def _print_aggregate_statistics(self, df):
        """Print aggregate statistics across all ranked samples."""
        lines: list[str] = []

        def emit(line: str = "") -> None:
            print(line)
            lines.append(line)

        emit("\n" + "=" * 70)
        emit("AGGREGATE STATISTICS")
        emit("=" * 70)

        ranked_df = df[df['status'] == 'ranked'].copy()

        if ranked_df.empty:
            emit("No ranked samples available to summarise.")
            return "\n".join(lines)

        # Ensure numeric columns are treated as such
        for col in ['avg_points_a', 'avg_points_b', 'avg_points_c']:
            ranked_df[col] = pd.to_numeric(ranked_df[col], errors='coerce')

        criteria = ['accuracy', 'conciseness', 'adequacy', 'code_context', 'design_patterns']

        emit("\nRanking Summary by Criterion:")
        emit("-" * 70)
        emit("(Shows how often each corpus ranked 1st, 2nd, or 3rd for each criterion)")

        for criterion in criteria:
            emit(f"\n{criterion.upper()}:")

            first_col = f'{criterion}_rank_1st'
            second_col = f'{criterion}_rank_2nd'
            third_col = f'{criterion}_rank_3rd'

            if first_col in ranked_df.columns:
                first_counts = ranked_df[first_col].value_counts()
                second_counts = ranked_df[second_col].value_counts()
                third_counts = ranked_df[third_col].value_counts()

                for corpus_num, corpus_name in [('1', 'A'), ('2', 'B'), ('3', 'C')]:
                    first_cnt = first_counts.get(corpus_num, 0)
                    second_cnt = second_counts.get(corpus_num, 0)
                    third_cnt = third_counts.get(corpus_num, 0)
                    emit(f"  Corpus {corpus_name}: {first_cnt} first, {second_cnt} second, {third_cnt} third")

        emit("\n" + "-" * 70)
        emit("Overall Average Points (3=1st, 2=2nd, 1=3rd per criterion):")
        emit(f"  Corpus A: {ranked_df['avg_points_a'].mean():.2f}")
        emit(f"  Corpus B: {ranked_df['avg_points_b'].mean():.2f}")
        emit(f"  Corpus C: {ranked_df['avg_points_c'].mean():.2f}")

        emit("\n" + "-" * 70)
        emit("Winner Distribution:")
        winner_counts = ranked_df['winner'].value_counts()
        for winner, count in winner_counts.items():
            percentage = (count / len(ranked_df)) * 100
            emit(f"  {winner}: {count}/{len(ranked_df)} ({percentage:.1f}%)")

        missing_df = df[df['status'] != 'ranked']
        if not missing_df.empty:
            emit("\n" + "-" * 70)
            emit("Entries skipped due to missing summaries: {}".format(len(missing_df)))
            breakdown = missing_df['missing_methods'].value_counts(dropna=False)
            for label, count in breakdown.items():
                label_display = label if label else 'Unknown'
                emit(f"  {label_display}: {count}")

        return "\n".join(lines)


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the optional model override and sample limit from the command line."""
    parser = argparse.ArgumentParser(description="Rank summaries against human references.")
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
    """Main entry point."""
    try:
        args = parse_arguments(argv)
        pipeline = SummaryRankingPipeline(model_override=args.model)
        console_buffer = io.StringIO()
        with redirect_stdout(console_buffer):
            pipeline.run(limit=args.limit)

        console_output = console_buffer.getvalue()
        print(console_output, end="")

        console_file = pipeline.results_dir / 'ranking_console_output.txt'
        with open(console_file, 'w', encoding='utf-8') as fh:
            fh.write(console_output)

        aggregate_text = getattr(pipeline, 'aggregate_stats_text', '')
        if aggregate_text:
            results_txt = pipeline.base_dir / 'evaluation-results' / 'results.txt'
            with open(results_txt, 'a', encoding='utf-8') as fh:
                fh.write("\n\n")
                fh.write(aggregate_text)
                if not aggregate_text.endswith("\n"):
                    fh.write("\n")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
