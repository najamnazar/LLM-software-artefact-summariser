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
import hashlib
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from itertools import permutations

import pandas as pd
import requests
from pathlib import Path
from dotenv import dotenv_values, load_dotenv


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


def build_match_key(project: str, folder: str, filename: str) -> str:
    """Build a canonical match key that tolerates naming and formatting differences.

    The design-pattern folder is part of the key because the corpus reuses the same
    class name across pattern folders (e.g. JamesZBL has nine Application.java files,
    one per pattern). Keying on project+filename alone collapsed those into a single
    entry, so every one of them was ranked against the summaries of whichever file
    happened to be read first — 16 of 150 rows were paired with the wrong class.
    """
    return (
        f"{_normalize_component(project)}"
        f"::{_normalize_component(folder)}"
        f"::{_normalize_component(filename)}"
    )


# Column holding the design-pattern folder, in priority order. Generated summary CSVs
# expose it as "Folder Name"; the human CSV uses "Design Pattern".
_FOLDER_COLUMN_CANDIDATES = ('folder_name', 'folder', 'design_pattern', 'pattern')


def resolve_folder_column(df: pd.DataFrame, source_label: str) -> str:
    """Return the column holding the design-pattern folder, or raise if none exists.

    Missing the folder would silently reintroduce cross-pattern collisions, so this
    fails loudly rather than falling back to a project+filename key.
    """
    for candidate in _FOLDER_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"{source_label}: no design-pattern folder column found. Expected one of "
        f"{list(_FOLDER_COLUMN_CANDIDATES)}; available columns: {sorted(df.columns)}"
    )


def assert_unique_match_keys(df: pd.DataFrame, source_label: str) -> None:
    """Fail if match keys collide, so duplicate rows can never be silently dropped.

    Downstream lookups use drop_duplicates(), which keeps the first row and discards
    the rest. That is only safe when keys are already unique.
    """
    duplicated = df['match_key'].duplicated(keep=False)
    if not duplicated.any():
        return
    counts = df.loc[duplicated, 'match_key'].value_counts()
    detail = ', '.join(f'{key} (x{count})' for key, count in counts.head(10).items())
    raise ValueError(
        f'{source_label}: {int(duplicated.sum())} rows share {len(counts)} duplicate '
        f'match keys, so summaries cannot be paired unambiguously. Offending keys: {detail}'
    )


# ---------------------------------------------------------------------------
# Presentation-order randomisation (LLM-judge position-bias control)
# ---------------------------------------------------------------------------

# The ranking prompts render the candidates as a numbered list, and slot 1 was
# always NLG, slot 2 always the LLM under test and slot 3 always SWUM. LLM judges
# favour particular list positions, so that fixed layout handed whichever slot the
# judge prefers a constant advantage that is indistinguishable from a real quality
# difference. Presentation order is now permuted per (item, criterion) and the
# judge's answer is translated back to the canonical systems before it is stored.
PRESENTATION_ORDERS = tuple(permutations(('A', 'B', 'C')))

# Canonical slot numbers used by every downstream CSV and report: A=NLG, B=model, C=SWUM.
SYSTEM_TO_CANONICAL_NUMBER = {'A': '1', 'B': '2', 'C': '3'}


def presentation_order(*seed_parts: object) -> tuple[str, str, str]:
    """Pick one of the six A/B/C orderings deterministically from ``seed_parts``.

    Deterministic rather than random so that a rerun reproduces the same layout and
    the order recorded in the CSV can be checked after the fact. MD5 is used because
    Python's hash() is salted per interpreter and would not be stable across runs.
    """
    digest = hashlib.md5('||'.join(str(part) for part in seed_parts).encode('utf-8')).hexdigest()
    return PRESENTATION_ORDERS[int(digest, 16) % len(PRESENTATION_ORDERS)]


@dataclass(frozen=True)
class RetryPolicy:
    """How a ranking call is retried. Every value is supplied by the .env file."""

    attempts: int
    backoff_seconds: tuple[int, ...]
    request_timeout_seconds: int

    def delay_before(self, attempt: int) -> int:
        """Seconds to wait before ``attempt`` (1-based); the last delay repeats if needed."""
        index = min(attempt - 2, len(self.backoff_seconds) - 1)
        return self.backoff_seconds[index]


def _require_env_int(name: str) -> int:
    """Read a positive integer from the environment, or explain what is missing."""
    raw = os.getenv(name)
    if not raw or not raw.strip():
        raise ValueError(f'{name} not found in .env file')
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer, got {raw!r}') from exc
    if value <= 0:
        raise ValueError(f'{name} must be positive, got {value}')
    return value


def resolve_retry_policy() -> RetryPolicy:
    """Build the retry policy from .env.

    The previous loop retried three times with no pause at all, which is the one thing
    that cannot help against the failure it was written for: a 429 resent immediately is
    a 429 again, and all three attempts were spent inside a second.
    """
    attempts = _require_env_int('RANK_SUMMARIES_RETRY_ATTEMPTS')

    raw_backoff = os.getenv('RANK_SUMMARIES_RETRY_BACKOFF_SECONDS')
    if not raw_backoff or not raw_backoff.strip():
        raise ValueError('RANK_SUMMARIES_RETRY_BACKOFF_SECONDS not found in .env file')
    try:
        backoff = tuple(int(part.strip()) for part in raw_backoff.split(',') if part.strip())
    except ValueError as exc:
        raise ValueError(
            'RANK_SUMMARIES_RETRY_BACKOFF_SECONDS must be comma-separated integers, '
            f'got {raw_backoff!r}'
        ) from exc
    if not backoff:
        raise ValueError('RANK_SUMMARIES_RETRY_BACKOFF_SECONDS must list at least one delay')
    if len(backoff) < attempts - 1:
        raise ValueError(
            f'RANK_SUMMARIES_RETRY_BACKOFF_SECONDS lists {len(backoff)} delay(s) but '
            f'RANK_SUMMARIES_RETRY_ATTEMPTS={attempts} needs {attempts - 1}'
        )

    return RetryPolicy(
        attempts=attempts,
        backoff_seconds=backoff,
        request_timeout_seconds=_require_env_int('RANK_SUMMARIES_REQUEST_TIMEOUT_SECONDS'),
    )


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


# Slot B of every comparison is the LLM under test. The roster is read from .env rather
# than listed in code: a hardcoded list is exactly how LLM_GEMINI_SUMMARY.csv outlived its
# data. Gemini was replaced by Qwen in .env, but three scripts went on asking for a Gemini
# CSV that no longer exists — summary_length_stats.py still fails outright because of it.
_MODEL_ENV_SUFFIX = '_MODEL'
# Ends in _MODEL but names the judge, not a system being judged.
_NON_SUMMARY_MODEL_KEYS = frozenset({'RANK_SUMMARIES_MODEL'})


@dataclass(frozen=True)
class ComparisonModel:
    """One summarisation model configured in .env, with the files it produces."""

    name: str            # e.g. 'QWEN', taken from the .env key QWEN_MODEL
    model_id: str        # e.g. 'qwen/qwen3.7-plus'

    @property
    def summary_csv(self) -> str:
        return f'LLM_{self.name}_SUMMARY.csv'

    @property
    def nc_summary_csv(self) -> str:
        """Narrative-context variant of the summary CSV."""
        return f'LLM_{self.name}_NC_SUMMARY.csv'


def resolve_model_roster(env_path: Path) -> list[ComparisonModel]:
    """Read every summarisation model from .env, in the order the file declares them.

    The .env file is parsed directly rather than read back out of os.environ so an
    unrelated shell variable ending in _MODEL cannot join the roster.
    """
    values = dotenv_values(env_path)
    roster = [
        ComparisonModel(name=key[: -len(_MODEL_ENV_SUFFIX)], model_id=value.strip())
        for key, value in values.items()
        if key.endswith(_MODEL_ENV_SUFFIX)
        and key not in _NON_SUMMARY_MODEL_KEYS
        and value
        and value.strip()
    ]
    if not roster:
        raise ValueError(
            f'No summarisation models found in {env_path}. Expected at least one key of '
            f'the form <NAME>{_MODEL_ENV_SUFFIX}, for example QWEN_MODEL=qwen/qwen3.7-plus.'
        )
    return roster


def resolve_no_reasoning_models(env_path: Path) -> frozenset[str]:
    """Model ids whose internal chain of thought must be switched off.

    LLM_NO_REASONING_MODELS lists labels (QWEN, ...); each is resolved through the matching
    <LABEL>_MODEL key so this makes exactly the same decision the Java summarisation pipeline
    makes from the same configuration.
    """
    values = dotenv_values(env_path)
    raw = values.get('LLM_NO_REASONING_MODELS') or ''
    labels = {part.strip().upper() for part in re.split(r'[;,]', raw) if part.strip()}
    return frozenset(
        value.strip()
        for label in labels
        if (value := values.get(f'{label}{_MODEL_ENV_SUFFIX}'))
    )


def should_disable_reasoning(model_id: str, env_path: Path) -> bool:
    """Whether ``model_id`` is a hybrid reasoning model configured to answer directly.

    Reasoning tokens count against max_tokens, and RANK_SUMMARIES_MAX_TOKENS is only 50 —
    a reasoning judge would spend the whole budget thinking and return no ranking at all,
    for every criterion of every file.
    """
    return model_id.strip() in resolve_no_reasoning_models(env_path)


def resolve_checkpoint_dir(base_dir: Path) -> Path:
    """Directory holding resume checkpoints, from .env; relative paths resolve under DPS_LLM."""
    raw = os.getenv('RANK_SUMMARIES_CHECKPOINT_DIR')
    if not raw or not raw.strip():
        raise ValueError('RANK_SUMMARIES_CHECKPOINT_DIR not found in .env file')
    path = Path(raw.strip())
    return path if path.is_absolute() else (base_dir / path)


class RankingCheckpoint:
    """Append-only log of completed rankings so an interrupted run resumes.

    A full pipeline is thousands of billed API calls taking hours; until now a crash,
    a dropped connection or a Ctrl-C at the last comparison discarded every call that had
    already succeeded. Each finished row is appended as one JSON line and flushed
    immediately, so the file stays readable even if the process dies mid-run.

    The first line records what the run was configured with. Resuming across a change of
    ranking model or prompt would silently blend results from two different judges, so a
    mismatch refuses to resume instead of quietly reusing the old rows.
    """

    def __init__(self, path: Path, fingerprint: dict, enabled: bool = True) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.enabled = enabled
        self._handle = None

    def load(self) -> dict[str, dict]:
        """Return completed rows keyed by row id, or empty when not resuming."""
        if not self.enabled or not self.path.exists():
            return {}

        completed: dict[str, dict] = {}
        header = None
        with self.path.open('r', encoding='utf-8') as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # Only the final line can be partial, because writes are appended and
                    # flushed one row at a time. Anything earlier means real corruption.
                    print(f'  Checkpoint {self.path.name}: discarding incomplete final '
                          f'line {line_number}')
                    break
                if '_meta' in entry:
                    header = entry['_meta']
                    continue
                completed[entry['_key']] = entry['record']

        if header is not None and header != self.fingerprint:
            differing = sorted(
                key for key in set(header) | set(self.fingerprint)
                if header.get(key) != self.fingerprint.get(key)
            )
            raise ValueError(
                f'Checkpoint {self.path} was written under a different configuration '
                f'({", ".join(differing)} changed), so resuming would mix results from two '
                f'different runs. Delete the file to start over, or pass --no-resume.'
            )

        if completed:
            print(f'  Resuming from {self.path.name}: {len(completed)} row(s) already ranked')
        return completed

    def open(self, completed: dict[str, dict]) -> None:
        """Open the log for appending, writing the header when starting fresh."""
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not completed
        if fresh and self.path.exists():
            self.path.unlink()
        self._handle = self.path.open('a', encoding='utf-8')
        if fresh:
            self._write({'_meta': self.fingerprint})

    def record(self, key: str, payload: dict) -> None:
        """Append one completed row and flush it to disk."""
        if self._handle is None:
            return
        self._write({'_key': key, 'record': payload})

    def _write(self, entry: dict) -> None:
        self._handle.write(json.dumps(entry, default=str) + '\n')
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class MultiCriteriaRanker:
    """Ranks summaries using multiple criteria via LLM API."""

    CRITERIA = {
        'accuracy': 'accuracy',
        'conciseness': 'conciseness',
        'adequacy': 'adequacy',
        'code_context': 'context',
        'design_patterns': 'pattern'
    }

    def __init__(self, api_key: str, api_url: str, model: str, prompts: dict[str, str],
                 max_tokens: int, retry_policy: RetryPolicy | None = None,
                 disable_reasoning: bool = False) -> None:
        """Initialise ranker with API configuration and prompt templates.

        ``retry_policy`` defaults to the one described by the .env file, so every caller
        picks up the configured behaviour without repeating it. ``disable_reasoning`` is
        set by the pipelines from LLM_NO_REASONING_MODELS.
        """
        # Prompts are provided externally via JSON so updates do not require code edits.
        self.prompts = prompts
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.retry_policy = retry_policy if retry_policy is not None else resolve_retry_policy()
        self.disable_reasoning = disable_reasoning

    def rank_single_criterion(self, human_summary, summaries, criterion_name, criterion_key,
                              order=('A', 'B', 'C')):
        """
        Rank three summaries on a single criterion using LLM.

        ``summaries`` maps the canonical system slots ('A', 'B', 'C') to summary text
        and ``order`` is the sequence those systems are presented in. The judge sees an
        ordinary 1/2/3 list and answers in list positions; the result is translated back
        to canonical numbers (1=A, 2=B, 3=C) before returning, so every caller and every
        CSV column keeps exactly the meaning it had before order was randomised.

        Returns:
            dict: Rankings for each method (1=best, 3=worst) and reasoning
        """
        # Build criterion-specific prompt using dedicated methods
        template = self.prompts.get(criterion_name)
        if not template:
            template = "Rank summaries 1, 2, 3 from best to worst. Output only the ranking."
        prompt = template.format(
            human_summary=human_summary,
            summary_a=summaries[order[0]],
            summary_b=summaries[order[1]],
            summary_c=summaries[order[2]],
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

        # A hybrid reasoning judge would spend the whole RANK_SUMMARIES_MAX_TOKENS budget on
        # its chain of thought and return no ranking. Same OpenRouter field, same .env
        # roster, and the same reason as the summarisation pipeline.
        if self.disable_reasoning:
            data["reasoning"] = {"enabled": False}

        # Log the selected ranking model before each request so one-sample runs can confirm the target model.
        print(f"Calling ranking model: {self.model}")

        policy = self.retry_policy
        last_error = None

        for attempt in range(1, policy.attempts + 1):
            if attempt > 1:
                delay = policy.delay_before(attempt)
                print(f"    transient failure ({last_error}); waiting {delay}s before "
                      f"retry {attempt - 1}/{policy.attempts - 1}")
                time.sleep(delay)

            try:
                response = requests.post(
                    self.api_url, headers=headers, json=data,
                    timeout=policy.request_timeout_seconds,
                )
            except requests.RequestException as exc:
                # Connection reset, DNS failure, read timeout — all worth resending.
                last_error = type(exc).__name__
                continue

            if self._is_retryable_status(response.status_code):
                last_error = f"HTTP {response.status_code}"
                continue

            if not response.ok:
                # A 4xx other than 429 is a problem with the request, not the network;
                # resending an identical payload cannot fix it.
                print(f"    ERROR calling API: HTTP {response.status_code} (not retryable)")
                return None

            try:
                payload = response.json()
                choice = payload['choices'][0]
                content = (choice['message']['content'] or '').strip()
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = f"malformed response ({type(exc).__name__})"
                continue

            # RANK_SUMMARIES_MAX_TOKENS is deliberately small, so a judge that starts
            # explaining itself is cut off mid-sentence. Parsing whatever digits survived
            # would invent a ranking the judge never finished stating.
            if choice.get('finish_reason') == 'length':
                reasoning_tokens = (
                    payload.get('usage', {})
                    .get('completion_tokens_details', {})
                    .get('reasoning_tokens', 0)
                )
                if reasoning_tokens and not self.disable_reasoning:
                    print(f"response truncated: {self.model} spent {reasoning_tokens} of "
                          f"{self.max_tokens} tokens reasoning and never answered. Add its "
                          f"label to LLM_NO_REASONING_MODELS in .env.")
                else:
                    print(f"response truncated at {self.max_tokens} tokens; skipping criterion")
                return None

            rankings = self._parse_ranking_output(content)
            if rankings is None:
                # Invalid/ambiguous output; don't bias results. Temperature is 0, so a
                # retry would return the same unusable text.
                print(f"invalid parse ({content!r}); skipping criterion")
                return None
            return self._to_canonical(rankings, order)

        print(f"    ERROR calling API after {policy.attempts} attempts: {last_error}")
        return None

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        """Whether resending an identical request is worth attempting.

        Mirrors LlmClient.isRetryableStatus on the Java side: 429 is rate limiting and
        5xx is a transient provider failure; any other 4xx is a problem with the request.
        """
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _to_canonical(rankings, order):
        """Translate slot-position answers back to canonical system numbers.

        The judge ranks list positions, which ``order`` maps to systems; this undoes
        that mapping so downstream code continues to read 1=A, 2=B, 3=C regardless of
        the layout the judge actually saw.
        """
        canonical = {'reasoning': rankings.get('reasoning', '')}
        for place in ('1', '2', '3'):
            slot = rankings.get(place)
            if slot not in ('1', '2', '3'):
                return None
            canonical[place] = SYSTEM_TO_CANONICAL_NUMBER[order[int(slot) - 1]]
        return canonical

    # The exact reply the prompt asks for: three digits with nothing between them but
    # separators. Anchored, so any commentary at all disqualifies this branch.
    _EXACT_PLACEMENT_PATTERN = re.compile(r"\s*[123](?:\s*[,>;/-]?\s*[123]){2}\s*\.?\s*")

    # Labelled ordinal form, e.g. "1st: 2, 2nd: 3, 3rd: 1". Each rank says which summary
    # holds it, so this stays unambiguous even when the reply is wordy.
    _ORDINAL_PATTERN = re.compile(
        r"1(?:st)?\D*([123]).*?2(?:nd)?\D*([123]).*?3(?:rd)?\D*([123])",
        re.IGNORECASE | re.DOTALL,
    )

    @classmethod
    def _parse_ranking_output(cls, content):
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

        Ambiguity is rejected rather than guessed at. The previous version took the first
        three digits appearing anywhere in the reply, so a single stray number silently
        shifted the whole ranking — "For criterion 1: 2, 3, 1" was read as 1>2>3 instead of
        2>3>1, and since that is still a valid permutation nothing downstream could tell.
        A rejected criterion is recorded as skipped, which is visible; a wrong one is not.
        """
        text = (content or "").strip()
        if not text:
            return None

        def placement(first: str, second: str, third: str):
            if {first, second, third} != {"1", "2", "3"}:
                return None
            return {"1": first, "2": second, "3": third, "reasoning": text}

        # 1. The canonical reply, e.g. "2,1,3" — nothing but the answer.
        if cls._EXACT_PLACEMENT_PATTERN.fullmatch(text):
            digits = re.findall(r"[123]", text)
            result = placement(*digits)
            if result is not None:
                return result

        # 2. Labelled ordinal form, which names the rank each summary holds.
        ordinal = cls._ORDINAL_PATTERN.findall(text)
        if ordinal:
            result = placement(*ordinal[0])
            if result is not None:
                return result

        # 3. Unlabelled placement inside a short phrase, e.g. "Summary 3, then 1, then 2".
        #    Accepted only when the reply contains exactly three rank digits in total, so
        #    no stray number can displace the answer.
        digits = re.findall(r"[123]", text)
        if len(digits) == 3:
            result = placement(*digits)
            if result is not None:
                return result

        # Unable to parse confidently
        return None

    def rank_summaries_all_criteria(self, human_summary, summary_a, summary_b, summary_c,
                                    file_name, project_name, labels=None, order_seed=None):
        """
        Rank summaries on all 5 criteria.

        ``order_seed`` seeds the presentation-order permutation. Callers that run the
        same class file through several comparisons should include the comparison name
        so each one draws its own layout; the default identifies the class file alone.

        Returns:
            dict: Results containing rankings for each criterion and aggregate statistics
        """
        if labels is None:
            labels = {'A': 'A', 'B': 'B', 'C': 'C'}
        if order_seed is None:
            order_seed = (project_name, file_name)
        summaries = {'A': summary_a, 'B': summary_b, 'C': summary_c}
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

            # Vary which system occupies which list position so the judge's position
            # bias is spread evenly across the three systems rather than accruing to
            # one of them. The layout is recorded so the effect can be audited later.
            order = presentation_order(order_seed, criterion_name)
            results[f'{criterion_name}_presentation_order'] = ''.join(order)

            ranking = self.rank_single_criterion(
                human_summary, summaries, criterion_name, criterion_key, order
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

    # Slot B is the variable LLM model; A=NLG and C=SWUM are always fixed. The roster is
    # built per run from .env (see resolve_model_roster), so adding or retiring a model
    # is a .env edit rather than a code change in three separate files.
    NLG_CSV  = 'nlg_summaries.csv'
    SWUM_CSV = 'swum_summaries.csv'

    def __init__(self, model_override: str | None = None, limit: int | None = None,
                 resume: bool = True) -> None:
        """Initialise pipeline, loading config and credentials from the .env file."""
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.limit = limit
        self.resume = resume

        self.output_dir  = (base_dir / 'output' / 'summary-output').resolve()
        self.input_dir   = (base_dir / 'input').resolve()
        self.results_dir = (base_dir / 'evaluation-results').resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        env_path = base_dir.parent / '.env'
        load_dotenv(env_path)

        self.comparisons = [
            (entry.name, entry.summary_csv) for entry in resolve_model_roster(env_path)
        ]
        print('Comparison roster from .env: '
              + ', '.join(f'{name} ({csv})' for name, csv in self.comparisons))

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


        # Disable the judge's chain of thought when .env says it is a hybrid reasoning
        # model; at RANK_SUMMARIES_MAX_TOKENS it would otherwise never reach an answer.
        disable_reasoning = should_disable_reasoning(model, env_path)
        if disable_reasoning:
            print('Reasoning disabled for the ranking model per LLM_NO_REASONING_MODELS.')

        self.ranker = MultiCriteriaRanker(
            api_key=api_key,
            api_url=api_url,
            model=model,
            prompts=ranking_prompts,
            max_tokens=max_tokens,
            disable_reasoning=disable_reasoning,
        )

        self.checkpoint_dir = resolve_checkpoint_dir(base_dir)
        # Identifies the configuration a checkpoint was written under. Resuming across a
        # change to the judge, its prompts or its token budget would silently mix two
        # different judges' verdicts into one result set.
        self.run_fingerprint = {
            'ranking_model': model,
            'max_tokens': max_tokens,
            'prompts_digest': hashlib.md5(
                json.dumps(ranking_prompts, sort_keys=True).encode('utf-8')
            ).hexdigest(),
        }

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
        folder_col = resolve_folder_column(df, filename)
        df['match_key'] = df.apply(
            lambda row: build_match_key(row['project_name'], row[folder_col], row['file_name']),
            axis=1,
        )
        assert_unique_match_keys(df, filename)
        return df

    def load_human_summaries(self) -> pd.DataFrame:
        """Load and normalise human summary references from input/human_summaries/DPS_Human_Summaries.csv."""
        human_path = self.input_dir / 'human_summaries' / 'DPS_Human_Summaries.csv'
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
        folder_col = resolve_folder_column(df, 'DPS_Human_Summaries.csv')
        df['match_key'] = df.apply(
            lambda row: build_match_key(row['project'], row[folder_col], row['file_name']),
            axis=1,
        )
        assert_unique_match_keys(df, 'DPS_Human_Summaries.csv')

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

        checkpoint = RankingCheckpoint(
            path=self.checkpoint_dir / f'rank_summaries_{comparison_name.lower()}.jsonl',
            fingerprint={**self.run_fingerprint, 'comparison': comparison_name},
            enabled=self.resume,
        )
        completed = checkpoint.load()
        checkpoint.open(completed)
        resumed_count = 0

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

            previous = completed.get(match_key)
            if previous is not None:
                # Already paid for in an earlier run; replay it rather than re-billing.
                results.append(previous)
                if previous.get('status') == 'ranked':
                    ranked_count += 1
                else:
                    skipped_count += 1
                resumed_count += 1
                continue

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
                checkpoint.record(match_key, result)
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
                order_seed=(comparison_name, match_key),
            )

            ranking_result['comparison']      = comparison_name
            ranking_result['llm_source_file'] = llm_filename
            ranking_result['status']          = 'ranked'
            ranking_result['missing_methods'] = ''
            ranking_result['match_key']       = match_key

            results.append(ranking_result)
            checkpoint.record(match_key, ranking_result)
            ranked_count += 1

        checkpoint.close()
        resumed_note = f', Resumed={resumed_count}' if resumed_count else ''
        print(f'\nComparison {comparison_name} complete: Ranked={ranked_count}, '
              f'Skipped={skipped_count}{resumed_note}')
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

        total_api_calls = len(df_human) * len(self.CRITERIA) * len(self.comparisons)
        print(f'\nUp to {total_api_calls} API calls across {len(self.comparisons)} comparisons '
              f'({len(df_human)} summaries × {len(self.CRITERIA)} criteria × {len(self.comparisons)} models)')

        all_results: list[list[dict]] = []
        all_stats:   list[dict]       = []

        for comparison_name, llm_filename in self.comparisons:
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
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help=(
            'Ignore any existing checkpoint and re-rank every row from scratch. '
            'Without this, a previously interrupted run continues where it stopped.'
        ),
    )
    parser.set_defaults(resume=True)
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
        pipeline = SummaryRankingPipeline(
            model_override=args.model, limit=args.limit, resume=args.resume
        )

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
