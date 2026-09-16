"""pr_llm_summariser.py

Pipeline for generating pull-request summaries using multiple LLMs (GPT, Qwen,
Claude, Mistral) via the OpenRouter API.

Workflow
--------
1. Load commit records (articles) from a JSONL / CSV dataset in input/dataset/.
   Gold descriptions are read from input/human_summaries/ and every record is
   keyed by the canonical 6-digit ``pr_id``.
2. For each record, call every configured LLM to produce a PR description.
3. Persist per-sample results to one JSONL file per model (summary text only).

Metric scoring (BERTScore + TF-IDF cosine similarity) is handled by the
companion script evaluate_pr_summaries.py. Rubric-based LLM ranking is
handled by rank_pr_summaries.py.

Configuration is external to this file:

* Prompts come from ``resources/prompts.json`` — ``pr_llm.system_prompt`` for
  the system message and ``pr_llm.user_prompt_template.sections`` for the
  per-sample user message.
* Model identifiers come from ``.env`` (GPT_MODEL, QWEN_MODEL, CLAUDE_MODEL,
  MISTRAL_MODEL), as do the endpoint (OPENROUTER_API_URL or
  OPENROUTER_BASE_URL), OPENROUTER_TEMPERATURE, OPENROUTER_MAX_COMPLETION_TOKENS
  and OPENROUTER_TIMEOUT.

So neither a model version nor any prompt wording is hardcoded in source.

:author: Najam Nazar
:version: 1.0.0
:date: 2026-02-23
:license: MIT
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------
__author__ = "Najam Nazar"
__version__ = "1.0.0"
__date__ = "2026-02-23"
__license__ = "MIT"

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv
from openai import OpenAI

import pr_corpus

# Load .env once at module import time so all classes share the same
# environment state without each needing to call load_dotenv() individually.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Configure logging before any third-party library can install its own handler.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logging.getLogger(__name__).addHandler(_console)
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger(__name__).propagate = False
_log = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class SummarySample:
    """Immutable data container for a single commit / PR record.

    Attributes
    ----------
    pr_id:             Canonical zero-padded corpus id (e.g. ``003485``). This is
                       the key every downstream script joins on.
    rank:              Curated ranking position in the selected dataset.
    dataset_index:     Original row index in the full upstream dataset.
    identifier:        Unique string ID (e.g. commit SHA or PR number).
    article:           Raw text of commit messages and inline code comments
                       that the LLM will summarise.
    reference_summary: Human-written ground-truth PR description used for
                       evaluation.
    """

    pr_id: str
    rank: int
    dataset_index: int
    identifier: str
    article: str
    reference_summary: str


class CommitDatasetLoader:
    """Load commit records from the curated dataset.

    Supports both JSONL and CSV formats.  Each row is normalised into a
    :class:`SummarySample` so downstream code is format-agnostic.

    The gold description is taken from ``input/human_summaries/`` — the same
    source the evaluation and ranking scripts read — rather than from a column
    inside the dataset file, so every stage of the pipeline compares against
    one authoritative reference. A dataset row whose ``pr_id`` has no human
    summary falls back to the row's own ``abstract`` column.
    """

    def __init__(self, dataset_path: Path, human_summaries: Optional[Dict[str, str]] = None) -> None:
        self.dataset_path = dataset_path
        self.human_summaries = human_summaries or {}

    def load(self, limit: Optional[int] = None) -> List[SummarySample]:
        """Read the dataset and return a list of :class:`SummarySample` objects.

        Parameters
        ----------
        limit:
            If provided, only the first *limit* rows are read.

        Raises
        ------
        FileNotFoundError:
            If :attr:`dataset_path` does not exist on disk.
        ValueError:
            If the file extension is neither ``.jsonl`` nor ``.csv``.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        if self.dataset_path.suffix.lower() == ".jsonl":
            rows = self._load_jsonl(limit)
        elif self.dataset_path.suffix.lower() == ".csv":
            rows = self._load_csv(limit)
        else:
            raise ValueError("Dataset must be .jsonl or .csv")

        samples: List[SummarySample] = []
        for row in rows:
            pr_id = pr_corpus.record_pr_id(row) or ""
            samples.append(
                SummarySample(
                    pr_id=pr_id,
                    rank=int(row.get("rank", 0) or 0),
                    dataset_index=int(row.get("dataset_index", -1) or -1),
                    identifier=str(row.get("id", "")),
                    article=str(row.get("article", "")),
                    reference_summary=self.human_summaries.get(
                        pr_id, str(row.get("abstract", ""))
                    ),
                )
            )
        return samples

    def _load_jsonl(self, limit: Optional[int]) -> List[dict]:
        rows: List[dict] = []
        with self.dataset_path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load_csv(self, limit: Optional[int]) -> List[dict]:
        rows: List[dict] = []
        with self.dataset_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    break
                rows.append(row)
        return rows


class PromptLoader:
    """Read the PR-LLM prompts from ``<repo root>/resources/prompts.json``.

    Both halves of every request live under the ``pr_llm`` namespace: the
    ``system_prompt`` that sets the summariser's role, and the
    ``user_prompt_template.sections`` that are rendered per sample. Loading both
    from the same file means the wording of every LLM call in the pipeline is
    edited in one place, alongside the ``pr_ranking`` prompts the judge uses.

    .. note::
       Previously only ``system_prompt`` was read here and the user prompt was
       duplicated as a hardcoded f-string in
       :meth:`PullRequestSummarizer._build_user_prompt`, so edits to
       ``user_prompt_template`` in the JSON had no effect on what the models
       received. The template is now the single source of truth.
    """

    def __init__(self, prompt_path: Path) -> None:
        self.prompt_path = prompt_path
        # Parsed lazily and cached: both loaders read the same file.
        self._payload: Optional[dict] = None

    def _pr_llm(self) -> dict:
        """Return the ``pr_llm`` section, reading the file on first use.

        Raises
        ------
        FileNotFoundError:
            If the JSON file does not exist at :attr:`prompt_path`.
        ValueError:
            If the file has no ``"pr_llm"`` namespace.
        """
        if self._payload is None:
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Prompt file missing: {self.prompt_path}")
            with self.prompt_path.open(encoding="utf-8") as handle:
                self._payload = json.load(handle)
        section = self._payload.get("pr_llm")
        if not isinstance(section, dict):
            raise ValueError(f"{self.prompt_path} has no 'pr_llm' section")
        return section

    def load_system_prompt(self) -> str:
        """Read and return the system prompt string.

        Raises
        ------
        ValueError:
            If the ``"system_prompt"`` field is missing or blank.
        """
        prompt = self._pr_llm().get("system_prompt", "").strip()
        if not prompt:
            raise ValueError("system_prompt is empty in the provided prompt file")
        return prompt

    def load_user_sections(self) -> List[str]:
        """Return the user-prompt sections from ``pr_llm.user_prompt_template``.

        Each section is a format string over the :class:`SummarySample` fields
        (``{rank}``, ``{dataset_index}``, ``{identifier}``, ``{article}``);
        :meth:`PullRequestSummarizer._build_user_prompt` joins them with
        newlines to form the user message.

        Raises
        ------
        ValueError:
            If ``user_prompt_template.sections`` is absent, empty, or holds
            anything other than strings.
        """
        template = self._pr_llm().get("user_prompt_template")
        sections = template.get("sections") if isinstance(template, dict) else None
        if not isinstance(sections, list) or not sections:
            raise ValueError(
                f"{self.prompt_path} is missing 'pr_llm.user_prompt_template.sections' "
                "(expected a non-empty list of format strings)"
            )
        if not all(isinstance(section, str) for section in sections):
            raise ValueError("Every entry of user_prompt_template.sections must be a string")
        return sections


class OpenRouterLLMClient:
    """Lightweight client for OpenRouter-powered chat completions.

    Wraps the OpenAI SDK, pointing it at the OpenRouter base URL so that any
    model available on OpenRouter can be addressed with a single client.
    """

    # Endpoint path the OpenAI SDK appends itself; stripped from OPENROUTER_API_URL
    # so either form of the variable works.
    _CHAT_COMPLETIONS_PATH = "/chat/completions"
    _DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    @classmethod
    def _resolve_base_url(cls) -> str:
        """Resolve the API base URL from ``.env``.

        Accepts either ``OPENROUTER_BASE_URL`` (already a base, e.g.
        ``https://openrouter.ai/api/v1``) or ``OPENROUTER_API_URL`` (the full
        endpoint ``.../v1/chat/completions``, which is what this repo's ``.env``
        defines). The OpenAI SDK appends ``/chat/completions`` itself, so that
        suffix is trimmed rather than passed through — previously only
        ``OPENROUTER_BASE_URL`` was read, so the configured value was ignored
        and the hardcoded default always won.
        """
        url = (os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENROUTER_API_URL") or "").strip()
        if not url:
            return cls._DEFAULT_BASE_URL
        url = url.rstrip("/")
        if url.endswith(cls._CHAT_COMPLETIONS_PATH):
            url = url[: -len(cls._CHAT_COMPLETIONS_PATH)]
        return url

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is required but was not found.")
        self.base_url = base_url or self._resolve_base_url()
        # Request timeout is configurable via .env (OPENROUTER_TIMEOUT), 120s default.
        if request_timeout is None:
            request_timeout = int(os.getenv("OPENROUTER_TIMEOUT", "120"))

        default_headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers or None,
            timeout=request_timeout,
        )

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        disable_reasoning: bool = False,
    ) -> str:
        """Send a chat-completion request and return the generated text.

        Parameters
        ----------
        disable_reasoning:
            Suppress a hybrid reasoning model's internal chain of thought via
            OpenRouter's ``reasoning`` field. See
            :attr:`PullRequestSummarizer.NO_REASONING_LABELS` for why this is
            needed for Qwen.
        """
        # OpenRouter-specific request field, passed through by the OpenAI SDK.
        extra_body = {"reasoning": {"enabled": False}} if disable_reasoning else None

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Model '{model}' returned null content in the response.")
        return content.strip()


class PullRequestSummarizer:
    """Generate PR-description summaries for each sample using multiple LLMs."""

    # Models whose internal chain of thought must be switched off.
    #
    # Every current-generation Qwen on OpenRouter (3.5 through 3.8, including
    # qwen3.7-plus) is a hybrid reasoning model that thinks before answering.
    # Under this pipeline's token budget it spent the whole allowance on that
    # chain of thought and hit finish_reason='length' before emitting any
    # answer, so the API returned content=None on every item:
    #
    #     completion_tokens: 512, reasoning_tokens: 512, content: None
    #
    # Two other fixes were rejected as worse for the study. Raising
    # max_tokens only for Qwen would give it extended reasoning the other three
    # models do not get, confounding the comparison the ranking and Friedman
    # tests are built on; and the only non-reasoning Qwen models on OpenRouter
    # are a generation behind GPT-5.4-mini, Claude Sonnet 4.6 and
    # Mistral Small 2603. Disabling reasoning keeps qwen3.7-plus in the lineup
    # and has all four models answer directly under the same budget.
    NO_REASONING_LABELS = frozenset({"PR_QWEN_SUMMARY"})

    def __init__(
        self,
        client: OpenRouterLLMClient,
        system_prompt: str,
        user_sections: List[str],
        model_map: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        # Format strings from prompts.json (pr_llm.user_prompt_template.sections),
        # rendered per sample by _build_user_prompt().
        self.user_sections = user_sections
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_map = model_map or self._default_models()

    @staticmethod
    def _default_models() -> Dict[str, str]:
        """Build the model map from environment variables defined in ``.env``."""
        return {
            "PR_GPT_SUMMARY": os.getenv("GPT_MODEL"),
            "PR_QWEN_SUMMARY": os.getenv("QWEN_MODEL"),
            "PR_CLAUDE_SUMMARY": os.getenv("CLAUDE_MODEL"),
            "PR_MISTRAL_SUMMARY": os.getenv("MISTRAL_MODEL"),
        }

    def _build_user_prompt(self, sample: SummarySample) -> str:
        """Render the user message for *sample* from the prompts.json template.

        The sections are filled from the sample's fields and joined with
        newlines. Substituted values are not re-scanned for placeholders, so an
        article containing literal braces is safe.

        Raises
        ------
        ValueError:
            If a section references a field the sample does not provide — a
            typo in prompts.json is reported by name rather than surfacing as a
            bare ``KeyError`` mid-run.
        """
        fields = {
            "rank": sample.rank,
            "dataset_index": sample.dataset_index,
            "identifier": sample.identifier,
            "article": sample.article,
            "pr_id": sample.pr_id,
        }
        try:
            return "\n".join(section.format(**fields) for section in self.user_sections)
        except KeyError as exc:
            raise ValueError(
                f"Unknown placeholder {exc} in pr_llm.user_prompt_template.sections; "
                f"available fields: {', '.join(sorted(fields))}"
            ) from exc

    def summarize(self, sample: SummarySample) -> Dict[str, str]:
        """Run all configured models against one sample and return their outputs."""
        user_prompt = self._build_user_prompt(sample)
        results: Dict[str, str] = {}
        for label, model_name in self.model_map.items():
            print(f"  -> Calling {label} ({model_name}) ...", flush=True)
            try:
                results[label] = self.client.generate(
                    model=model_name,
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    # Qwen only — see NO_REASONING_LABELS above.
                    disable_reasoning=label in self.NO_REASONING_LABELS,
                )
            except openai.RateLimitError as exc:
                _log.error("[%s] Rate limit exceeded: %s", label, exc)
                results[label] = ""
            except openai.APITimeoutError as exc:
                _log.error("[%s] Request timed out: %s", label, exc)
                results[label] = ""
            except openai.APIConnectionError as exc:
                _log.error("[%s] Connection error: %s", label, exc)
                results[label] = ""
            except openai.APIStatusError as exc:
                _log.error("[%s] API status %s: %s", label, exc.status_code, exc.message)
                results[label] = ""
            except Exception as exc:  # noqa: BLE001
                _log.error("[%s] Unexpected error for sample '%s': %s", label, sample.identifier, exc)
                results[label] = ""
        return results


class ResultWriter:
    """Persist per-sample results incrementally to a JSONL file."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Always start fresh — truncate any existing file.
        self.output_path.write_text("", encoding="utf-8")

    def append(self, record: dict) -> None:
        """Serialise *record* to JSON and append it as a new line."""
        line = json.dumps(record, ensure_ascii=False)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class SummarizationPipeline:
    """Orchestrate the load → summarise → write workflow.

    Generates PR descriptions for every configured LLM and writes per-sample
    results (summary text only) to JSONL files. Run evaluate_pr_summaries.py
    next to compute BERTScore and TF-IDF cosine metrics.
    """

    def __init__(
        self,
        loader: CommitDatasetLoader,
        summarizer: PullRequestSummarizer,
        writers: Dict[str, ResultWriter],
    ) -> None:
        self.loader = loader
        self.summarizer = summarizer
        self.writers = writers

    def run(self, limit: Optional[int] = None) -> None:
        """Generate summaries for all samples and write to JSONL files.

        Parameters
        ----------
        limit:
            Process only the first *limit* samples.  ``None`` processes all.
        """
        samples = self.loader.load(limit=limit)
        total = len(samples)

        print(f"[Generating] {total} samples...", flush=True)
        for idx, sample in enumerate(samples, start=1):
            print(f"[{idx}/{total}] pr_id={sample.pr_id}  id={sample.identifier}", flush=True)
            try:
                generated = self.summarizer.summarize(sample)
                for label, summary_text in generated.items():
                    self.writers[label].append({
                        "pr_id": sample.pr_id,
                        "rank": sample.rank,
                        "dataset_index": sample.dataset_index,
                        "id": sample.identifier,
                        "reference_summary": sample.reference_summary,
                        "summary": summary_text,
                    })
            except Exception as exc:  # noqa: BLE001
                _log.error("Sample '%s' generation failed — skipping: %s", sample.identifier, exc)

        output_dir = next(iter(self.writers.values())).output_path.parent
        print(f"\n[Done] Summaries written to: {output_dir}", flush=True)
        print("Next: run evaluate_pr_summaries.py to compute BERTScore and cosine metrics.", flush=True)


# Maps short CLI names to the corresponding _default_models() key.
_MODEL_ALIASES: Dict[str, str] = {
    "GPT": "PR_GPT_SUMMARY",
    "QWEN": "PR_QWEN_SUMMARY",
    "CLAUDE": "PR_CLAUDE_SUMMARY",
    "MISTRAL": "PR_MISTRAL_SUMMARY",
}


def discover_dataset() -> Path:
    """Locate the article dataset inside ``PR_LLM/input/dataset/``.

    The corpus ships the human and tool summaries as plain text directories but
    keeps the PR articles (commit messages and code comments) in a single
    ``.jsonl`` or ``.csv`` file. Prefer JSONL when both are present.

    Raises
    ------
    FileNotFoundError:
        If no dataset file is present, with the expected location spelled out
        so the pipeline fails with a clear instruction rather than an empty run.
    """
    candidates = sorted(pr_corpus.DATASET_DIR.glob("*.jsonl")) + sorted(
        pr_corpus.DATASET_DIR.glob("*.csv")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No article dataset found in {pr_corpus.DATASET_DIR}.\n"
            "The summariser needs the PR articles (commit messages and code "
            "comments) for the corpus in input/human_summaries/. Place a "
            ".jsonl or .csv there with an 'article' column plus a 'ref_file', "
            "'pr_id' or 'dataset_index' column that matches the 6-digit ids, "
            "or pass --dataset explicitly."
        )
    return candidates[0]


def build_default_pipeline(
    dataset_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    prompt_path: Optional[Path] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    models: Optional[List[str]] = None,
) -> SummarizationPipeline:
    """Construct a fully wired :class:`SummarizationPipeline` with sensible defaults.

    Parameters
    ----------
    dataset_path:
        Path to the ``.jsonl`` or ``.csv`` commit dataset.
        Default: the first file discovered in ``<PR_LLM>/input/dataset/``.
    output_path:
        Directory where per-model JSONL summary files are written.
        Default: ``<PR_LLM>/output/``.
    prompt_path:
        Path to ``prompts.json``.
        Default: ``<repo root>/resources/prompts.json``.
    temperature:
        LLM sampling temperature.  Falls back to ``OPENROUTER_TEMPERATURE``
        from the ``.env`` file, or ``0.0`` if unset.
    max_tokens:
        Maximum tokens per generated summary.  Falls back to
        ``OPENROUTER_MAX_COMPLETION_TOKENS`` from the ``.env`` file, or ``512`` if unset.
    models:
        List of short model names to run: any combination of ``GPT``,
        ``QWEN``, ``CLAUDE``, ``MISTRAL``, or ``["ALL"]`` for all four.
        Defaults to ``["ALL"]``.  Case-insensitive.

    Returns
    -------
    SummarizationPipeline
        Ready-to-run pipeline instance.
    """
    if temperature is None:
        temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.0"))
    if max_tokens is None:
        max_tokens = int(os.getenv("OPENROUTER_MAX_COMPLETION_TOKENS", "512"))

    pr_llm_root = pr_corpus.PR_LLM_ROOT
    repo_root = pr_corpus.REPO_ROOT
    dataset = dataset_path or discover_dataset()
    base_output_dir = output_path or pr_corpus.OUTPUT_DIR
    prompt_file = prompt_path or (repo_root / "resources" / "prompts.json")

    # Both the system prompt and the per-sample user template come from
    # prompts.json; nothing about the request wording is hardcoded here.
    prompts = PromptLoader(prompt_file)
    prompt_text = prompts.load_system_prompt()
    user_sections = prompts.load_user_sections()
    client = OpenRouterLLMClient()

    # Resolve which models to run.
    full_map = PullRequestSummarizer._default_models()
    selected = [m.upper() for m in (models or ["ALL"])]
    if selected == ["ALL"]:
        filtered_map = full_map
    else:
        unknown = [m for m in selected if m not in _MODEL_ALIASES]
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. Choose from: ALL, "
                + ", ".join(_MODEL_ALIASES)
            )
        filtered_map = {_MODEL_ALIASES[m]: full_map[_MODEL_ALIASES[m]] for m in selected}

    # Validate that all selected models have identifiers configured in .env.
    _env_key_map = {v: k + "_MODEL" for k, v in _MODEL_ALIASES.items()}
    missing_ids = [
        _env_key_map.get(label, label)
        for label, model_id in filtered_map.items()
        if not model_id
    ]
    if missing_ids:
        raise EnvironmentError(
            f"The following model env variables are not set in .env: {missing_ids}. "
            "Add them before running."
        )

    print("[Models] Active models (from .env):", flush=True)
    for label, model_id in filtered_map.items():
        short = next((k for k, v in _MODEL_ALIASES.items() if v == label), label)
        print(f"  {short:<8} {label:<24} -> {model_id}", flush=True)
    print(flush=True)

    summarizer = PullRequestSummarizer(
        client, prompt_text, user_sections, model_map=filtered_map,
        temperature=temperature, max_tokens=max_tokens,
    )
    writers = {
        label: ResultWriter(base_output_dir / f"{label}.jsonl")
        for label in summarizer.model_map.keys()
    }
    # Gold descriptions come from input/human_summaries/ when present so the
    # summariser, evaluator and ranker all quote the same reference text.
    try:
        human_summaries = pr_corpus.load_human_summaries()
        print(f"[Corpus] {len(human_summaries)} human summaries loaded from "
              f"{pr_corpus.HUMAN_DIR}", flush=True)
    except FileNotFoundError as exc:
        print(f"[Corpus] {exc}\n         Falling back to the dataset's own "
              f"'abstract' column for references.", flush=True)
        human_summaries = {}

    print(f"[Corpus] Article dataset: {dataset}", flush=True)
    loader = CommitDatasetLoader(dataset, human_summaries)
    return SummarizationPipeline(loader, summarizer, writers)


def main() -> None:
    """CLI entry point — parse arguments and run the summarisation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PR summaries via OpenRouter and write them to JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        metavar="MODEL",
        nargs="*",
        default=["ALL"],
        help=(
            "One or more models to run: GPT, QWEN, CLAUDE, MISTRAL, or ALL. "
            "Example: GPT QWEN CLAUDE"
        ),
    )
    parser.add_argument("--dataset", type=Path, help="Path to selected_commits.jsonl or CSV.")
    parser.add_argument("--output", type=Path, help="Directory to write JSONL summary files.")
    parser.add_argument("--prompt", type=Path, help="Path to prompts.json.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional limit for debugging (defaults to all samples).",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: OPENROUTER_TEMPERATURE from .env, else 0.0).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens per generated summary (default: OPENROUTER_MAX_COMPLETION_TOKENS from .env, else 512).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    args = parser.parse_args()

    all_choices_upper = [m.upper() for m in (["ALL"] + list(_MODEL_ALIASES))]
    bad = [m for m in args.models if m.upper() not in all_choices_upper]
    if bad:
        parser.error(f"Unknown model name(s): {bad}. Valid choices: {all_choices_upper}")

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    _log.setLevel(level)
    _log.handlers[0].setLevel(level)

    try:
        pipeline = build_default_pipeline(
            dataset_path=args.dataset,
            output_path=args.output,
            prompt_path=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            models=[m.upper() for m in args.models],
        )
        pipeline.run(limit=args.limit)
    except KeyboardInterrupt:
        _log.info("Run interrupted by user (Ctrl+C).")
        raise SystemExit(130)
    except (EnvironmentError, FileNotFoundError, ValueError) as exc:
        _log.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc
    except OSError as exc:
        _log.error("File-system error: %s", exc)
        raise SystemExit(1) from exc
    except Exception as exc:
        _log.error("Fatal error: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
