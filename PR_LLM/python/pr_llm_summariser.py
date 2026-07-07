"""pr_llm_summariser.py

Pipeline for generating pull-request summaries using multiple LLMs (GPT, Qwen,
Claude, Mistral) via the OpenRouter API.

Workflow
--------
1. Load commit records from a JSONL / CSV dataset.
2. For each record, call every configured LLM to produce a PR description.
3. Persist per-sample results to one JSONL file per model (summary text only).

Metric scoring (BERTScore + TF-IDF cosine similarity) is handled by the
companion script evaluate_pr_summaries.py. Rubric-based LLM ranking is
handled by rank_pr_summaries.py.

Model identifiers are driven exclusively by the .env file
(GPT_MODEL, QWEN_MODEL, CLAUDE_MODEL, MISTRAL_MODEL) so that no versions
need to be hardcoded in source.

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
    rank:              Curated ranking position in the selected dataset.
    dataset_index:     Original row index in the full upstream dataset.
    identifier:        Unique string ID (e.g. commit SHA or PR number).
    article:           Raw text of commit messages and inline code comments
                       that the LLM will summarise.
    reference_summary: Human-written ground-truth PR description used for
                       evaluation.
    """

    rank: int
    dataset_index: int
    identifier: str
    article: str
    reference_summary: str


class CommitDatasetLoader:
    """Load commit records from the curated dataset.

    Supports both JSONL and CSV formats.  Each row is normalised into a
    :class:`SummarySample` so downstream code is format-agnostic.
    """

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

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

        samples: List[SummarySample] = [
            SummarySample(
                rank=int(row.get("rank", 0)),
                dataset_index=int(row.get("dataset_index", -1)),
                identifier=str(row.get("id", "")),
                article=str(row.get("article", "")),
                reference_summary=str(row.get("abstract", "")),
            )
            for row in rows
        ]
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


class SystemPromptLoader:
    """Read the system prompt from ``<repo root>/resources/prompts.json``."""

    def __init__(self, prompt_path: Path) -> None:
        self.prompt_path = prompt_path

    def load(self) -> str:
        """Read and return the system prompt string.

        Raises
        ------
        FileNotFoundError:
            If the JSON file does not exist at :attr:`prompt_path`.
        ValueError:
            If the ``"system_prompt"`` field is missing or blank.
        """
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"System prompt file missing: {self.prompt_path}")
        with self.prompt_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        prompt = payload.get("pr_llm", {}).get("system_prompt", "").strip()
        if not prompt:
            raise ValueError("system_prompt is empty in the provided prompt file")
        return prompt


class OpenRouterLLMClient:
    """Lightweight client for OpenRouter-powered chat completions.

    Wraps the OpenAI SDK, pointing it at the OpenRouter base URL so that any
    model available on OpenRouter can be addressed with a single client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is required but was not found.")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

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
    ) -> str:
        """Send a chat-completion request and return the generated text."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Model '{model}' returned null content in the response.")
        return content.strip()


class PullRequestSummarizer:
    """Generate PR-description summaries for each sample using multiple LLMs."""

    def __init__(
        self,
        client: OpenRouterLLMClient,
        system_prompt: str,
        model_map: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
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
        return (
            f"Rank: {sample.rank}\n"
            f"Dataset Index: {sample.dataset_index}\n"
            f"Identifier: {sample.identifier}\n"
            "Commit messages and code comments:\n"
            f"{sample.article}\n\n"
            "Produce a concise pull request description based only on the content above."
        )

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
            print(f"[{idx}/{total}] id={sample.identifier}  rank={sample.rank}", flush=True)
            try:
                generated = self.summarizer.summarize(sample)
                for label, summary_text in generated.items():
                    self.writers[label].append({
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
        Default: ``<PR_LLM>/input/dataset/selected_commits.jsonl``.
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

    pr_llm_root = Path(__file__).resolve().parent.parent  # PR_LLM/
    repo_root = pr_llm_root.parent
    dataset = dataset_path or (pr_llm_root / "input" / "dataset" / "selected_commits.jsonl")
    base_output_dir = output_path or (pr_llm_root / "output")
    prompt_file = prompt_path or (repo_root / "resources" / "prompts.json")

    prompt_text = SystemPromptLoader(prompt_file).load()
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
        client, prompt_text, model_map=filtered_map,
        temperature=temperature, max_tokens=max_tokens,
    )
    writers = {
        label: ResultWriter(base_output_dir / f"{label}.jsonl")
        for label in summarizer.model_map.keys()
    }
    loader = CommitDatasetLoader(dataset)
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
