"""pr_corpus.py

Shared corpus loader for the PR_LLM pipeline.

Mirrors DPS_LLM's data model: the human (gold) summaries and the deterministic
tool baseline live in ``PR_LLM/input/``, the LLM summaries live in
``PR_LLM/output/`` as one JSONL file per model, and every system is joined to
the gold on a single canonical key.

Corpus layout
-------------
``input/human_summaries/<pr_id>_reference.txt``
    Human-written PR description (gold standard).
``input/tool_summaries/<pr_id>_decoded.txt``
    Deterministic tool baseline, decoded output.
``output/PR_<MODEL>_SUMMARY.jsonl``
    LLM summaries, one JSON record per line.

The canonical join key is ``pr_id`` — the zero-padded numeric identifier taken
from the input filenames (e.g. ``003485``). It is the same key DPS_LLM builds
with ``build_match_key``: a value every system can be resolved by, independent
of the file format it is stored in.

:author: Najam Nazar
:version: 2.0.0
:date: 2026-09-16
:license: MIT
"""

from __future__ import annotations

__author__ = "Najam Nazar"
__version__ = "2.0.0"
__date__ = "2026-09-16"
__license__ = "MIT"

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PR_LLM_ROOT = Path(__file__).resolve().parent.parent      # PR_LLM/
REPO_ROOT = PR_LLM_ROOT.parent                            # TOSEM/
INPUT_DIR = PR_LLM_ROOT / "input"
HUMAN_DIR = INPUT_DIR / "human_summaries"
TOOL_DIR = INPUT_DIR / "tool_summaries"
DATASET_DIR = INPUT_DIR / "dataset"
OUTPUT_DIR = PR_LLM_ROOT / "output"
RESULTS_DIR = PR_LLM_ROOT / "evaluation-results"
PROMPTS_JSON = REPO_ROOT / "resources" / "prompts.json"

HUMAN_SUFFIX = "_reference.txt"
TOOL_SUFFIX = "_decoded.txt"

# ---------------------------------------------------------------------------
# System labels
# ---------------------------------------------------------------------------
# The deterministic baseline. Kept in its own constant so that adding a second
# tool later only means extending TOOL_LABELS and the ranking prompt.
TOOL_LABEL = "TOOL"
TOOL_LABELS: List[str] = [TOOL_LABEL]

# short name -> JSONL stem in output/
MODEL_ALIASES: Dict[str, str] = {
    "GPT": "PR_GPT_SUMMARY",
    "QWEN": "PR_QWEN_SUMMARY",
    "CLAUDE": "PR_CLAUDE_SUMMARY",
    "MISTRAL": "PR_MISTRAL_SUMMARY",
}

# Display names used in tables, plots and the LLM-judge transcript.
DISPLAY_NAMES: Dict[str, str] = {
    "TOOL": "Tool",
    "GPT": "GPT",
    "QWEN": "Qwen",
    "CLAUDE": "Claude",
    "MISTRAL": "Mistral",
}


def system_order(models: Optional[List[str]] = None) -> List[str]:
    """Return the canonical system order: deterministic tool first, then LLMs.

    Mirrors DPS_LLM's fixed-slot convention (NLG in slot A, SWUM in slot C)
    where the deterministic baselines hold stable positions and the LLMs vary.
    """
    return TOOL_LABELS + list(models or MODEL_ALIASES.keys())


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------
_ID_RE = re.compile(r"(\d+)")


def normalise_pr_id(value: object) -> Optional[str]:
    """Normalise any identifier form to the canonical zero-padded ``pr_id``.

    Accepts the raw id (``'003485'``), an int or numeric string
    (``3485``), or a reference filename (``'003485_reference.txt'``).
    Returns ``None`` when no digits are present.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _ID_RE.search(text)
    if not match:
        return None
    return match.group(1).zfill(6)


def record_pr_id(record: dict) -> Optional[str]:
    """Derive the canonical ``pr_id`` from a JSONL summary record.

    Checks, in order of authority: an explicit ``pr_id``, the ``ref_file``
    the dataset row was built from, then ``dataset_index``. The fallback chain
    lets JSONL files written before the corpus change still be joined.
    """
    for key in ("pr_id", "ref_file", "dataset_index"):
        if key in record:
            pr_id = normalise_pr_id(record[key])
            if pr_id:
                return pr_id
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_text_dir(directory: Path, suffix: str) -> Dict[str, str]:
    """Load every ``*<suffix>`` file in *directory*, keyed by canonical pr_id."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    summaries: Dict[str, str] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        pr_id = normalise_pr_id(path.name)
        if pr_id is None:
            continue
        summaries[pr_id] = path.read_text(encoding="utf-8", errors="replace").strip()

    if not summaries:
        raise FileNotFoundError(f"No '*{suffix}' files found in {directory}")
    return summaries


def load_human_summaries(directory: Optional[Path] = None) -> Dict[str, str]:
    """Load the gold PR descriptions from ``input/human_summaries/``."""
    return _load_text_dir(directory or HUMAN_DIR, HUMAN_SUFFIX)


def load_tool_summaries(directory: Optional[Path] = None) -> Dict[str, str]:
    """Load the deterministic tool baseline from ``input/tool_summaries/``."""
    return _load_text_dir(directory or TOOL_DIR, TOOL_SUFFIX)


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    """Read a JSONL file into a list of records."""
    records: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[dict]) -> None:
    """Write *records* to *path*, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_llm_summaries(
    label: str,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """Load one model's JSONL summaries, keyed by canonical pr_id.

    Parameters
    ----------
    label:      JSONL stem, e.g. ``PR_GPT_SUMMARY``.
    output_dir: Directory holding the JSONL files. Defaults to ``PR_LLM/output``.
    limit:      Read only the first *limit* records.

    Raises
    ------
    FileNotFoundError: If the JSONL file does not exist.
    """
    path = (output_dir or OUTPUT_DIR) / f"{label}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    summaries: Dict[str, str] = {}
    for record in load_jsonl(path, limit):
        pr_id = record_pr_id(record)
        if pr_id is None:
            continue
        summaries[pr_id] = str(record.get("summary", "")).strip()
    return summaries


def load_system_summaries(
    system: str,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """Load summaries for any system — the deterministic tool or an LLM.

    This is the single entry point the evaluation and ranking scripts use, so
    the tool baseline and the LLMs are treated identically downstream.
    """
    if system in TOOL_LABELS:
        return load_tool_summaries()
    if system not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown system '{system}'. Valid: {system_order()}"
        )
    return load_llm_summaries(MODEL_ALIASES[system], output_dir, limit)


def available_systems(
    models: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> List[str]:
    """Return the systems whose summaries are actually present on disk."""
    present: List[str] = []
    for system in system_order(models):
        try:
            load_system_summaries(system, output_dir, limit=1)
        except (FileNotFoundError, ValueError):
            continue
        present.append(system)
    return present


def common_pr_ids(*maps: Dict[str, str]) -> List[str]:
    """Return the sorted pr_ids present in every supplied summary map.

    Only non-empty summaries count, so an item a model failed to generate is
    excluded from the comparison rather than scored as a blank string.
    """
    if not maps:
        return []
    id_sets = [{k for k, v in m.items() if v} for m in maps]
    return sorted(set.intersection(*id_sets))


def load_ranking_prompts() -> Dict[str, str]:
    """Load the ``pr_llm.pr_ranking`` criterion templates from prompts.json."""
    if not PROMPTS_JSON.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_JSON}")
    with PROMPTS_JSON.open(encoding="utf-8") as handle:
        data = json.load(handle)
    prompts = data.get("pr_llm", {}).get("pr_ranking")
    if not isinstance(prompts, dict):
        raise ValueError("prompts.json is missing the pr_llm.pr_ranking section")
    return prompts
