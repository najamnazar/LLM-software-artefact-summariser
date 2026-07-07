#!/usr/bin/env python3
"""rank_pr_summaries.py

Rubric-based LLM ranking of PR summaries across 5 quality criteria.

Evaluates GPT, QWEN, CLAUDE, and MISTRAL pull request summaries against
reference summaries on 5 criteria:

  1. Accuracy          - Does the description correctly represent the changes?
  2. Adequacy          - Does the description cover the main aspects of the change?
  3. Conciseness       - Is the description brief while still conveying essential information?
  4. Context Awareness - Does the description reflect relevant PR context (commits, rationale)?
  5. Clarity           - Is the description easy for reviewers to understand?

Approach (mirrors rank_summaries.py for DPS):
  - For each PR entry all 4 model summaries are ranked 1-4 per criterion against
    the reference summary via the LLM judge.
  - Points awarded: 1st=4, 2nd=3, 3rd=2, 4th=1.
  - Per-model per-criterion and overall averages computed and merged into results.json.

Run this script after evaluate_pr_summaries.py has produced results.json.

Usage:
    python rank_pr_summaries.py [--limit N]

    --limit N   Process only the first N entries (useful for testing).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PR_LLM_ROOT = Path(__file__).resolve().parent.parent  # PR_LLM/
REPO_ROOT = PR_LLM_ROOT.parent
OUTPUT_DIR = PR_LLM_ROOT / "output"
RESULTS_JSON = OUTPUT_DIR / "results.json"
CHECKPOINT_FILE = OUTPUT_DIR / "rubric_eval_checkpoint.json"
PROMPTS_JSON = REPO_ROOT / "resources" / "prompts.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = [
    "PR_GPT_SUMMARY",
    "PR_QWEN_SUMMARY",
    "PR_CLAUDE_SUMMARY",
    "PR_MISTRAL_SUMMARY",
]
MODEL_LABELS = {
    "PR_GPT_SUMMARY": "GPT",
    "PR_QWEN_SUMMARY": "QWEN",
    "PR_CLAUDE_SUMMARY": "CLAUDE",
    "PR_MISTRAL_SUMMARY": "MISTRAL",
}

# ---------------------------------------------------------------------------
# Evaluation criteria
# ---------------------------------------------------------------------------
# Criteria definitions moved to <repo root>/resources/prompts.json under "pr_llm.pr_ranking".
# Loaded at runtime via load_ranking_prompts().
# CRITERIA: dict[str, str] = {
#     "accuracy": (
#         "Accuracy: Does the description correctly represent the changes in the pull request?"
#     ),
#     "adequacy": (
#         "Adequacy: Does the description cover the main aspects of the change?"
#     ),
#     "conciseness": (
#         "Conciseness: Is the description brief while still conveying the essential information?"
#     ),
#     "context_awareness": (
#         "Context Awareness: Does the description reflect relevant information from the PR "
#         "context (e.g., commits, rationale)?"
#     ),
#     "clarity": (
#         "Clarity: Is the description easy for reviewers to understand?"
#     ),
# }

# Rank -> points mapping (4 models)
POINTS_MAP: dict[int, int] = {1: 4, 2: 3, 3: 2, 4: 1}


# ---------------------------------------------------------------------------
# Environment / API helpers
# ---------------------------------------------------------------------------

def load_env() -> tuple[str, str, str, int, float]:
    """Load and validate .env configuration."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    api_url = os.getenv("OPENROUTER_API_URL", "")
    model = os.getenv("RANK_SUMMARIES_MODEL", "")
    max_tokens_raw = os.getenv("RANK_SUMMARIES_MAX_TOKENS", "100")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    if not api_url:
        raise ValueError("OPENROUTER_API_URL not found in .env")
    if not model:
        raise ValueError("RANK_SUMMARIES_MODEL not found in .env")

    try:
        max_tokens = int(max_tokens_raw)
    except ValueError:
        max_tokens = 100

    # Ensure at least 100 tokens so the 4-item ranking output is never truncated.
    max_tokens = max(max_tokens, 100)
    temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.0"))
    return api_key, api_url, model, max_tokens, temperature


def call_api(
    api_key: str,
    api_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    retries: int = 3,
) -> Optional[str]:
    """POST to OpenRouter-compatible API and return the text reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries):
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"      [API error] {exc}")
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> dict[int, dict]:
    """Load a JSONL file and index entries by dataset_index."""
    entries: dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["dataset_index"]] = entry
    return entries


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Replaced by load_ranking_prompts() + template.format() in evaluate_entry.
# def build_ranking_prompt(
#     criterion_desc: str,
#     reference: str,
#     summaries: list[str],
# ) -> str:
#     """Build a prompt asking the LLM to rank *len(summaries)* summaries."""
#     n = len(summaries)
#     summaries_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(summaries))
#     return (
#         f"{criterion_desc}\n\n"
#         f"Reference description:\n{reference}\n\n"
#         f"Generated summaries:\n{summaries_block}\n\n"
#         f"Rank the generated summaries from best (1) to worst ({n}) based on the "
#         f"criterion above. Output only the ranking as {n} space-separated integers, "
#         f"e.g.: 2 1 4 3"
#     )


def load_ranking_prompts() -> dict[str, str]:
    """Load pr_llm.pr_ranking prompt templates from <repo root>/resources/prompts.json."""
    if not PROMPTS_JSON.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_JSON}")
    with open(PROMPTS_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ranking_prompts = data.get("pr_llm", {}).get("pr_ranking")
    if not isinstance(ranking_prompts, dict):
        raise ValueError("prompts.json is missing the pr_llm.pr_ranking section")
    return ranking_prompts


# ---------------------------------------------------------------------------
# Ranking output parser
# ---------------------------------------------------------------------------

def parse_ranking(content: str, n: int) -> Optional[list[int]]:
    """
    Parse LLM output into a list of integer ranks (length *n*).

    Returns a list where result[i] is the rank assigned to summary i+1,
    or None if the output cannot be reliably parsed.
    """
    digits = re.findall(r"[1-9]", content)
    # Keep only values in [1, n]
    valid = [int(d) for d in digits if 1 <= int(d) <= n]
    # Deduplicate while preserving order
    seen: set[int] = set()
    deduped: list[int] = []
    for v in valid:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
        if len(deduped) == n:
            break

    if len(deduped) != n or set(deduped) != set(range(1, n + 1)):
        return None
    return deduped


# ---------------------------------------------------------------------------
# Per-entry evaluation
# ---------------------------------------------------------------------------

def evaluate_entry(
    api_key: str,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    reference: str,
    summaries_ordered: list[str],
    model_keys: list[str],
    prompts: dict[str, str],
) -> dict[str, Optional[dict[str, int]]]:
    """
    Evaluate one PR entry across all 5 criteria.

    Returns a dict mapping criterion_name -> {model_key: rank, ...} or None on failure.
    """
    n = len(summaries_ordered)
    criterion_results: dict[str, Optional[dict[str, int]]] = {}

    for criterion_name, template in prompts.items():
        prompt = template.format(
            human_summary=reference,
            **{f"summary_{i + 1}": s for i, s in enumerate(summaries_ordered)},
        )
        content = call_api(api_key, api_url, model, prompt, max_tokens, temperature)

        if content is None:
            criterion_results[criterion_name] = None
            print(f"        [{criterion_name}] skipped (API error)")
            continue

        ranks = parse_ranking(content, n)
        if ranks is None:
            criterion_results[criterion_name] = None
            print(f"        [{criterion_name}] parse failed: {content!r}")
            continue

        ranking_dict = {model_keys[i]: ranks[i] for i in range(n)}
        criterion_results[criterion_name] = ranking_dict

        rank_str = "  ".join(
            f"{MODEL_LABELS.get(k, k)}={ranks[i]}" for i, k in enumerate(model_keys)
        )
        print(f"        [{criterion_name}] {rank_str}")

    return criterion_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(
    all_entry_results: list[dict],
    model_keys: list[str],
    criteria: list[str],
) -> dict[str, dict]:
    """Aggregate per-entry rankings into per-model summary statistics."""
    # model -> criterion -> list of points
    collected: dict[str, dict[str, list[int]]] = {
        m: {c: [] for c in criteria} for m in model_keys
    }

    for entry in all_entry_results:
        for criterion_name, rankings in entry["criteria"].items():
            if rankings is None:
                continue
            for model_key, rank in rankings.items():
                if model_key in collected and criterion_name in collected[model_key]:
                    collected[model_key][criterion_name].append(
                        POINTS_MAP.get(rank, 0)
                    )

    result: dict[str, dict] = {}
    for m in model_keys:
        stats: dict = {}
        all_points: list[int] = []

        for c in criteria:
            pts = collected[m][c]
            if pts:
                avg = round(sum(pts) / len(pts), 4)
                stats[f"{c}_avg_points"] = avg
                stats[f"{c}_entries"] = len(pts)
                all_points.extend(pts)
            else:
                stats[f"{c}_avg_points"] = None
                stats[f"{c}_entries"] = 0

        stats["total_rubric_points"] = sum(all_points)
        stats["overall_avg_points"] = (
            round(sum(all_points) / len(all_points), 4) if all_points else None
        )
        result[m] = stats

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rubric ranking for PR summaries")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N entries (useful for testing)",
    )
    args = parser.parse_args()

    # --- Load API config ---
    api_key, api_url, judge_model, max_tokens, temperature = load_env()
    print(f"Judge model : {judge_model}")
    print(f"Max tokens  : {max_tokens}")
    print(f"Temperature : {temperature}")

    # --- Load ranking prompts ---
    ranking_prompts = load_ranking_prompts()
    criteria = list(ranking_prompts.keys())
    print(f"Criteria    : {', '.join(criteria)}")

    # --- Load JSONL data ---
    print("\nLoading JSONL files...")
    model_data: dict[str, dict[int, dict]] = {}
    for m in MODELS:
        path = OUTPUT_DIR / f"{m}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found – skipping {m}")
            continue
        model_data[m] = load_jsonl(path)
        print(f"  {m}: {len(model_data[m])} entries")

    available_models = [m for m in MODELS if m in model_data]
    if len(available_models) < 2:
        print("Need at least 2 model files to compare. Exiting.")
        sys.exit(1)

    # --- Find common dataset indices ---
    common_indices: list[int] = sorted(
        set.intersection(*[set(model_data[m].keys()) for m in available_models])
    )
    print(f"\nCommon entries across {len(available_models)} models: {len(common_indices)}")

    if args.limit:
        common_indices = common_indices[: args.limit]
        print(f"(Limited to first {args.limit} entries)")

    # --- Load checkpoint ---
    checkpoint: dict[str, dict] = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        print(f"Loaded checkpoint: {len(checkpoint)} completed entries")

    # --- Evaluate ---
    evaluated_entries: list[dict] = []
    from_checkpoint = 0

    for seq, idx in enumerate(common_indices, start=1):
        idx_str = str(idx)
        if idx_str in checkpoint:
            evaluated_entries.append(checkpoint[idx_str])
            from_checkpoint += 1
            continue

        reference = model_data[available_models[0]][idx]["reference_summary"]
        pr_id = model_data[available_models[0]][idx]["id"]
        summaries_ordered = [model_data[m][idx]["summary"] for m in available_models]

        print(f"\n[{seq}/{len(common_indices)}] {pr_id}  (idx={idx})")

        criteria_results = evaluate_entry(
            api_key,
            api_url,
            judge_model,
            max_tokens,
            temperature,
            reference,
            summaries_ordered,
            available_models,
            ranking_prompts,
        )

        entry_result = {
            "dataset_index": idx,
            "id": pr_id,
            "criteria": criteria_results,
        }
        evaluated_entries.append(entry_result)

        # Persist checkpoint after each entry
        checkpoint[idx_str] = entry_result
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, indent=2)

    print(
        f"\nEvaluation complete: {len(evaluated_entries)} entries "
        f"({from_checkpoint} from checkpoint, "
        f"{len(evaluated_entries) - from_checkpoint} newly evaluated)"
    )

    # --- Aggregate ---
    print("\nAggregating scores...")
    rubric_scores = aggregate_scores(evaluated_entries, available_models, criteria)

    print("\n=== Rubric Ranking Results ===")
    for m in available_models:
        label = MODEL_LABELS.get(m, m)
        s = rubric_scores[m]
        print(f"\n{label}:")
        print(f"  Overall avg points : {s['overall_avg_points']} / 4.0")
        print(f"  Total rubric points: {s['total_rubric_points']}")
        for c in criteria:
            print(f"  {c:<20}: avg_points={s[f'{c}_avg_points']}")

    # --- Update results.json ---
    print(f"\nUpdating {RESULTS_JSON} ...")
    with open(RESULTS_JSON, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    for m in available_models:
        if m not in results:
            results[m] = {}
        results[m]["rubric_evaluation"] = rubric_scores[m]

    results["rubric_evaluation_meta"] = {
        "judge_model": judge_model,
        "criteria": criteria,
        "criteria_prompts": ranking_prompts,
        "total_entries_evaluated": len(evaluated_entries),
        "models_compared": available_models,
        "ranking_system": "1st=4pts, 2nd=3pts, 3rd=2pts, 4th=1pt",
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("Done. Results written to results.json")


if __name__ == "__main__":
    main()
