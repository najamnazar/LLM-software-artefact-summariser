#!/usr/bin/env python3
"""rank_sumslice_summaries.py

Rubric-based LLM ranking of SUMSLICE method summaries across 6 quality criteria.

Evaluates CLAUDE, GPT, MISTRAL, and QWEN method summaries against human
ground-truth summaries on 6 criteria:

  1. Accuracy    - Is the summary an accurate description of the method?
  2. Content     - Is the summary missing important information that would
                   hinder understanding of the method?
  3. Conciseness - Does the summary contain unnecessary information?
  4. What        - Does the summary help you understand what the method does
                   (e.g. the internals of the method)?
  5. Why         - Does the summary help you understand why the method exists
                   in the project (e.g. the consequences of altering or
                   removing it)?
  6. How         - Does the summary help you understand how to use the method?

Approach (mirrors rank_pr_summaries.py in PR_LLM):
  - Summaries are matched to ground truth by (project, methodName, className).
  - For each matched entry, all 4 model summaries are ranked 1-4 per criterion
    against the gold summary via the LLM judge.
  - Points awarded: 1st=4, 2nd=3, 3rd=2, 4th=1.
  - Per-model per-criterion and overall averages are computed and written to
    evaluation_results/rubric_results.json.

Run this script after the SUMSLICE_*_SUMMARY.json files have been generated.

Usage:
    python rank_sumslice_summaries.py [--limit N]

    --limit N   Process only the first N matched entries (useful for testing).
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
SUMSLICE_ROOT = Path(__file__).resolve().parent.parent  # SUMSLICE_LLM/
OUTPUT_DIR = SUMSLICE_ROOT / "output"
GT_DIR = SUMSLICE_ROOT / "input" / "ground-truth"
RESULTS_DIR = SUMSLICE_ROOT / "evaluation_results"
RESULTS_JSON = RESULTS_DIR / "rubric_results.json"
CHECKPOINT_FILE = OUTPUT_DIR / "sumslice_rubric_eval_checkpoint.json"
PROMPTS_JSON = SUMSLICE_ROOT / "resources" / "prompts.json"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_FILES: dict[str, str] = {
    "CLAUDE": "SUMSLICE_CLAUDE_SUMMARY.json",
    "GPT": "SUMSLICE_GPT_SUMMARY.json",
    "MISTRAL": "SUMSLICE_MISTRAL_SUMMARY.json",
    "QWEN": "SUMSLICE_QWEN_SUMMARY.json",
}
MODEL_LABELS: dict[str, str] = {
    "CLAUDE": "Claude",
    "GPT": "GPT",
    "MISTRAL": "Mistral",
    "QWEN": "Qwen",
}

# Ground-truth file names keyed by the project name used in LLM output.
PROJECT_GT_FILES: dict[str, str] = {
    "jajuk": "jajuk-example-summaries.json",
    "jEdit": "jedit-example-summaries.json",
    "jhotdraw": "jhotdraw-example-summaries.json",
    "jtopas": "jtopas-example-summaries.json",
    "nanoxml": "nanoXML-example-summaries.json",
    "siena-master": "siena-example-summaries.json",
}

# Rank -> points mapping (4 models)
POINTS_MAP: dict[int, int] = {1: 4, 2: 3, 3: 2, 4: 1}

MethodKey = tuple[str, str, str]  # (project, methodName, className)


# ---------------------------------------------------------------------------
# Environment / API helpers
# ---------------------------------------------------------------------------

def load_env() -> tuple[str, str, str, int, float]:
    """Load and validate .env configuration."""
    load_dotenv(SUMSLICE_ROOT / ".env")
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

def load_model_summaries(json_path: Path) -> dict[MethodKey, list[str]]:
    """Load a SUMSLICE_*_SUMMARY.json file into a key -> [summary, ...] map.

    Some (project, name, class) keys correspond to overloaded methods that
    appear more than once in the corpus; all occurrences are kept, in file
    order, so they can be paired positionally with the matching ground-truth
    occurrences (see load_ground_truth).
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    lookup: dict[MethodKey, list[str]] = {}
    for entry in data["summaries"]:
        key = (entry["project"], entry["name"], entry["class"])
        lookup.setdefault(key, []).append(entry["summary"])
    return lookup


def load_ground_truth(gt_dir: Path) -> dict[MethodKey, list[str]]:
    """Build a (project, methodName, className) -> [summary, ...] lookup from all GT files.

    Keys with more than one occurrence are overloaded methods (same name and
    class, different signatures); all occurrences are kept in file order.
    """
    lookup: dict[MethodKey, list[str]] = {}
    for llm_project, gt_filename in PROJECT_GT_FILES.items():
        gt_path = gt_dir / gt_filename
        if not gt_path.exists():
            print(f"  WARNING: Ground-truth file not found: {gt_path}")
            continue
        with open(gt_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data["summaries"]:
            key = (llm_project, entry["methodName"], entry["className"])
            lookup.setdefault(key, []).append(entry["summary"])
    return lookup


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_ranking_prompts() -> dict[str, str]:
    """Load sumslice-llm.summary_ranking prompt templates from resources/prompts.json."""
    if not PROMPTS_JSON.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_JSON}")
    with open(PROMPTS_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ranking_prompts = data.get("sumslice-llm", {}).get("summary_ranking")
    if not isinstance(ranking_prompts, dict):
        raise ValueError("prompts.json is missing the sumslice-llm.summary_ranking section")
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
    on_result: Optional[callable] = None,
) -> dict[str, Optional[dict[str, int]]]:
    """
    Evaluate one method entry across the given *prompts* (a subset of criteria
    is fine — callers pass only the criteria still missing after a checkpoint
    resume).

    Returns a dict mapping criterion_name -> {model_key: rank, ...} or None on
    failure. If *on_result* is given, it is called as on_result(criterion_name,
    ranking_or_none) immediately after each criterion so callers can persist
    progress incrementally (e.g. surviving a mid-entry network drop).
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
            if on_result:
                on_result(criterion_name, None)
            continue

        ranks = parse_ranking(content, n)
        if ranks is None:
            criterion_results[criterion_name] = None
            print(f"        [{criterion_name}] parse failed: {content!r}")
            if on_result:
                on_result(criterion_name, None)
            continue

        ranking_dict = {model_keys[i]: ranks[i] for i in range(n)}
        criterion_results[criterion_name] = ranking_dict

        rank_str = "  ".join(
            f"{MODEL_LABELS.get(k, k)}={ranks[i]}" for i, k in enumerate(model_keys)
        )
        print(f"        [{criterion_name}] {rank_str}")
        if on_result:
            on_result(criterion_name, ranking_dict)

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
    parser = argparse.ArgumentParser(description="Rubric ranking for SUMSLICE method summaries")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N ranking entries (useful for testing)",
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

    # --- Load ground truth ---
    print("\nLoading ground-truth summaries...")
    gt_lookup = load_ground_truth(GT_DIR)
    print(f"  Ground-truth entries: {len(gt_lookup)}")

    # --- Load model summary files ---
    print("\nLoading model summary files...")
    model_data: dict[str, dict[MethodKey, list[str]]] = {}
    for short_name, filename in MODEL_FILES.items():
        path = OUTPUT_DIR / filename
        if not path.exists():
            print(f"  WARNING: {path} not found – skipping {short_name}")
            continue
        model_data[short_name] = load_model_summaries(path)
        print(f"  {short_name}: {len(model_data[short_name])} unique method keys")

    available_models = [m for m in MODEL_FILES if m in model_data]
    if len(available_models) < 2:
        print("Need at least 2 model files to compare. Exiting.")
        sys.exit(1)

    # --- Find common method keys across all models and ground truth ---
    common_keys: list[MethodKey] = sorted(
        set.intersection(*[set(model_data[m].keys()) for m in available_models])
        & set(gt_lookup.keys())
    )

    # Some (project, name, class) keys are shared by overloaded methods and
    # occur more than once in the corpus (same key, different summaries).
    # Expand each key into one ranking entry per occurrence, pairing the i-th
    # ground-truth occurrence with the i-th occurrence from every model. This
    # is only valid because the occurrence counts match exactly across GT and
    # all 4 models for every duplicated key (verified against the corpus).
    expanded_entries: list[dict] = []
    for key in common_keys:
        project, method_name, class_name = key
        n_occurrences = min(
            len(gt_lookup[key]),
            min(len(model_data[m][key]) for m in available_models),
        )
        for occ_idx in range(n_occurrences):
            expanded_entries.append({
                "key": key,
                "occurrence_index": occ_idx,
                "reference": gt_lookup[key][occ_idx],
                "summaries_ordered": [
                    model_data[m][key][occ_idx] for m in available_models
                ],
            })

    print(
        f"\nCommon method identities across {len(available_models)} models and "
        f"ground truth: {len(common_keys)}"
    )
    print(f"Total ranking entries (including overloaded-method duplicates): "
          f"{len(expanded_entries)}")

    if args.limit:
        expanded_entries = expanded_entries[: args.limit]
        print(f"(Limited to first {args.limit} entries)")

    # --- Load checkpoint ---
    checkpoint: dict[str, dict] = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        print(f"Loaded checkpoint: {len(checkpoint)} completed entries")

    def save_checkpoint() -> None:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, indent=2)

    # --- Evaluate ---
    evaluated_entries: list[dict] = []
    from_checkpoint = 0
    resumed = 0

    for seq, item in enumerate(expanded_entries, start=1):
        project, method_name, class_name = item["key"]
        occ_idx = item["occurrence_index"]
        checkpoint_key = "::".join(item["key"]) + f"::{occ_idx}"

        # Criteria already scored successfully in a prior run (survives a
        # partial/interrupted run — only the missing/failed criteria below
        # get re-queried, not the whole entry).
        existing_criteria: dict[str, Optional[dict[str, int]]] = checkpoint.get(
            checkpoint_key, {}
        ).get("criteria", {})
        missing_prompts = {
            c: tmpl for c, tmpl in ranking_prompts.items()
            if not existing_criteria.get(c)
        }

        if not missing_prompts:
            evaluated_entries.append(checkpoint[checkpoint_key])
            from_checkpoint += 1
            continue

        if existing_criteria:
            resumed += 1

        reference = item["reference"]
        summaries_ordered = item["summaries_ordered"]

        merged_criteria: dict[str, Optional[dict[str, int]]] = dict(existing_criteria)

        def persist(criterion_name: str, result: Optional[dict[str, int]]) -> None:
            merged_criteria[criterion_name] = result
            checkpoint[checkpoint_key] = {
                "project": project,
                "class_name": class_name,
                "method_name": method_name,
                "occurrence_index": occ_idx,
                "criteria": merged_criteria,
            }
            save_checkpoint()

        print(f"\n[{seq}/{len(expanded_entries)}] {project}/{class_name}.{method_name}"
              f"{f' (overload #{occ_idx + 1})' if occ_idx else ''}"
              f"{f'  [resuming {len(missing_prompts)} missing criteria]' if existing_criteria else ''}")

        evaluate_entry(
            api_key,
            api_url,
            judge_model,
            max_tokens,
            temperature,
            reference,
            summaries_ordered,
            available_models,
            missing_prompts,
            on_result=persist,
        )

        entry_result = checkpoint[checkpoint_key]
        evaluated_entries.append(entry_result)

    print(
        f"\nEvaluation complete: {len(evaluated_entries)} entries "
        f"({from_checkpoint} fully from checkpoint, "
        f"{resumed} resumed after partial failure, "
        f"{len(evaluated_entries) - from_checkpoint - resumed} newly evaluated)"
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
            print(f"  {c:<12}: avg_points={s[f'{c}_avg_points']}")

    # --- Write rubric_results.json ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nUpdating {RESULTS_JSON} ...")
    results: dict = {}
    if RESULTS_JSON.exists():
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

    print("Done. Results written to rubric_results.json")


if __name__ == "__main__":
    main()
