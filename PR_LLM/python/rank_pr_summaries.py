#!/usr/bin/env python3
"""rank_pr_summaries.py

Rubric-based LLM ranking of PR summaries across 5 quality criteria.

Ranks the deterministic tool baseline and every LLM against the human gold
description on 5 criteria:

  1. Accuracy          - Does the description correctly represent the changes?
  2. Adequacy          - Does the description cover the main aspects of the change?
  3. Conciseness       - Is the description brief while still conveying essential information?
  4. Context Awareness - Does the description reflect relevant PR context (commits, rationale)?
  5. Clarity           - Is the description easy for reviewers to understand?

Approach (mirrors DPS_LLM's rank_summaries.py):
  - The deterministic tool holds the first, fixed slot — the role NLG and SWUM
    play in DPS_LLM — and the LLMs follow in a stable order.
  - For each PR item every system's summary is ranked 1..n per criterion
    against the human gold summary via the LLM judge.
  - Points awarded: 1st = n points ... last = 1 point, so the scale adapts to
    however many systems are present.
  - Per-system per-criterion and overall averages are merged into results.json.

Run after evaluate_pr_summaries.py has produced results.json.

Usage:
    python rank_pr_summaries.py [SYSTEM ...] [--limit N] [--fresh]

    SYSTEM      Subset to rank: TOOL, GPT, QWEN, CLAUDE, MISTRAL, or ALL.
    --limit N   Process only the first N items (useful for testing).
    --fresh     Ignore any existing checkpoint and re-rank from scratch.

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

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

import pr_corpus
from pr_corpus import (
    DISPLAY_NAMES,
    OUTPUT_DIR,
    RESULTS_DIR,
    REPO_ROOT,
    system_order,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_JSON = OUTPUT_DIR / "results.json"
CHECKPOINT_FILE = OUTPUT_DIR / "rubric_eval_checkpoint.json"

# Repair runs re-issue only the criterion calls that failed on an earlier pass.
# Those failures are overwhelmingly truncations — the judge opened with an
# analysis and ran out of budget before emitting its ranking — so retrying at
# the same limit and temperature would reproduce the same truncated reply and
# waste the call. The retry therefore gets a larger budget; paired with
# parse_ranking_tail, the longer reply is read from its conclusion rather than
# from its opening prose.
REPAIR_TOKEN_MULTIPLIER = 6


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

    # Ensure enough tokens that the n-item ranking output is never truncated.
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
# Ranking output parser
# ---------------------------------------------------------------------------

def parse_ranking(content: str, n: int) -> Optional[List[int]]:
    """Parse LLM output into a list of integer ranks of length *n*.

    Returns a list where ``result[i]`` is the rank assigned to summary ``i+1``,
    or ``None`` if the output cannot be reliably parsed.
    """
    # Match multi-digit runs so a 10+ system comparison still parses, then keep
    # only values inside the valid rank range.
    digits = re.findall(r"\d+", content)
    valid = [int(d) for d in digits if 1 <= int(d) <= n]

    seen: set[int] = set()
    deduped: List[int] = []
    for value in valid:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
        if len(deduped) == n:
            break

    if len(deduped) != n or set(deduped) != set(range(1, n + 1)):
        return None
    return deduped


def parse_ranking_tail(content: str, n: int) -> Optional[List[int]]:
    """Parse a ranking from the END of a verbose reply.

    ``parse_ranking`` scans forward and keeps the first *n* in-range integers it
    meets. That is correct for a bare ``"3, 1, 5, 2, 4"`` answer, but wrong when
    the judge writes an analysis first: its prose numbering ("1. The first
    generated summary ...") gets consumed as if it were the ranking, which would
    turn an honest parse failure into a silently incorrect one.

    This variant walks backwards over the in-range integers and returns the last
    window of *n* that forms a permutation of 1..n — the ranking the model
    settled on after its reasoning, rather than the numbering it opened with.

    Used only on the repair path (see ``main``), so the primary ranking pass
    keeps its original, stricter parsing behaviour untouched.
    """
    valid = [int(d) for d in re.findall(r"\d+", content) if 1 <= int(d) <= n]
    for start in range(len(valid) - n, -1, -1):
        window = valid[start:start + n]
        if set(window) == set(range(1, n + 1)):
            return window
    return None


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------

def build_summaries_block(summaries_ordered: List[str]) -> str:
    """Render the numbered 'Generated summaries' block for the judge prompt."""
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(summaries_ordered))


def evaluate_item(
    api_key: str,
    api_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    human_summary: str,
    summaries_ordered: List[str],
    systems: List[str],
    prompts: Dict[str, str],
    parse: Callable[[str, int], Optional[List[int]]] = parse_ranking,
) -> Dict[str, Optional[Dict[str, int]]]:
    """Evaluate one PR item across the criteria in *prompts*.

    Returns a dict mapping criterion -> {system: rank} or ``None`` on failure.

    *prompts* may hold a subset of the criteria, which is how the repair path in
    ``main`` re-runs only the criteria a cached item is missing. *parse* selects
    the reply parser; the repair path overrides it with ``parse_ranking_tail``.
    """
    n = len(summaries_ordered)
    summaries_block = build_summaries_block(summaries_ordered)
    criterion_results: Dict[str, Optional[Dict[str, int]]] = {}

    for criterion_name, template in prompts.items():
        prompt = template.format(
            human_summary=human_summary,
            summaries_block=summaries_block,
            n=n,
        )
        content = call_api(api_key, api_url, model, prompt, max_tokens, temperature)

        if content is None:
            criterion_results[criterion_name] = None
            print(f"        [{criterion_name}] skipped (API error)")
            continue

        ranks = parse(content, n)
        if ranks is None:
            criterion_results[criterion_name] = None
            print(f"        [{criterion_name}] parse failed: {content!r}")
            continue

        criterion_results[criterion_name] = {
            systems[i]: ranks[i] for i in range(n)
        }

        rank_str = "  ".join(
            f"{DISPLAY_NAMES.get(s, s)}={ranks[i]}" for i, s in enumerate(systems)
        )
        print(f"        [{criterion_name}] {rank_str}")

    return criterion_results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def points_for_rank(rank: int, n_systems: int) -> int:
    """Convert a 1-based rank into points: 1st = n points ... last = 1 point."""
    return (n_systems + 1) - rank


def aggregate_scores(
    all_item_results: List[dict],
    systems: List[str],
    criteria: List[str],
) -> Dict[str, dict]:
    """Aggregate per-item rankings into per-system summary statistics."""
    n_systems = len(systems)
    collected: Dict[str, Dict[str, List[int]]] = {
        s: {c: [] for c in criteria} for s in systems
    }

    for item in all_item_results:
        for criterion_name, rankings in item["criteria"].items():
            if not rankings:
                continue
            for system, rank in rankings.items():
                if system in collected and criterion_name in collected[system]:
                    collected[system][criterion_name].append(
                        points_for_rank(rank, n_systems)
                    )

    result: Dict[str, dict] = {}
    for system in systems:
        stats: dict = {}
        all_points: List[int] = []

        for criterion in criteria:
            points = collected[system][criterion]
            if points:
                stats[f"{criterion}_avg_points"] = round(sum(points) / len(points), 4)
                stats[f"{criterion}_entries"] = len(points)
                all_points.extend(points)
            else:
                stats[f"{criterion}_avg_points"] = None
                stats[f"{criterion}_entries"] = 0

        stats["total_rubric_points"] = sum(all_points)
        stats["overall_avg_points"] = (
            round(sum(all_points) / len(all_points), 4) if all_points else None
        )
        stats["max_points_per_item"] = n_systems
        result[system] = stats

    return result


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(systems: List[str], fresh: bool) -> Dict[str, dict]:
    """Load cached per-item rankings, discarding any built for a different set.

    The checkpoint records which systems produced it. If the comparison set has
    changed — a model added, the corpus swapped — the cached ranks no longer
    describe the same slots, so they are dropped rather than silently reused.
    """
    if fresh or not CHECKPOINT_FILE.exists():
        return {}

    try:
        with CHECKPOINT_FILE.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        print("  WARNING: checkpoint unreadable — starting fresh")
        return {}

    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict) or meta.get("systems") != systems:
        print("  Checkpoint was built for a different system set — starting fresh")
        return {}

    entries = payload.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def save_checkpoint(entries: Dict[str, dict], systems: List[str], judge_model: str) -> None:
    """Persist per-item rankings together with the system set that produced them."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "systems": systems,
            "judge_model": judge_model,
            "corpus": str(pr_corpus.INPUT_DIR),
        },
        "entries": entries,
    }
    with CHECKPOINT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    valid_systems = system_order()

    parser = argparse.ArgumentParser(description="Rubric ranking for PR summaries")
    parser.add_argument(
        "systems",
        metavar="SYSTEM",
        nargs="*",
        default=["ALL"],
        help=f"Systems to rank: {', '.join(valid_systems)}, or ALL. Default: ALL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N items (useful for testing)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint and re-rank from scratch",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Accept checkpointed items exactly as cached, without re-running "
             "criteria whose judge reply failed to parse on an earlier run",
    )
    args = parser.parse_args()

    # --- Load API config ---
    api_key, api_url, judge_model, max_tokens, temperature = load_env()
    print(f"Judge model : {judge_model}")
    print(f"Max tokens  : {max_tokens}")
    print(f"Temperature : {temperature}")

    # --- Load ranking prompts ---
    ranking_prompts = pr_corpus.load_ranking_prompts()
    criteria = list(ranking_prompts.keys())
    print(f"Criteria    : {', '.join(criteria)}")

    # --- Resolve the comparison set ---
    requested = [s.upper() for s in args.systems]
    if requested == ["ALL"]:
        wanted = valid_systems
    else:
        unknown = [s for s in requested if s not in valid_systems]
        if unknown:
            print(f"Unknown system(s): {unknown}. Valid: {valid_systems}")
            sys.exit(1)
        # Preserve canonical order so the tool keeps its fixed leading slot.
        wanted = [s for s in valid_systems if s in requested]

    print("\nLoading summaries...")
    summary_maps: Dict[str, Dict[str, str]] = {}
    for system in wanted:
        try:
            summaries = pr_corpus.load_system_summaries(system)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} – skipping {system}")
            continue
        summary_maps[system] = summaries
        print(f"  {DISPLAY_NAMES.get(system, system)}: {len(summaries)} entries")

    systems = [s for s in wanted if s in summary_maps]
    if len(systems) < 2:
        print("Need at least 2 systems to compare. Exiting.")
        sys.exit(1)

    gold = pr_corpus.load_human_summaries()
    print(f"  Human gold: {len(gold)} entries")

    # --- Items every system covers ---
    common_ids = pr_corpus.common_pr_ids(gold, *(summary_maps[s] for s in systems))
    print(f"\nCommon items across {len(systems)} systems: {len(common_ids)}")
    if not common_ids:
        print("No overlapping pr_ids between the gold summaries and the systems. Exiting.")
        sys.exit(1)

    if args.limit:
        common_ids = common_ids[: args.limit]
        print(f"(Limited to first {args.limit} items)")

    # --- Load checkpoint ---
    checkpoint = load_checkpoint(systems, args.fresh)
    if checkpoint:
        print(f"Loaded checkpoint: {len(checkpoint)} completed items")

    # --- Evaluate ---
    evaluated_items: List[dict] = []
    from_checkpoint = 0
    repaired_items = 0
    repaired_cells = 0

    for seq, pr_id in enumerate(common_ids, start=1):
        if pr_id in checkpoint:
            # A cached item can still carry gaps: individual criteria whose
            # judge reply failed to parse — typically an analysis preamble that
            # ran past max_tokens, so the ranking line was never emitted. The
            # item is cached because its OTHER criteria succeeded, which used to
            # make those gaps permanent: a cached item was accepted whole, and
            # the only way back was --fresh, re-ranking all 150 items.
            #
            # Re-issue just the missing criterion calls (7 calls, not 750). The
            # prompt is byte-identical, so a repaired cell is scored under the
            # same rubric as every other cell; only the token budget and the
            # reply parser are relaxed, since a truncated reply is not a
            # judgement the run can keep either way.
            #
            # Old behaviour (cached item always accepted as-is):
            # evaluated_items.append(checkpoint[pr_id])
            # from_checkpoint += 1
            # continue
            cached = checkpoint[pr_id]
            cached_criteria = cached.get("criteria") or {}
            missing = [c for c in criteria if not cached_criteria.get(c)]

            if not missing or args.no_repair:
                evaluated_items.append(cached)
                from_checkpoint += 1
                continue

            print(f"\n[{seq}/{len(common_ids)}] pr_id={pr_id} "
                  f"— repairing {len(missing)}: {', '.join(missing)}")

            retried = evaluate_item(
                api_key,
                api_url,
                judge_model,
                max_tokens * REPAIR_TOKEN_MULTIPLIER,
                temperature,
                gold[pr_id],
                [summary_maps[s][pr_id] for s in systems],
                systems,
                {c: ranking_prompts[c] for c in missing},
                parse=parse_ranking_tail,
            )

            recovered = {c: r for c, r in retried.items() if r}
            cached_criteria.update(retried)
            cached["criteria"] = cached_criteria
            evaluated_items.append(cached)
            from_checkpoint += 1

            print(f"        [repair] recovered {len(recovered)}/{len(missing)}")
            if recovered:
                repaired_items += 1
                repaired_cells += len(recovered)
                checkpoint[pr_id] = cached
                save_checkpoint(checkpoint, systems, judge_model)
            continue

        human_summary = gold[pr_id]
        summaries_ordered = [summary_maps[s][pr_id] for s in systems]

        print(f"\n[{seq}/{len(common_ids)}] pr_id={pr_id}")

        criteria_results = evaluate_item(
            api_key,
            api_url,
            judge_model,
            max_tokens,
            temperature,
            human_summary,
            summaries_ordered,
            systems,
            ranking_prompts,
        )

        item_result = {
            "pr_id": pr_id,
            "criteria": criteria_results,
        }
        evaluated_items.append(item_result)

        # Persist checkpoint after each item so a long run is resumable — but
        # only when at least one criterion actually produced a ranking. An item
        # whose criteria all came back None (API outage, unparseable reply) was
        # previously cached as "done" and then skipped by every later run, so
        # the only way to retry it was --fresh, which re-ranks the whole corpus.
        #
        # Old behaviour (checkpointed unconditionally):
        # checkpoint[pr_id] = item_result
        # save_checkpoint(checkpoint, systems, judge_model)
        if any(criteria_results.values()):
            checkpoint[pr_id] = item_result
            save_checkpoint(checkpoint, systems, judge_model)
        else:
            print("        [checkpoint] not cached — every criterion failed; "
                  "this item will be retried on the next run")

    print(
        f"\nEvaluation complete: {len(evaluated_items)} items "
        f"({from_checkpoint} from checkpoint, "
        f"{len(evaluated_items) - from_checkpoint} newly evaluated)"
    )
    if repaired_cells:
        print(f"  Repaired {repaired_cells} previously-unparsed criterion "
              f"ranking(s) across {repaired_items} item(s)")

    # --- Aggregate ---
    print("\nAggregating scores...")
    rubric_scores = aggregate_scores(evaluated_items, systems, criteria)

    print("\n=== Rubric Ranking Results ===")
    for system in systems:
        stats = rubric_scores[system]
        print(f"\n{DISPLAY_NAMES.get(system, system)}:")
        print(f"  Overall avg points : {stats['overall_avg_points']} / {len(systems)}.0")
        print(f"  Total rubric points: {stats['total_rubric_points']}")
        for criterion in criteria:
            print(f"  {criterion:<20}: avg_points={stats[f'{criterion}_avg_points']}")

    # --- Write per-item ranking detail alongside the metric scores ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_rows: List[dict] = []
    for item in evaluated_items:
        row: dict = {"pr_id": item["pr_id"]}
        for criterion in criteria:
            ranks = item["criteria"].get(criterion) or {}
            for system in systems:
                row[f"{criterion}_{system}_rank"] = ranks.get(system)
        detail_rows.append(row)
    detail_path = RESULTS_DIR / "ranking_detail.jsonl"
    write_jsonl(detail_path, detail_rows)

    summary_path = RESULTS_DIR / "ranking_summary.jsonl"
    write_jsonl(
        summary_path,
        [{"system": s, **rubric_scores[s]} for s in systems],
    )
    print(f"\nRanking detail : {detail_path}")
    print(f"Ranking summary: {summary_path}")

    # --- Update results.json ---
    print(f"\nUpdating {RESULTS_JSON} ...")
    results: dict = {}
    if RESULTS_JSON.exists():
        try:
            with RESULTS_JSON.open(encoding="utf-8") as fh:
                results = json.load(fh)
        except Exception:
            print("  WARNING: results.json unreadable — writing a fresh file")

    for system in systems:
        results.setdefault(system, {})["rubric_evaluation"] = rubric_scores[system]

    results["rubric_evaluation_meta"] = {
        "judge_model": judge_model,
        "criteria": criteria,
        "criteria_prompts": ranking_prompts,
        "total_items_evaluated": len(evaluated_items),
        "systems_compared": systems,
        "ranking_system": f"1st={len(systems)}pts ... {len(systems)}th=1pt",
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("Done. Results written to results.json")


if __name__ == "__main__":
    main()
