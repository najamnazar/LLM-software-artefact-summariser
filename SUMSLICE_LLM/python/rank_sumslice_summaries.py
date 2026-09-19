#!/usr/bin/env python3
"""rank_sumslice_summaries.py

LLM-judge ranking (Borda points) of the four LLM summaries of each method,
against the SumSlice (tool) summary, on the six criteria in
resources/prompts.json (sumslice_llm.summary_ranking_criteria).

Scoring is the Borda count PR_LLM and DPS_LLM use: with k candidates the best
gets k points and the worst 1, so with the four LLMs here 1st = 4 ... 4th = 1.

Order control differs from the other two projects, because the candidate set
does. PR_LLM and DPS_LLM rank a deterministic tool against the LLMs and pin the
tool to a fixed leading slot (NLG in slot A, SWUM in slot C in DPS_LLM), so
their presentation order is fixed by design. Here the SumSlice tool summary is
the reference, not a candidate, and the four candidates are interchangeable
LLMs, so nothing has to hold a fixed slot and judge position bias is controlled
instead:
* candidates are anonymised letters, shuffled per method and criterion (seeded);
* each method/criterion is judged in --orders presentations (default 2):
  a random order and its reverse; points are averaged over presentations;
* the observed share of first places per slot is written to
  judge_position_bias.csv as a check on the residual bias.

As in PR_LLM the judge reply is parsed strictly, unparsed cells are retried
(with a larger token budget, the usual cause being a truncated preamble), raw
replies are kept, and a cell that failed is re-judged on the next run.

Outputs in SUMSLICE_LLM/evaluation_results/:
  judge_checkpoint.json, judge_rankings_long.csv, judge_position_bias.csv,
  results.json (keys "judge" and "judge_meta")

:author: Najam Nazar
:version: 2.0.0
:license: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from sumslice_common import (
    DISPLAY_NAMES, ENV_FILE, RESULTS_DIR, load_prompts, load_reference,
    load_system, normalise, resolve_systems,
)

load_dotenv(ENV_FILE)
JudgeFn = Callable[..., Optional[str]]

# A reply that does not parse is nearly always a truncation: the judge opened
# with prose and ran out of budget before the JSON. Retrying at the same limit
# would reproduce it, so the retry gets a larger budget, as in PR_LLM.
REPAIR_TOKEN_MULTIPLIER = 6


def require_env(name: str, cast: Callable[[str], object]) -> object:
    """Read a judge setting from .env, with no in-code default.

    DPS_LLM does the same, so that the documented configuration can never be
    silently overridden by a fallback value baked into the source.
    """
    raw = os.getenv(name)
    if not raw:
        raise SystemExit(f"{name} is not set in {ENV_FILE}")
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(f"{name} must be {cast.__name__}, got {raw!r}")


def make_openrouter_judge(model: str, max_tokens: int, temperature: float) -> JudgeFn:
    import requests

    api_key = require_env("OPENROUTER_API_KEY", str)
    api_url = require_env("OPENROUTER_API_URL", str)

    def call(prompt: str, tokens: Optional[int] = None) -> Optional[str]:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": tokens or max_tokens}
        for attempt in range(3):
            try:
                r = requests.post(api_url, json=payload, timeout=60,
                                  headers={"Authorization": f"Bearer {api_key}"})
                r.raise_for_status()
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"      [API error] {exc}")
                time.sleep(2 ** attempt)
        return None

    return call


def presentation_orders(systems: List[str], seed: int, key: str, n: int) -> List[List[str]]:
    rng = random.Random(int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:16], 16))
    first = systems[:]
    rng.shuffle(first)
    orders = [first]
    if n >= 2:
        orders.append(first[::-1])
    for _ in range(2, n):
        extra = systems[:]
        rng.shuffle(extra)
        orders.append(extra)
    return orders


def parse_order(reply: Optional[str], letters: List[str]) -> Optional[List[str]]:
    if not reply:
        return None
    text = reply.strip()
    cands = []
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            order = json.loads(m.group(0)).get("order")
            if isinstance(order, list):
                cands.append([str(x).strip().upper() for x in order])
        except (json.JSONDecodeError, AttributeError):
            pass
    if not cands:
        tokens = re.findall(r"\b([A-Z])\b", text.upper())
        if tokens and re.fullmatch(r"[\sA-Z,>\-\[\]\"'.]*", text.upper()):
            cands.append(tokens)
    for c in cands:
        if len(c) == len(letters) and set(c) == set(letters):
            return c
    return None


def build_prompt(template: str, crit: dict, reference: str, texts: List[str]) -> str:
    letters = list(string.ascii_uppercase[:len(texts)])
    return template.format(
        criterion_name=crit["name"], criterion_definition=crit["definition"],
        human_summary=reference,
        summaries_block="\n\n".join(f"{l}. {t}" for l, t in zip(letters, texts)),
        k=len(texts), example_order=", ".join(f'"{l}"' for l in letters),
    )


def run(systems: List[str], judge: JudgeFn, judge_model: str, max_tokens: int, seed: int,
        n_orders: int, retries: int, limit: Optional[int], fresh: bool) -> List[tuple]:
    cfg = load_prompts()
    template, criteria = cfg["summary_ranking_template"], cfg["summary_ranking_criteria"]
    ref = load_reference()
    data = {s: load_system(s) for s in systems}
    for s, d in data.items():
        if not d:
            raise SystemExit(f"No summaries for {s}; run the Java LlmSummaryGenerator first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / "judge_checkpoint.json"
    ckpt: Dict[str, dict] = {} if fresh or not ckpt_path.exists() else json.loads(ckpt_path.read_text())
    sig = {"systems": sorted(systems), "seed": seed, "judge_model": judge_model,
           "prompt_sha": hashlib.sha256(json.dumps([template, criteria], sort_keys=True).encode()).hexdigest()[:12]}

    keys = sorted(ref)[:limit] if limit else sorted(ref)
    skipped = []
    for n, key in enumerate(keys, 1):
        texts = {s: normalise(data[s].get(key, {}).get("summary", "")) for s in systems}
        if not all(texts.values()):
            skipped.append(key)
            continue
        reference = normalise(ref[key]["summary"])
        for crit_key, crit in criteria.items():
            for o_idx, order in enumerate(presentation_orders(systems, seed, f"{key}|{crit_key}", n_orders)):
                ck = f"{key[0]}::{key[1]}::{crit_key}::{o_idx}"
                done = ckpt.get(ck)
                if done and done.get("status") == "ok" and done.get("config") == sig:
                    continue
                letters = list(string.ascii_uppercase[:len(order)])
                prompt = build_prompt(template, crit, reference, [texts[s] for s in order])
                raws, parsed = [], None
                for attempt in range(retries + 1):
                    p = prompt if attempt == 0 else prompt + (
                        f"\n\nYour previous reply was not valid. Reply with exactly one JSON object "
                        f"containing all {len(letters)} letters.")
                    raw = judge(p) if attempt == 0 else judge(p, max_tokens * REPAIR_TOKEN_MULTIPLIER)
                    raws.append(raw)
                    parsed = parse_order(raw, letters)
                    if parsed:
                        break
                ckpt[ck] = {"project": key[0], "method_id": key[1], "criterion": crit_key,
                            "presentation": o_idx, "mapping": dict(zip(letters, order)),
                            "raw_replies": raws, "attempts": len(raws), "order_letters": parsed,
                            "status": "ok" if parsed else "failed", "config": sig}
                ckpt_path.write_text(json.dumps(ckpt, indent=1))
        print(f"[{n}/{len(keys)}] {key[0]} {ref[key]['class']}.{ref[key]['name']} done")
    if skipped:
        print(f"WARNING: {len(skipped)} methods skipped (a model has no summary): {skipped[:5]}")
    return skipped


def aggregate(systems: List[str]) -> Tuple[pd.DataFrame, dict]:
    path = RESULTS_DIR / "judge_checkpoint.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run the judge first (without --aggregate-only).")
    ckpt = json.loads(path.read_text())
    k = len(systems)
    rows = []
    for e in ckpt.values():
        if e["status"] != "ok" or sorted(e["mapping"].values()) != sorted(systems):
            continue
        slots = sorted(e["mapping"])
        for rank, letter in enumerate(e["order_letters"], 1):
            rows.append({"project": e["project"], "method_id": e["method_id"], "criterion": e["criterion"],
                         "presentation": e["presentation"], "system": e["mapping"][letter],
                         "slot": slots.index(letter) + 1, "rank": rank, "points": k - rank + 1})
    long = pd.DataFrame(rows)
    long.to_csv(RESULTS_DIR / "judge_rankings_long.csv", index=False)
    if not long.empty:
        firsts = long[long["rank"] == 1]
        bias = (firsts.groupby("slot").size() / len(firsts)).rename("share_ranked_first").reset_index()
        bias["expected_if_unbiased"] = 1 / k
        bias.to_csv(RESULTS_DIR / "judge_position_bias.csv", index=False)
    failed = sum(e["status"] != "ok" for e in ckpt.values())
    counts = {"judge_calls": len(ckpt), "failed_after_retries": failed,
              "items_ranked": int(long.groupby(["project", "method_id"]).ngroups) if not long.empty else 0}
    print(f"Judge calls: {len(ckpt)}, failed after retries: {failed}, "
          f"methods ranked: {counts['items_ranked']}")
    return long, counts


def summarise(long: pd.DataFrame) -> dict:
    item = long.groupby(["project", "method_id", "criterion", "system"])["points"].mean().reset_index()
    out = {}
    for system, g in item.groupby("system"):
        out[system] = {
            "overall_mean_points": float(g["points"].mean()),
            "criteria": {c: {"mean_points": float(x["points"].mean()), "n_methods": int(len(x))}
                         for c, x in g.groupby("criterion")},
            "by_project": {p: {c: float(y["points"].mean()) for c, y in gp.groupby("criterion")}
                           for p, gp in g.groupby("project")},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", nargs="*", default=["ALL"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--orders", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and re-judge from scratch")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    systems = resolve_systems(args.systems)
    judge_model = os.getenv("RANK_SUMMARIES_MODEL", "")
    skipped: List[tuple] = []
    if not args.aggregate_only:
        if not judge_model:
            raise SystemExit(f"RANK_SUMMARIES_MODEL is not set in {ENV_FILE}")
        max_tokens = require_env("RANK_SUMMARIES_MAX_TOKENS", int)
        temperature = require_env("OPENROUTER_TEMPERATURE", float)
        print(f"Judge model : {judge_model}\nMax tokens  : {max_tokens}\nTemperature : {temperature}")
        print(f"Criteria    : {', '.join(load_prompts()['summary_ranking_criteria'])}")
        judge = make_openrouter_judge(judge_model, max_tokens, temperature)
        skipped = run(systems, judge, judge_model, max_tokens, args.seed, args.orders,
                      args.retries, args.limit, args.fresh)

    long, counts = aggregate(systems)
    if long.empty:
        raise SystemExit("No valid judge rankings yet.")
    path = RESULTS_DIR / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    prompts = load_prompts()
    prev = results.get("judge_meta", {})
    results["judge"] = summarise(long)
    results["judge_meta"] = {"judge_model": judge_model or prev.get("judge_model"),
                             "systems": systems, "seed": args.seed,
                             "presentations_per_item": args.orders,
                             "points": f"1st={len(systems)} ... last=1",
                             "reference": "SumSlice tool summaries (input/tool_summaries/SUMSLICE_TOOL_SUMMARY.jsonl)",
                             "criteria": list(prompts["summary_ranking_criteria"]),
                             "criteria_definitions": prompts["summary_ranking_criteria"],
                             "skipped_methods": ([list(k) for k in skipped] if not args.aggregate_only
                                                 else prev.get("skipped_methods", [])),
                             **counts}
    path.write_text(json.dumps(results, indent=2))
    table = (long.groupby(["project", "method_id", "criterion", "system"])["points"].mean()
             .groupby(["system", "criterion"]).mean().unstack().rename(index=DISPLAY_NAMES))
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
