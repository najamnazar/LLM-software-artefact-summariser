"""evaluate_sumslice_summaries.py

Reference-based metrics for the four LLMs against the SumSlice (tool) summaries.

Pairs are matched by (project, method_id), so overloaded methods can no longer
be confused. For each pair: TF-IDF cosine (lexical), BERTScore P/R/F1
(roberta-large, no rescaling) and length in words, on normalised text.

Outputs in SUMSLICE_LLM/evaluation_results/:
  metrics_per_item.csv, results.json (key "metrics": overall and per project),
  violin_scores.png

:author: Najam Nazar
:version: 2.0.0
:license: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from sumslice_common import (
    DISPLAY_NAMES, RESULTS_DIR, load_reference, load_system, normalise, resolve_systems,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Same quieting as PR_LLM (evaluate_pr_summaries.py) and DPS_LLM: BERTScore pulls
# roberta-large through transformers/huggingface_hub, which otherwise log every
# cache revalidation at INFO and print a load report for the unused lm_head.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
for _noisy in ("transformers", "bert_score", "httpx", "huggingface_hub", "filelock",
               "urllib3", "py.warnings"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
_log = logging.getLogger("evaluate_sumslice")

BERT_MODEL = "roberta-large"
METRICS = ["cosine_similarity", "bert_precision", "bert_recall", "bert_f1", "words"]


def tfidf_cosine(a: str, b: str) -> float:
    try:
        m = TfidfVectorizer().fit_transform([a, b])
        return float(sk_cosine(m[0:1], m[1:2])[0, 0])
    except ValueError:
        return 0.0


def evaluate(systems: List[str], use_bert: bool, model_type: str, batch_size: int) -> pd.DataFrame:
    ref = load_reference()
    rows = []
    ref_words = {k: len(normalise(v["summary"]).split()) for k, v in ref.items()}
    for short in systems:
        data = load_system(short)
        if not data:
            _log.warning("[%s] no output file - skipped", short)
            continue
        missing = sorted(set(ref) - set(data))
        extra = sorted(set(data) - set(ref))
        if missing or extra:
            _log.warning("[%s] %d reference methods without summary, %d summaries without reference",
                         short, len(missing), len(extra))
        keys = [k for k in sorted(ref) if k in data and data[k].get("summary", "").strip()]
        cands = [normalise(data[k]["summary"]) for k in keys]
        refs = [normalise(ref[k]["summary"]) for k in keys]
        if use_bert:
            from bert_score import score
            _log.info("[%s] BERTScore on %d pairs", short, len(keys))
            p, r, f = score(cands, refs, model_type=model_type, rescale_with_baseline=False,
                            batch_size=batch_size, verbose=False)
            p, r, f = p.tolist(), r.tolist(), f.tolist()
        else:
            p = r = f = [None] * len(keys)
        for i, k in enumerate(keys):
            rows.append({
                "system": short, "project": k[0], "method_id": k[1],
                "class": ref[k]["class"], "name": ref[k]["name"],
                "cosine_similarity": tfidf_cosine(cands[i], refs[i]),
                "bert_precision": p[i], "bert_recall": r[i], "bert_f1": f[i],
                "words": len(cands[i].split()), "reference_words": ref_words[k],
            })
    return pd.DataFrame(rows)


def block(g: pd.DataFrame) -> dict:
    out = {"n": int(len(g))}
    for c in METRICS:
        v = pd.to_numeric(g[c], errors="coerce").dropna()
        out[f"mean_{c}"] = float(v.mean()) if len(v) else None
        out[f"sd_{c}"] = float(v.std(ddof=1)) if len(v) > 1 else None
    return out


def plot(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df.assign(model=df["system"].map(DISPLAY_NAMES))
    order = [DISPLAY_NAMES[s] for s in dict.fromkeys(df["system"])]
    panels = [("cosine_similarity", "Cosine similarity (TF-IDF)")]
    if d["bert_f1"].notna().any():
        panels.append(("bert_f1", "BERTScore F1"))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), squeeze=False)
    for ax, (metric, title) in zip(axes[0], panels):
        sub = d.dropna(subset=[metric])
        sns.violinplot(data=sub, x="model", y=metric, order=order, cut=0, color="#9ecae1", ax=ax)
        means = sub.groupby("model")[metric].mean()
        ax.scatter(range(len(order)), [means.get(m, np.nan) for m in order], marker="D", s=45,
                   color="darkred", zorder=3, label="Mean")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(loc="upper left")
    fig.suptitle("SumSlice: LLM summaries vs tool summaries", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", nargs="*", default=["ALL"])
    ap.add_argument("--bert-model", default=BERT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()

    df = evaluate(resolve_systems(args.systems), not args.no_bertscore, args.bert_model, args.batch_size)
    if df.empty:
        raise SystemExit("No LLM summaries found; run the Java LlmSummaryGenerator first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "metrics_per_item.csv", index=False)
    path = RESULTS_DIR / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results["metrics"] = {
        s: {"overall": block(g), "by_project": {p: block(gp) for p, gp in g.groupby("project")}}
        for s, g in df.groupby("system")
    }
    results["metrics_meta"] = {
        "reference": "SumSlice tool summaries (input/tool_summaries/SUMSLICE_TOOL_SUMMARY.jsonl)",
        "pairing": "(project, method_id)",
        "cosine": "TF-IDF, fitted per pair (lexical)",
        "bertscore_model": None if args.no_bertscore else args.bert_model,
        "bertscore_rescaled": False,
        "mean_reference_words": float(df.drop_duplicates(["project", "method_id"])["reference_words"].mean()),
    }
    path.write_text(json.dumps(results, indent=2))
    plot(df, RESULTS_DIR / "violin_scores.png")
    print(df.groupby("system")[METRICS].mean().rename(index=DISPLAY_NAMES).round(4).to_string())
    print(f"\nWrote {RESULTS_DIR}")


if __name__ == "__main__":
    main()
