# Pull Request Summarizer (PR-LLM)

A research pipeline that generates pull-request descriptions from commit messages using multiple LLMs (GPT, Qwen, Claude, Mistral) via the OpenRouter API, then evaluates the generated summaries with automated metrics (BERTScore, cosine similarity) and a multi-criteria LLM rubric judge.

## Overview

For each commit record in the dataset the pipeline:

1. Calls every configured LLM to produce a PR description from commit messages and inline code comments.
2. Scores each generated description against the human-written reference using **BERTScore F1** (roberta-large) and **TF-IDF cosine similarity** (sklearn).
3. Persists per-sample results to one JSONL file per model and writes an aggregated `results.json`.
4. Optionally runs a rubric-based LLM judge that ranks the four model outputs on five criteria and merges the scores back into `results.json`.

The pipeline is split across three scripts that mirror the DPS_LLM structure:

| Script | Role |
|--------|------|
| `pr_llm_summariser.py` | Step 1 — call LLMs, write per-model JSONL files (summaries only) |
| `evaluate_pr_summaries.py` | Step 2 — compute BERTScore + cosine metrics, rewrite JSONL files, write `results.json` |
| `rank_pr_summaries.py` | Step 3 — rubric-based LLM ranking across 5 quality criteria |

---

## Project Structure

```
PR_LLM/
├── .env                                    # API credentials (not committed)
├── input/
│   ├── dataset/
│   │   ├── selected_commits.jsonl          # Input dataset (JSONL)
│   │   └── selected_commits.csv            # Input dataset (CSV alternative)
│   └── ground_truth/
│       └── *.txt                           # Human-written reference descriptions
├── python/
│   ├── pr_llm_summariser.py                # Stage 1 — generate summaries (LLM API calls)
│   ├── evaluate_pr_summaries.py            # Stage 2 — BERTScore + cosine metrics
│   └── rank_pr_summaries.py                # Stage 3 — rubric-based LLM ranking
│
└── output/                                 # Created automatically on first run
    ├── PR_GPT_SUMMARY.jsonl
    ├── PR_QWEN_SUMMARY.jsonl
    ├── PR_CLAUDE_SUMMARY.jsonl
    ├── PR_MISTRAL_SUMMARY.jsonl
    ├── results.json                        # Aggregated BERTScore + cosine metrics
    └── rubric_eval_checkpoint.json         # Checkpoint for rubric evaluation

# resources/prompts.json now lives at the REPO ROOT (../resources/prompts.json),
# shared by DPS_LLM, PR_LLM, and SUMSLICE_LLM — it is not a PR_LLM subdirectory.
```

---

## Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Python | 3.9 | Run all pipeline scripts |
| pip | 21.0 | Python package installation |
| OpenRouter API key | — | LLM generation and rubric judging |

---

## Installation

### 1. Set up a Python virtual environment

**Linux / macOS**
```bash
# From the TOSEM repository root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai python-dotenv bert-score scikit-learn torch requests
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install openai python-dotenv bert-score scikit-learn torch requests
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install openai python-dotenv bert-score scikit-learn torch requests
```

> **PyTorch / GPU note**: The commands above install the CPU version of PyTorch, which works on all machines. If you have an NVIDIA GPU, install the matching CUDA build first:
> ```bash
> # CUDA 11.8
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> # CUDA 12.1
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Configure the `.env` file

The `.env` file at `PR_LLM/.env` holds all API credentials and model identifiers. Edit the values to match your OpenRouter account and desired models:

```dotenv
# ── OpenRouter connection (required) ─────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions

# ── LLM generation settings ──────────────────────────────────────────────────
OPENROUTER_MAX_COMPLETION_TOKENS=512
OPENROUTER_TEMPERATURE=0

# ── Models for PR summary generation ─────────────────────────────────────────
GPT_MODEL=openai/gpt-5.4-mini
QWEN_MODEL=qwen/qwen3.7-plus
CLAUDE_MODEL=anthropic/claude-sonnet-4.6
MISTRAL_MODEL=mistralai/mistral-small-2603

# ── Model used by the rubric evaluation judge ─────────────────────────────────
RANK_SUMMARIES_MODEL=meta-llama/llama-4-maverick
RANK_SUMMARIES_MAX_TOKENS=50

```

> Both scripts resolve `PR_LLM/.env` automatically from their location — no need to specify it on the command line.

---

## Running the Pipeline

### Execution order

```
Step 1 → pr_llm_summariser.py    — generate PR descriptions, write per-model JSONL files
Step 2 → evaluate_pr_summaries.py — compute BERTScore / cosine metrics, write results.json
Step 3 → rank_pr_summaries.py    — rubric-based LLM ranking across 5 quality criteria
```

---

### Step 1 — Generate PR summaries

Run `pr_llm_summariser.py` from anywhere — it resolves all paths relative to its own location (`PR_LLM/python/`).
Writes per-model JSONL files containing summary text only (no metrics yet).

**Linux / macOS**
```bash
# Run all four models (default)
python PR_LLM/python/pr_llm_summariser.py

# Equivalent explicit form
python PR_LLM/python/pr_llm_summariser.py ALL

# Run specific models
python PR_LLM/python/pr_llm_summariser.py GPT QWEN CLAUDE

# Run a single model
python PR_LLM/python/pr_llm_summariser.py GPT
```

**Windows (PowerShell / Command Prompt)**
```cmd
python PR_LLM\python\pr_llm_summariser.py ALL
python PR_LLM\python\pr_llm_summariser.py GPT QWEN CLAUDE
python PR_LLM\python\pr_llm_summariser.py MISTRAL
```

#### Overriding default paths

The defaults resolve to `PR_LLM/input/dataset/`, `PR_LLM/output/`, and the repo-root `resources/` automatically. Override with flags if you need to point elsewhere:

**Linux / macOS**
```bash
python PR_LLM/python/pr_llm_summariser.py ALL \
  --dataset PR_LLM/input/dataset/selected_commits.jsonl \
  --output PR_LLM/output
```

**Windows (PowerShell)**
```powershell
python PR_LLM\python\pr_llm_summariser.py ALL `
  --dataset PR_LLM\input\dataset\selected_commits.jsonl `
  --output PR_LLM\output
```

#### Quick smoke-test with a sample limit

```bash
# Process only the first 5 entries (useful for verifying the setup)
python PR_LLM/python/pr_llm_summariser.py GPT --limit 5

# Two models, 10 entries
python PR_LLM/python/pr_llm_summariser.py GPT CLAUDE --limit 10
```

#### All command-line flags

```
usage: pr_llm_summariser.py [MODEL ...] [options]

positional arguments:
  MODEL                 One or more of: GPT, QWEN, CLAUDE, MISTRAL, ALL
                        Default: ALL

optional arguments:
  --dataset PATH        Path to selected_commits.jsonl or .csv
                        Default: <PR_LLM>/input/dataset/selected_commits.jsonl
  --output DIR          Directory for per-model JSONL summary files
                        Default: <PR_LLM>/output/
  --prompt PATH         Path to prompts.json
                        Default: <repo root>/resources/prompts.json
  --limit N             Process only the first N samples (debugging)
  --temperature FLOAT   Sampling temperature for all models
                        Default: OPENROUTER_TEMPERATURE from .env, else 0.0
  --max-tokens INT      Max tokens per generated summary
                        Default: OPENROUTER_MAX_COMPLETION_TOKENS from .env, else 512
  --log-level LEVEL     Logging verbosity: DEBUG, INFO, WARNING, ERROR
                        Default: INFO
```

#### Output files (after Step 1)

| File | Description |
|------|-------------|
| `PR_GPT_SUMMARY.jsonl` | Per-sample GPT summaries (no metrics yet) |
| `PR_QWEN_SUMMARY.jsonl` | Per-sample Qwen summaries |
| `PR_CLAUDE_SUMMARY.jsonl` | Per-sample Claude summaries |
| `PR_MISTRAL_SUMMARY.jsonl` | Per-sample Mistral summaries |

Metrics are added to these files by Step 2.

---

### Step 2 — Compute BERTScore and cosine metrics

Run `evaluate_pr_summaries.py` **after** Step 1. Reads the JSONL files, batch-computes BERTScore (roberta-large, one call per model label) and TF-IDF cosine similarity, rewrites each JSONL file with a `"metrics"` key per record, and writes aggregated `results.json`.

**Linux / macOS**
```bash
# Evaluate all four models
python PR_LLM/python/evaluate_pr_summaries.py

# Evaluate specific models
python PR_LLM/python/evaluate_pr_summaries.py GPT QWEN

# Quick smoke-test (5 entries)
python PR_LLM/python/evaluate_pr_summaries.py GPT --limit 5
```

**Windows (PowerShell / Command Prompt)**
```cmd
python PR_LLM\python\evaluate_pr_summaries.py ALL
python PR_LLM\python\evaluate_pr_summaries.py GPT --limit 5
```

#### All command-line flags

```
usage: evaluate_pr_summaries.py [MODEL ...] [options]

positional arguments:
  MODEL                 One or more of: GPT, QWEN, CLAUDE, MISTRAL, ALL
                        Default: ALL

optional arguments:
  --output DIR          Directory containing JSONL summary files
                        Default: <PR_LLM>/output/
  --limit N             Only evaluate the first N entries per model
  --bertscore-lang STR  Language for BERTScore — 'en' selects roberta-large
                        Default: en
  --log-level LEVEL     Logging verbosity: DEBUG, INFO, WARNING, ERROR
                        Default: INFO
```

#### Output files (after Step 2)

| File | Description |
|------|-------------|
| `PR_GPT_SUMMARY.jsonl` | Per-sample GPT summaries + BERTScore / cosine metrics |
| `PR_QWEN_SUMMARY.jsonl` | Per-sample Qwen summaries + metrics |
| `PR_CLAUDE_SUMMARY.jsonl` | Per-sample Claude summaries + metrics |
| `PR_MISTRAL_SUMMARY.jsonl` | Per-sample Mistral summaries + metrics |
| `results.json` | Dataset-level averages (BERTScore P/R/F1, cosine) for all evaluated models |

---

### Step 3 — Rubric-based LLM ranking

Run `rank_pr_summaries.py` **after** Step 2 has produced `results.json`. The script reads from `PR_LLM/output/` and writes ranking results back into `results.json`.

**Linux / macOS**
```bash
# From anywhere — the script resolves its own paths from __file__
python PR_LLM/python/rank_pr_summaries.py
```

**Windows**
```cmd
python PR_LLM\python\rank_pr_summaries.py
```

#### Quick smoke-test

```bash
# Rank only the first 5 entries
python PR_LLM/python/rank_pr_summaries.py --limit 5
```

#### All command-line flags

```
usage: rank_pr_summaries.py [options]

optional arguments:
  --limit N   Only process the first N common entries (useful for testing)
```

#### Evaluation criteria

Each PR entry is ranked 1–4 per criterion; points awarded: 1st = 4, 2nd = 3, 3rd = 2, 4th = 1.

| Criterion | What is measured |
|-----------|-----------------|
| Accuracy | Does the description correctly represent the changes? |
| Adequacy | Does the description cover the main aspects of the change? |
| Conciseness | Is the description brief while still conveying essential information? |
| Context Awareness | Does the description reflect relevant PR context (commits, rationale)? |
| Clarity | Is the description easy for reviewers to understand? |

#### Output

Rubric results are merged into `PR_LLM/output/results.json` under a `"rubric_evaluation"` key for each model. A checkpoint file (`rubric_eval_checkpoint.json`) is updated after each entry so interrupted runs can resume without re-querying already-evaluated entries.

> **Note:** `rank_pr_summaries.py` reads `results.json` at the end to merge ranking data. Run Step 2 (`evaluate_pr_summaries.py`) first so `results.json` exists.

---

## Quick Reference: Full Pipeline Commands

### Linux / macOS

```bash
# Activate virtual environment (from TOSEM root)
source .venv/bin/activate

# Step 1 — generate summaries (all models)
python PR_LLM/python/pr_llm_summariser.py ALL

# Step 1 — single model run
python PR_LLM/python/pr_llm_summariser.py GPT

# Step 2 — compute BERTScore / cosine metrics (all models)
python PR_LLM/python/evaluate_pr_summaries.py ALL

# Step 2 — quick test (5 entries)
python PR_LLM/python/evaluate_pr_summaries.py GPT --limit 5

# Step 3 — rubric ranking (all models)
python PR_LLM/python/rank_pr_summaries.py

# Step 3 — rubric ranking (quick test, 10 entries)
python PR_LLM/python/rank_pr_summaries.py --limit 10
```

### Windows (PowerShell)

```powershell
# Activate virtual environment (from TOSEM root)
.\.venv\Scripts\Activate.ps1

# Step 1 — generate summaries
python PR_LLM\python\pr_llm_summariser.py ALL

# Step 2 — compute metrics
python PR_LLM\python\evaluate_pr_summaries.py ALL

# Step 3 — rubric ranking
python PR_LLM\python\rank_pr_summaries.py
```

### Windows (Command Prompt)

```cmd
:: Activate virtual environment (from TOSEM root)
.venv\Scripts\activate.bat

:: Step 1 — generate summaries
python PR_LLM\python\pr_llm_summariser.py ALL

:: Step 2 — compute metrics
python PR_LLM\python\evaluate_pr_summaries.py ALL

:: Step 3 — rubric ranking
python PR_LLM\python\rank_pr_summaries.py
```

---

## Configuration Reference

### `.env` keys

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter authentication key |
| `OPENROUTER_API_URL` | Yes | — | OpenRouter endpoint URL |
| `GPT_MODEL` | Yes (if using GPT) | — | OpenRouter model ID for GPT |
| `QWEN_MODEL` | Yes (if using Qwen) | — | OpenRouter model ID for Qwen |
| `CLAUDE_MODEL` | Yes (if using Claude) | — | OpenRouter model ID for Claude |
| `MISTRAL_MODEL` | Yes (if using Mistral) | — | OpenRouter model ID for Mistral |
| `OPENROUTER_MAX_COMPLETION_TOKENS` | No | `512` | Max tokens per generated summary |
| `OPENROUTER_TEMPERATURE` | No | `0.0` | Sampling temperature (0.0 = greedy) |
| `OPENROUTER_HTTP_REFERER` | No | — | Optional HTTP-Referer header sent to OpenRouter |
| `OPENROUTER_APP_TITLE` | No | — | Optional X-Title header sent to OpenRouter |
| `RANK_SUMMARIES_MODEL` | Yes (Step 2) | — | Judge model used by `evaluate_pr_summaries.py` |
| `RANK_SUMMARIES_MAX_TOKENS` | No | `100` | Max tokens for judge ranking responses |

### Python dependencies

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI-compatible SDK used to call OpenRouter |
| `python-dotenv` | `.env` file loading |
| `bert-score` | BERTScore semantic similarity (uses roberta-large via `lang='en'`) |
| `scikit-learn` | TF-IDF cosine similarity |
| `torch` | Required by BERTScore |
| `requests` | HTTP calls in `evaluate_pr_summaries.py` |

---

## Metrics Explained

### BERTScore (Precision / Recall / F1)
Computes token-level semantic overlap between the generated and reference description using `roberta-large` (selected via `lang='en'`). F1 is the primary comparison metric. Downloaded automatically on first use (~500 MB). Matches the approach used in DPS_LLM.

### Cosine Similarity (0–1)
Computed with TF-IDF vectors (sklearn `TfidfVectorizer` + `cosine_similarity`). No model download or load is needed — a fresh vectorizer is fitted on each (generated, reference) pair. Matches DPS_LLM's `MetricsCalculator.cosine_similarity`.

### Rubric Points (1–4)
The judge LLM ranks all four model outputs from best (1st = 4 pts) to worst (4th = 1 pt) on each of the five criteria. Per-model averages and totals are reported in `results.json`.

---

## Notes

- **First run**: BERTScore downloads `roberta-large` (~500 MB) on the first scoring call. It is cached at `~/.cache/huggingface/hub/` and reused on all subsequent runs across any project. Cosine similarity uses TF-IDF (no download needed).
- **Resuming**: `pr_llm_summariser.py` and `evaluate_pr_summaries.py` always start fresh (truncating existing JSONL files). `rank_pr_summaries.py` uses `rubric_eval_checkpoint.json` to resume interrupted runs without re-querying completed entries.
- **Rate limits**: If a model returns a rate-limit error, the pipeline logs the error, records an empty summary for that sample, and continues with the remaining samples and models.
- **Single-model runs**: `evaluate_pr_summaries.py` can be run on a subset of models (e.g. `GPT QWEN`). `rank_pr_summaries.py` requires at least two model JSONL files to produce a comparative ranking.

---

## License

MIT

---

**Last Updated**: June 2026
**Version**: 1.0.0
