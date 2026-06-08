# Design Pattern Summarizer (DPS-LLM)

A research pipeline that generates and evaluates natural language summaries of Java design pattern implementations using three automated approaches — NLG, SWUM, and LLM — benchmarked against human-written summaries.

## Overview

The project compares three summary generation methods across 150 Java class files drawn from three open-source repositories, covering nine design patterns:

| Method | Technique | External dependency |
|--------|-----------|-------------------|
| **DPS-NLG** | Template-based Natural Language Generation (SimpleNLG) | None |
| **DPS-SWUM** | Software Word Usage Model — identifier decomposition | None (reads NLG JSON) |
| **DPS-LLM** | Large Language Model via OpenRouter API | API key required |

Generated summaries are evaluated against 150 human-written references using:

- **Automated metrics**: TF-IDF Cosine Similarity and BERTScore (Precision / Recall / F1)
- **Additional NLG metrics**: BLEU, ROUGE, METEOR
- **Multi-criteria LLM ranking**: An LLM judge ranks three summaries across five quality criteria (Accuracy, Conciseness, Adequacy, Code Context, Design Pattern Recognition)
- **Statistical tests**: Paired Wilcoxon signed-rank tests and Friedman non-parametric tests with Bonferroni-corrected post-hoc analysis

---

## Project Structure

```
DPS_LLM/
├── src/main/java/
│   ├── common/
│   │   ├── projectparser/          # Java source parser (JavaParser + symbol solver)
│   │   │   └── ParseProject.java   # Core parser; produces nested HashMaps
│   │   ├── designpatternidentifier/# Rule-based pattern detection (9 patterns)
│   │   └── utils/                  # Shared helpers (Utils, ProjectPathFormatter)
│   │
│   ├── dps_nlg/                    # Pipeline 1 — NLG
│   │   ├── Application.java        # Entry point: discovers input/, runs NLG
│   │   └── summarygenerator/       # SimpleNLG sentence builders + templates
│   │
│   ├── dps_llm/                    # Pipeline 2 — LLM (OpenRouter)
│   │   ├── DpsLlmApplication.java  # Entry point: reads .env, calls API
│   │   ├── client/LlmClient.java   # HTTP client for OpenRouter
│   │   ├── config/DotEnvLoader.java# .env parser
│   │   ├── model/                  # ClassFeatureSnapshot (immutable DTO)
│   │   ├── prompt/                 # PromptManager + LlmPromptBuilder
│   │   └── summary/                # ClassFeatureExtractor, LlmSummaryService,
│   │                               #   LlmSummaryWriter
│   │
│   └── dps_swum/                   # Pipeline 3 — SWUM
│       ├── SWUMApplication.java    # Entry point: reads NLG JSON output
│       └── swum/                   # SWUM grammar, summarizer, pipeline
│
├── python/                         # Evaluation scripts (run after Java)
│   ├── evaluate_summaries.py       # Cosine Similarity + BERTScore vs human
│   ├── evaluate_nlg_additional_metrics.py  # BLEU, ROUGE, METEOR for NLG
│   ├── evaluate_iterations.py      # Metrics across prompt word-limit variants
│   ├── rank_summaries.py           # LLM multi-criteria ranking (A vs B vs C)
│   ├── rank_model_comparisons.py   # LLM model head-to-head ranking
│   ├── rank_design_patterns.py     # Ranking grouped by design pattern
│   ├── calculate_wilcoxon_tests.py # Paired Wilcoxon tests (NLG/SWUM vs LLM)
│   ├── friedman_test.py            # Friedman + post-hoc Bonferroni analysis
│   ├── spearman_sensitivity_analysis.py  # Correlation between metrics
│   ├── summary_length_stats.py     # Word/char count stats per method
│   ├── summary_length_stats_llm_prompts.py  # Stats across prompt variants
│   └── analyze_conciseness.py      # Conciseness analysis vs word-limit targets
│
├── input/
│   ├── AbdurRKhalid/{pattern}/     # Educational design pattern examples
│   ├── JamesZBL/{pattern}/         # Alternative implementations
│   ├── spring-framework/{pattern}/ # Real-world enterprise patterns
│   └── DPS_Human_Summaries.csv    # 150 human-written reference summaries
│
├── output/
│   ├── json-output/
│   │   ├── nlg/    # Parsed JSON per project (27 files) — produced by DPS-NLG
│   │   ├── swum/   # (if applicable)
│   │   └── llm/    # Parsed JSON per project (27 files) — produced by DPS-LLM
│   └── summary-output/
│       ├── nlg_summaries.csv          # NLG-generated summaries
│       ├── swum_summaries.csv         # SWUM-generated summaries
│       ├── LLM_CLAUDE_SUMMARY.csv     # Claude summaries
│       ├── LLM_GEMINI_SUMMARY.csv     # Gemini summaries
│       ├── LLM_GPT_SUMMARY.csv        # GPT summaries
│       └── LLM_MISTRAL_SUMMARY.csv    # Mistral summaries
│
├── evaluation-results/             # All evaluation outputs
│   ├── {method}_vs_human_class_scores.csv    # Per-file metric scores
│   ├── {method}_vs_human_project_scores.csv  # Aggregated by project
│   ├── {method}_vs_human_pattern_scores.csv  # Aggregated by design pattern
│   ├── multi_criteria_rankings.csv            # LLM judge ranks per file
│   ├── results.txt                            # Appended statistical test output
│   ├── evaluation_summary.txt                 # Summary of all methods
│   └── methods_comparison_violin_plots.png    # Distribution plots
│
├── resources/
│   └── prompts.json                # System prompt templates (20/40/50/60/80 words)
├── pom.xml                         # Maven build + exec targets
├── requirements.txt                # Python dependencies
└── .env                            # API credentials (not committed)
```

---

## Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| JDK | 14 | Compile and run Java pipelines |
| Maven | 3.6 | Build management and execution |
| Python | 3.8 | Evaluation scripts |
| pip | 21.0 | Python package installation |
| OpenRouter API key | — | DPS-LLM and LLM ranking scripts |

---

## Installation

### 1. Build the Java project

```bash
# Linux / macOS
mvn clean install

# Windows (Command Prompt or PowerShell)
mvn clean install
```

> Maven downloads all Java dependencies (JavaParser, SimpleNLG, Jackson, commons-collections4) automatically from the POM.

### 2. Set up a Python virtual environment

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **PyTorch / GPU note**: `requirements.txt` installs the CPU version of PyTorch, which works on all machines and is sufficient for BERTScore evaluation. If you have an NVIDIA GPU, install the matching CUDA build manually *before* running `pip install -r requirements.txt`:
> ```bash
> # CUDA 11.8
> pip install torch>=2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
> # CUDA 12.1
> pip install torch>=2.0.0+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Configure the `.env` file

Create a file named `.env` in the project root. All Java and Python components read this file automatically.

```dotenv
# ── OpenRouter connection (required for DPS-LLM and ranking scripts) ──────────
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions

# ── Model to use for summary generation (set exactly one) ────────────────────
# The first non-empty value found wins in this order: MISTRAL → GPT → CLAUDE → GEMINI
MISTRAL_MODEL=mistralai/mistral-small-2503
GPT_MODEL=openai/gpt-4.1-mini
CLAUDE_MODEL=anthropic/claude-sonnet-4-5
GEMINI_MODEL=google/gemini-2.0-flash-001

# ── LLM generation settings ──────────────────────────────────────────────────
OPENROUTER_MAX_TOKENS=256
OPENROUTER_TEMPERATURE=0.2

# ── Optional: HTTP headers forwarded to OpenRouter ───────────────────────────
OPENROUTER_HTTP_REFERER=https://yoursite.example.com
OPENROUTER_TITLE=DPS-LLM Research

# ── Model used by the Python ranking scripts (rank_summaries.py etc.) ────────
RANK_SUMMARIES_MODEL=meta-llama/llama-3.1-70b-instruct

# ── Optional: override the default output CSV path for LLM summaries ─────────
# LLM_SUMMARY_PATH=output/summary-output/my_custom_summaries.csv

# ── Optional: process only the first N project directories ───────────────────
# LLM_PROJECT_LIMIT=5

# ── Optional: run multiple prompt word-limit variants in one pass ─────────────
# LLM_PROMPT_ALIASES=SENIOR_ANALYST_50_WORDS
# LLM_PROMPT_ALIASES=SENIOR_ANALYST_20_WORDS,SENIOR_ANALYST_50_WORDS,SENIOR_ANALYST_80_WORDS
```

---

## Running the Pipelines

### Execution order

```
Step 1  →  DPS-NLG   (Java)   — must run first; SWUM depends on its JSON output
Step 2  →  DPS-SWUM  (Java)   — reads output/json-output/nlg/
Step 3  →  DPS-LLM   (Java)   — independent; requires API key
Step 4  →  evaluate_summaries.py        — requires steps 1–3 outputs
Step 5  →  evaluate_nlg_additional_metrics.py  — requires step 1 output
Step 6  →  rank_summaries.py            — requires A/B/C CSVs + API key
Step 7  →  calculate_wilcoxon_tests.py  — requires step 4 outputs
Step 8  →  friedman_test.py             — requires step 6 output
Step 9  →  (optional analysis scripts)
```

---

### Stage 1 — Java: Generate Summaries

All commands below must be run from the **project root** (the directory containing `pom.xml`).

#### 1a. DPS-NLG (template-based, no API needed)

**Linux / macOS**
```bash
mvn exec:java@dps-app
```

**Windows (Command Prompt / PowerShell)**
```cmd
mvn exec:java@dps-app
```

Outputs:
- `output/json-output/nlg/*.json` — 27 structured JSON files (one per project/pattern pair)
- `output/summary-output/nlg_summaries.csv` — 150 NLG-generated summaries

#### 1b. DPS-SWUM (identifier analysis, reads NLG JSON)

> Run **after** DPS-NLG. The application will exit with an error if `output/json-output/` is missing.

**Linux / macOS**
```bash
mvn exec:java@swum-pipeline
```

**Windows**
```cmd
mvn exec:java@swum-pipeline
```

Output:
- `output/summary-output/swum_summaries.csv` — 150 SWUM-generated summaries

#### 1c. DPS-LLM (OpenRouter API)

> Requires `OPENROUTER_API_KEY` and at least one model key set in `.env`.

**Linux / macOS**
```bash
mvn exec:java@llm-summaries
```

**Windows**
```cmd
mvn exec:java@llm-summaries
```

Output:
- `output/summary-output/LLM_{MODEL}_SUMMARY.csv` — 150 LLM-generated summaries
- `output/json-output/llm/*.json` — 27 structured JSON files

**Override the model at runtime** (takes priority over `.env` model keys):

```bash
# Linux
mvn exec:java@llm-summaries -Dexec.args="anthropic/claude-sonnet-4-5"

# Windows
mvn exec:java@llm-summaries "-Dexec.args=anthropic/claude-sonnet-4-5"
```

**Run multiple prompt word-limit variants in one pass** (via `.env`):

```bash
# Set in .env:
#   LLM_PROMPT_ALIASES=SENIOR_ANALYST_20_WORDS,SENIOR_ANALYST_50_WORDS,SENIOR_ANALYST_80_WORDS
# Then run:
mvn exec:java@llm-summaries
```

Each alias produces a separate CSV file under `output/summary-output/`.

**Available prompt aliases** (defined in `resources/prompts.json`):

| Alias | Word limit |
|-------|-----------|
| `SENIOR_ANALYST_20_WORDS` | 20 words |
| `SENIOR_ANALYST_40_WORDS` | 40 words |
| `SENIOR_ANALYST_50_WORDS` | 50 words (default) |
| `SENIOR_ANALYST_60_WORDS` | 60 words |
| `SENIOR_ANALYST_80_WORDS` | 80 words |

#### Maven execution IDs (full reference)

| ID | Main class | Purpose |
|----|-----------|---------|
| `dps-app` | `dps_nlg.Application` | NLG summary generation |
| `swum-pipeline` | `dps_swum.swum.SWUMEvaluationPipeline` | SWUM summary generation |
| `llm-summaries` | `dps_llm.DpsLlmApplication` | LLM summary generation |
| `prompt-demo` | `dps_llm.prompt.PromptConfigurationExample` | Print available prompt aliases |

---

### Stage 2 — Python: Evaluation and Analysis

All commands must be run from the **project root** with the virtual environment activated.

**Linux / macOS — activate venv**
```bash
source .venv/bin/activate
```

**Windows PowerShell — activate venv**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt — activate venv**
```cmd
.venv\Scripts\activate.bat
```

---

#### Step 4 — Automated metrics (Cosine Similarity + BERTScore)

Evaluates all generated summaries against human references and produces per-class, per-project, and per-pattern score files plus violin plots.

```bash
# Linux / macOS / Windows
python python/evaluate_summaries.py
```

Outputs written to `evaluation-results/`:
- `{method}_vs_human_class_scores.csv` — one row per matched class
- `{method}_vs_human_project_scores.csv` — aggregated by project
- `{method}_vs_human_pattern_scores.csv` — aggregated by design pattern
- `overall_comparison.csv` — all methods side by side
- `evaluation_summary.txt` — plain-text summary report
- `methods_comparison_violin_plots.png` — distribution charts

> **Note**: BERTScore downloads a BERT model on first run (~420 MB). Subsequent runs use the cached model. Expect 2–5 minutes per method on CPU.

The script auto-discovers CSV files by default paths. Override with flags:

```bash
python python/evaluate_summaries.py \
  --human-csv input/DPS_Human_Summaries.csv \
  --nlg-csv output/summary-output/nlg_summaries.csv \
  --swum-csv output/summary-output/swum_summaries.csv \
  --llm-claude-csv output/summary-output/LLM_CLAUDE_SUMMARY.csv \
  --llm-gemini-csv output/summary-output/LLM_GEMINI_SUMMARY.csv \
  --llm-gpt-csv output/summary-output/LLM_GPT_SUMMARY.csv \
  --llm-mistral-csv output/summary-output/LLM_MISTRAL_SUMMARY.csv \
  --output-dir evaluation-results
```

**Windows** (PowerShell multi-line):
```powershell
python python/evaluate_summaries.py `
  --nlg-csv output/summary-output/nlg_summaries.csv `
  --llm-claude-csv output/summary-output/LLM_CLAUDE_SUMMARY.csv `
  --output-dir evaluation-results
```

---

#### Step 5 — Additional NLG metrics (BLEU, ROUGE, METEOR)

```bash
python python/evaluate_nlg_additional_metrics.py
```

Output:
- `evaluation-results/nlg_vs_human_additional_metrics.csv`
- `evaluation-results/nlg_vs_human_additional_metrics_summary.csv`

---

#### Step 6 — Multi-criteria LLM ranking

Ranks three blinded summary corpora (`A.csv`, `B.csv`, `C.csv`) against human references across five criteria using an LLM judge. Requires `RANK_SUMMARIES_MODEL` in `.env`.

> Place the three method CSVs you want to compare as `output/summary-output/A.csv`, `B.csv`, `C.csv` before running. Typically A = NLG, B = LLM, C = SWUM.

```bash
python python/rank_summaries.py
```

Override the ranking model at the command line:

```bash
python python/rank_summaries.py --model meta-llama/llama-3.1-70b-instruct
```

Outputs:
- `evaluation-results/multi_criteria_rankings.csv` — per-file ranks for all 5 criteria + winner column
- `evaluation-results/multi_criteria_rankings_reasoning.csv` — same with LLM reasoning text
- `evaluation-results/multi_criteria_rankings_summary.csv` — aggregate statistics per criterion

**Ranking criteria** (each scored 3/2/1 for 1st/2nd/3rd place):

| Criterion | What is measured |
|-----------|-----------------|
| Accuracy | Factual faithfulness to source code |
| Conciseness | Information density without padding |
| Adequacy | Coverage of important functionality |
| Code Context | Reflection of class relationships and intent |
| Design Pattern Recognition | Identification and description of pattern roles |

---

#### Step 6b — Model comparison ranking

Compares individual LLM models head-to-head using the same ranking criteria.

```bash
python python/rank_model_comparisons.py
```

Outputs written to `evaluation-results/`:
- `model_comparisons_ranking_report.txt`
- `model_comparisons_ranking_detail.csv`
- `model_comparisons_ranking_summary.csv`

---

#### Step 6c — Design pattern ranking

Applies the ranking pipeline grouped by design pattern.

```bash
python python/rank_design_patterns.py
```

---

#### Step 7 — Wilcoxon signed-rank tests

Runs paired Wilcoxon signed-rank tests comparing DPS_NLG vs DPS_LLM and DPS_SWUM vs DPS_LLM on both Cosine Similarity and BERTScore F1.

> **Prerequisite**: Step 4 must have run; the script reads `*_vs_human_class_scores.csv` from `evaluation-results/`.

```bash
python python/calculate_wilcoxon_tests.py
```

Results are appended to `evaluation-results/results.txt`.

---

#### Step 8 — Friedman test + post-hoc analysis

Applies the Friedman non-parametric test (two levels) and Bonferroni-corrected pairwise Wilcoxon post-hoc tests.

> **Prerequisite**: Step 6 must have run; the script reads `evaluation-results/multi_criteria_rankings.csv`.

```bash
python python/friedman_test.py
```

Results are printed to the console and appended to `evaluation-results/results.txt`.

**Test design**:

| Part | Blocks | Treatments | Metric source |
|------|--------|------------|--------------|
| Part 1 — per-criterion | ~150 class files | 3 systems | Rank scores (3/2/1) from `multi_criteria_rankings.csv` |
| Part 2 — cross-criteria | 5 criteria | 3 systems | Mean score per system per criterion |

Effect sizes reported as Kendall's W (0.1 weak → 0.3 moderate → 0.5 strong → 0.7 very strong).

---

#### Step 9 — Optional analysis scripts

```bash
# Spearman correlation between automated metrics and human-criteria rankings
python python/spearman_sensitivity_analysis.py

# Word and character count statistics per method
python python/summary_length_stats.py

# Same statistics across LLM prompt word-limit variants
python python/summary_length_stats_llm_prompts.py

# Conciseness analysis: does output stay within the instructed word limit?
python python/analyze_conciseness.py

# Metrics across LLM prompt word-limit variants (20/40/60/80-word runs)
python python/evaluate_iterations.py
```

---

## Quick Reference: Full Pipeline Commands

### Linux / macOS (Bash)

```bash
# ── Environment ───────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── Stage 1: Generate summaries ──────────────────────────────────────────────
mvn exec:java@dps-app          # NLG  → output/summary-output/nlg_summaries.csv
mvn exec:java@swum-pipeline    # SWUM → output/summary-output/swum_summaries.csv
mvn exec:java@llm-summaries    # LLM  → output/summary-output/LLM_*_SUMMARY.csv

# ── Stage 2: Evaluate ─────────────────────────────────────────────────────────
python python/evaluate_summaries.py                # Cosine + BERTScore
python python/evaluate_nlg_additional_metrics.py   # BLEU, ROUGE, METEOR

# ── Stage 2: Rank ─────────────────────────────────────────────────────────────
python python/rank_summaries.py                    # Multi-criteria LLM ranking
python python/rank_model_comparisons.py            # Model head-to-head
python python/rank_design_patterns.py              # Per-pattern ranking

# ── Stage 2: Statistical tests ────────────────────────────────────────────────
python python/calculate_wilcoxon_tests.py          # Paired Wilcoxon
python python/friedman_test.py                     # Friedman + post-hoc

# ── Stage 2: Descriptive analysis ────────────────────────────────────────────
python python/spearman_sensitivity_analysis.py
python python/summary_length_stats.py
python python/analyze_conciseness.py
```

### Windows (PowerShell)

```powershell
# ── Environment ───────────────────────────────────────────────────────────────
.\.venv\Scripts\Activate.ps1

# ── Stage 1: Generate summaries ──────────────────────────────────────────────
mvn exec:java@dps-app
mvn exec:java@swum-pipeline
mvn exec:java@llm-summaries

# ── Stage 2: Evaluate ─────────────────────────────────────────────────────────
python python/evaluate_summaries.py
python python/evaluate_nlg_additional_metrics.py

# ── Stage 2: Rank ─────────────────────────────────────────────────────────────
python python/rank_summaries.py
python python/rank_model_comparisons.py
python python/rank_design_patterns.py

# ── Stage 2: Statistical tests ────────────────────────────────────────────────
python python/calculate_wilcoxon_tests.py
python python/friedman_test.py

# ── Stage 2: Descriptive analysis ────────────────────────────────────────────
python python/spearman_sensitivity_analysis.py
python python/summary_length_stats.py
python python/analyze_conciseness.py
```

### Windows (Command Prompt)

```cmd
:: ── Environment ────────────────────────────────────────────────────────────
.venv\Scripts\activate.bat

:: ── Stage 1: Generate summaries ─────────────────────────────────────────────
mvn exec:java@dps-app
mvn exec:java@swum-pipeline
mvn exec:java@llm-summaries

:: ── Stage 2: Evaluate ────────────────────────────────────────────────────────
python python\evaluate_summaries.py
python python\evaluate_nlg_additional_metrics.py

:: ── Stage 2: Rank ────────────────────────────────────────────────────────────
python python\rank_summaries.py
python python\rank_model_comparisons.py

:: ── Stage 2: Statistical tests ───────────────────────────────────────────────
python python\calculate_wilcoxon_tests.py
python python\friedman_test.py

:: ── Stage 2: Descriptive analysis ───────────────────────────────────────────
python python\summary_length_stats.py
python python\analyze_conciseness.py
```

---

## Data Flow

```
input/ (150 Java files + DPS_Human_Summaries.csv)
    │
    ├─▶ [mvn dps-app]       DPS-NLG  ──▶ nlg_summaries.csv
    │                                  ──▶ json-output/nlg/*.json
    │                                              │
    ├─▶ [mvn swum-pipeline] DPS-SWUM ─────────────┘ (reads NLG JSON)
    │                                  ──▶ swum_summaries.csv
    │
    └─▶ [mvn llm-summaries] DPS-LLM  ──▶ LLM_{MODEL}_SUMMARY.csv
              (OpenRouter API)         ──▶ json-output/llm/*.json
                              │
                              ▼
         [evaluate_summaries.py]
              Cosine Similarity + BERTScore vs human
              ──▶ *_vs_human_class/project/pattern_scores.csv
              ──▶ violin_plots.png
                              │
                              ▼
         [rank_summaries.py]
              LLM judge: A vs B vs C on 5 criteria
              ──▶ multi_criteria_rankings.csv
                              │
              ┌───────────────┴──────────────────┐
              ▼                                  ▼
  [calculate_wilcoxon_tests.py]      [friedman_test.py]
    Paired Wilcoxon                  Friedman + post-hoc
    NLG/SWUM vs LLM                 Bonferroni correction
    ──▶ appended to results.txt     ──▶ appended to results.txt
```

---

## Design Patterns Supported

| Category | Patterns |
|----------|---------|
| Creational | Factory Method, Abstract Factory, Singleton |
| Structural | Adapter, Decorator, Facade |
| Behavioral | Observer, Visitor, Memento |

---

## Evaluation Results

### Automated Metrics (vs 150 human summaries)

| Method | Avg Cosine Similarity | Avg BERT F1 | Combined Score |
|--------|-----------------------|-------------|---------------|
| **LLM** | **0.3210** | 0.8622 | **0.5916** |
| SWUM | 0.2486 | **0.8642** | 0.5564 |
| NLG | 0.1628 | 0.8423 | 0.5025 |

### Friedman Test Results (multi-criteria ranking, 150 class files)

| Criterion | n | Friedman χ² | p-value | Kendall's W | Best system |
|-----------|---|------------|---------|-------------|-------------|
| Accuracy | 149 | 201.36 | < 0.001 | 0.676 (strong) | DPS_LLM |
| Conciseness | 149 | 11.93 | 0.0026 | 0.040 (weak) | **DPS_NLG** |
| Adequacy | 150 | 206.44 | < 0.001 | 0.688 (strong) | DPS_LLM |
| Code Context | 149 | 196.56 | < 0.001 | 0.660 (strong) | DPS_LLM |
| Design Pattern | 147 | 152.38 | < 0.001 | 0.518 (strong) | DPS_LLM |

Cross-criteria Friedman (5 criteria as blocks): **χ² = 8.40**, p = 0.015, **Kendall's W = 0.84** (very strong). Overall system ranking: **DPS_LLM > DPS_NLG > DPS_SWUM** across all criteria.

---

## Configuration Reference

### `.env` keys

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter authentication key |
| `OPENROUTER_API_URL` | Yes | — | OpenRouter endpoint URL |
| `MISTRAL_MODEL` | One of four | — | Mistral model ID (checked first) |
| `GPT_MODEL` | One of four | — | GPT model ID |
| `CLAUDE_MODEL` | One of four | — | Claude model ID |
| `GEMINI_MODEL` | One of four | — | Gemini model ID |
| `OPENROUTER_MAX_TOKENS` | No | `256` | Max tokens in LLM response |
| `OPENROUTER_TEMPERATURE` | No | `0.2` | Sampling temperature (0.0–2.0) |
| `OPENROUTER_HTTP_REFERER` | No | — | Optional HTTP-Referer header |
| `OPENROUTER_TITLE` | No | — | Optional X-Title header |
| `RANK_SUMMARIES_MODEL` | Yes (ranking) | — | Model used by Python ranking scripts |
| `RANK_SUMMARIES_API_URL` | No | `OPENROUTER_API_URL` | Separate endpoint for ranking |
| `RANK_SUMMARIES_MAX_TOKENS` | No | `50` | Max tokens for ranking responses |
| `LLM_PROMPT_ALIASES` | No | `SENIOR_ANALYST_50_WORDS` | Comma-separated aliases for multi-run |
| `LLM_SUMMARY_PATH` | No | `output/summary-output/llm_summaries.csv` | Override default output CSV path |
| `LLM_PROJECT_LIMIT` | No | unlimited | Process only the first N projects |

### Python dependencies

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `scipy` | Wilcoxon signed-rank and Friedman tests |
| `scikit-learn` | TF-IDF vectorization and cosine similarity |
| `bert-score`, `transformers` | BERTScore semantic similarity |
| `nltk`, `rouge-score` | BLEU, ROUGE, METEOR metrics |
| `matplotlib`, `seaborn` | Violin plots and charts |
| `requests` | HTTP calls to OpenRouter (ranking scripts) |
| `python-dotenv` | `.env` file loading |
| `torch` | Required by BERTScore / Transformers |

---

## Metrics Explained

### Cosine Similarity (0–1)
Measures lexical overlap using TF-IDF vectors. Captures word-choice similarity; sensitive to vocabulary differences between methods.

### BERTScore (Precision / Recall / F1)
Uses contextual BERT embeddings to measure semantic similarity beyond exact word matching. F1 is the primary comparison metric.

### Combined Score
`(Cosine Similarity + BERT F1) / 2` — a balanced measure of lexical and semantic quality used for ranking in output CSVs.

### Multi-Criteria Ranking (points)
1st place = 3 pts, 2nd = 2 pts, 3rd = 1 pt per criterion. Five criteria contribute to a total-points winner per class file.

---

## Input Data

### Code repositories (input/)

| Repository | Type | Patterns |
|-----------|------|---------|
| `AbdurRKhalid` | Educational examples | All 9 |
| `JamesZBL` | Alternative implementations | All 9 |
| `spring-framework` | Real-world enterprise code | All 9 |

### Human summaries (input/DPS_Human_Summaries.csv)

150 manually written class-level summaries with columns: `Project`, `Design Pattern`, `File Name`, `URL`, `Human Summary`.

---

## Limitations

1. **Java only**: The parser and pattern detectors target Java source code exclusively.
2. **Nine patterns**: Coverage limited to the nine GoF patterns listed above.
3. **LLM dependency**: The DPS-LLM pipeline and all ranking scripts require an OpenRouter API key and internet access.
4. **Token budget**: LLM summaries are capped at 256 tokens by default; very large classes are truncated.
5. **Evaluation scope**: Human reference summaries cover only the 150 files in the three input repositories.

---

## Research Context

This project supports research in automated code documentation and design pattern understanding. The evaluation methodology compares automated approaches against human judgment to assess readability, accuracy, completeness, and usefulness of generated summaries.

## Design Patterns Supported

- **Creational**: Factory Method, Abstract Factory, Singleton
- **Structural**: Adapter, Decorator, Facade
- **Behavioral**: Observer, Visitor, Memento

## License

[Your License Here]

## Citation

If you use this work in research, please cite:

```
[Add citation information]
```

## Acknowledgments

- Human summaries provided by domain experts
- Design pattern examples from open-source repositories (AbdurRKhalid, JamesZBL, Spring Framework)
- BERTScore, SimpleNLG, JavaParser, and OpenRouter libraries
- Statistical methodology: Wilcoxon signed-rank, Friedman, and Bonferroni correction

## Contact

[Add contact information]

---

**Last Updated**: June 2026
**Version**: 2.0
**Status**: Active Development
