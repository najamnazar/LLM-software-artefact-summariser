# sumslice-llm

`sumslice-llm` is a Java command-line project that analyzes Java source code and generates method summaries with an LLM.

It runs in three stages:

1. **Static analysis stage** (`MethodFeatureExtractor`, Java)
   - Parses Java files with Eclipse JDT AST.
   - Extracts method metadata (name, class, return type, arguments).
   - Infers lightweight SWUM-style action/object labels from method names.
   - Builds call/called-by relationships.
   - Writes structured JSON and intermediate artifacts.
2. **LLM summarization stage** (`LlmSummaryGenerator`, Java)
   - Reads generated method JSON.
   - Builds prompts from `resources/prompts.json`.
   - Calls an OpenRouter-compatible chat completion API.
   - Writes method summaries JSON.
3. **Evaluation stage** (`evaluate_sumslice_summaries.py` + `rank_sumslice_summaries.py`, Python)
   - Scores generated summaries against ground truth (BERTScore + TF-IDF cosine) and produces violin plots.
   - Runs a rubric-based LLM judge that ranks the models across quality criteria.

## Project Structure

- `src/`
  - `MethodFeatureExtractor.java`: Static analysis entry point.
  - `LlmSummaryGenerator.java`: LLM summarization entry point.
  - `MethodInfo.java`, `SwumRecord.java`, `JsonWriter.java`: supporting model/output classes.
- `bin/`: Compiled `.class` files (created by the `javac -d bin` build step below).
- `python/`
  - `evaluate_sumslice_summaries.py`: Evaluation stage — BERTScore + cosine metrics vs ground truth, violin plots.
  - `rank_sumslice_summaries.py`: Evaluation stage — rubric-based LLM ranking across quality criteria.
- `input/corpus/`: Input Java corpora (used by Stage 1).
- `input/ground-truth/`: Reference summaries for evaluation.
- `dataset/`: Raw source checkouts for the corpus projects (jEdit, jtopas, nanoxml, siena-master, jhotdraw60b1, jajuk-src-1.10.5) — predates `input/corpus/` and is not read by any current script; kept for provenance, safe to remove if not needed.
- `output/`: Generated artifacts from Stages 1–2.
- `evaluation_results/`: Generated artifacts from Stage 3 (CSVs, `rubric_results.json`, `violin_scores.png`).
- `.env`: API configuration (required for LLM stage).

`resources/prompts.json` lives at the **repo root** (`../resources/prompts.json` relative to `SUMSLICE_LLM/`), shared with DPS_LLM and PR_LLM — it is not a `SUMSLICE_LLM` subdirectory, despite the relative paths used in the commands below.

## Prerequisites

1. **Java JDK 14+** installed and available in terminal (`java`, `javac`).
2. **Eclipse JDT Core jar(s)** for AST parsing.
   - This repository uses jars from the sibling project `../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib`.
3. Internet access for OpenRouter (or compatible API endpoint).

## Dependencies

This project imports `org.eclipse.jdt.core.dom.*`.

Compile and run with this on the classpath:

- `../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*`

## Configuration (`.env`)

`LlmSummaryGenerator` requires:

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_API_URL` (required)
- `OPENROUTER_TEMPERATURE` (required)
- `<MODEL_KEY>_MODEL` for whichever model key you pass on the command line, e.g. `QWEN_MODEL`, `GPT_MODEL`, `CLAUDE_MODEL`, `MISTRAL_MODEL` (required for that key)

Optional:

- `OPENROUTER_MAX_COMPLETION_TOKENS` (default: `512`; falls back to the deprecated `OPENROUTER_MAX_TOKENS` if set)

Security note:

- Do not commit real API keys.
- Rotate any key that has been exposed.

## Build From Terminal (bash)

Run from the `SUMSLICE_LLM` directory:

```bash
mkdir -p bin
javac -cp "../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" -d bin src/*.java
```

## Run Stage 1: Static Analysis

Generate method graph + metadata JSON for a project's source:

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" MethodFeatureExtractor \
  input/corpus/nanoxml \
  output/corpus/nanoxml
```

Generated files (named after the source folder):

- `output/corpus/nanoxml/nanoxml-methods.json`
- `output/corpus/nanoxml/NanoXML.out`
- `output/corpus/nanoxml/nanograph-ast.txt`

Repeat for each project folder under `input/corpus/` (`jajuk`, `jEdit`, `jhotdraw`, `jtopas`, `nanoxml`, `siena-master`), pointing `outputDir` at the matching subfolder under `output/corpus/`.

## Run Stage 2: LLM Summary Generation

`LlmSummaryGenerator` has two modes: single-project and `ALL` (loops over every project folder automatically).

### Single project, single model

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator \
  output/corpus/nanoxml/nanoxml-methods.json \
  output/corpus/nanoxml/nanoxml-qwen-summaries.json \
  ../.env \
  -1 \
  ../resources/prompts.json \
  QWEN \
  input/ground-truth/nanoXML-example-summaries.json
```

Arguments for single-project mode:

1. Input methods JSON path (Stage 1 output).
2. Output summaries JSON path.
3. `.env` path.
4. Max methods (`-1` means all methods that survive the ground-truth filter).
5. Prompt config JSON path.
6. **Model key** (required) — selects which `<KEY>_MODEL` env var to use. One of `QWEN`, `GPT`, `CLAUDE`, `MISTRAL`.
7. Ground-truth JSON path (optional) — when given, only methods whose `methodId` appears in this file are summarized.

Swap the model key (and output filename) to run a different model, e.g. `GPT`, `CLAUDE`, or `MISTRAL`:

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator \
  output/corpus/nanoxml/nanoxml-methods.json \
  output/corpus/nanoxml/nanoxml-gpt-summaries.json \
  ../.env -1 ../resources/prompts.json GPT \
  input/ground-truth/nanoXML-example-summaries.json
```

### All projects, one model (recommended for the full corpus)

This loops every project under `output/corpus/`, auto-matches each project's ground-truth file in `input/ground-truth/`, and writes `output/corpus/<project>/<project>-<model>-summaries.json`:

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator \
  ALL QWEN output/corpus ../.env -1 ../resources/prompts.json input/ground-truth
```

Run again with a different model key to generate the other sets:

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator ALL GPT    output/corpus ../.env -1 ../resources/prompts.json input/ground-truth
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator ALL CLAUDE output/corpus ../.env -1 ../resources/prompts.json input/ground-truth
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator ALL MISTRAL output/corpus ../.env -1 ../resources/prompts.json input/ground-truth
```

Arguments for `ALL` mode:

1. Literal `ALL`.
2. Model key (required) — `QWEN`, `GPT`, `CLAUDE`, or `MISTRAL`.
3. Directory containing one subfolder per project, each with a `*-methods.json` file (default: `output/all`).
4. `.env` path (default: `../.env`, i.e. repo root).
5. Max methods per project (default: `-1`, all).
6. Prompt config JSON path (default: `../resources/prompts.json`, i.e. repo root).
7. Ground-truth directory (default: `input/ground-truth`) — files are matched to project folders by name prefix.

## Run Stage 3: Evaluation (Python)

Run from the `SUMSLICE_LLM` directory with a Python environment that has `pandas`, `bert-score`, `scikit-learn`, `matplotlib`, `python-dotenv`, and `requests` installed (see DPS_LLM/PR_LLM for equivalent dependency lists — this project has no separate `requirements.txt` of its own).

### Step 1 — Metrics (BERTScore + cosine) and violin plots

```bash
python python/evaluate_sumslice_summaries.py
```

Reads `output/SUMSLICE_{MODEL}_SUMMARY.json` for each model and the ground-truth files under `input/ground-truth/`, and writes to `evaluation_results/`:
- `{model}_vs_gt_method_scores.csv` — per-method metric scores
- `{model}_vs_gt_project_scores.csv` — aggregated by project
- `overall_comparison.csv` — all models side by side
- `evaluation_summary.txt`, `results.txt`
- `violin_scores.png` — distribution plot

Useful flags: `--output-dir`, `--gt-dir`, `--results-dir`, `--models CLAUDE GPT MISTRAL QWEN`, `--bertscore-lang`.

> **Known gap**: `evaluate_sumslice_summaries.py` expects one flat, all-projects file per model directly under `output/` (`output/SUMSLICE_{MODEL}_SUMMARY.json`, with a top-level `summaries` list spanning every project). The `ALL`-mode `LlmSummaryGenerator` command documented above writes **per-project** files under `output/corpus/<project>/<project>-<model>-summaries.json` instead. There is currently no script in this repo that merges the per-project outputs into the flat file Stage 3 expects — the `output/SUMSLICE_*_SUMMARY.json` files present in this repo were produced by a step not captured here. If you're re-running the pipeline from scratch, you'll need to merge the per-project `summaries` arrays yourself (or write a small aggregation script) before running Stage 3.

### Step 2 — Rubric-based LLM ranking

Run **after** Step 1. Requires at least two model summary files and an OpenRouter-compatible judge configured in `.env` (see Configuration above).

```bash
python python/rank_sumslice_summaries.py
```

Loads ranking criteria from `../resources/prompts.json` (`sumslice_llm.sumslice-llm.summary_ranking`), ranks all available models against ground truth, and writes `evaluation_results/rubric_results.json`.

Quick smoke-test: `python python/rank_sumslice_summaries.py --limit 5`

---

## End-to-End Example (bash)

```bash
mkdir -p bin
javac -cp "../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" -d bin src/*.java
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" MethodFeatureExtractor input/corpus/nanoxml output/corpus/nanoxml
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator \
  output/corpus/nanoxml/nanoxml-methods.json \
  output/corpus/nanoxml/nanoxml-qwen-summaries.json \
  ../.env -1 ../resources/prompts.json QWEN \
  input/ground-truth/nanoXML-example-summaries.json
```

## Windows Notes

Use `;` instead of `:` in the classpath, and `\` for paths if you prefer, e.g.:

```powershell
java -cp "bin;..\ORIGINAL_SUMSLICE\SumsliceXMLGenerator\lib\*" MethodFeatureExtractor input\corpus\nanoxml output\corpus\nanoxml
```

## CLI Defaults and Important Notes

- `MethodFeatureExtractor` defaults to `input/corpus/nanoxml` and `output` if no args are given.
- `LlmSummaryGenerator` (single-project mode) defaults to:
  - `output/nanoxml-methods.json`
  - `output/nanoxml-method-summaries.json`
  - `../.env` (repo root)
  - max methods `-1`
  - `../resources/prompts.json` (repo root)
  - **model key has no default — it is required as the 6th argument**, or the command throws `Model key required as 6th argument. Example: MISTRAL, GPT, CLAUDE, QWEN`.

## Output Schema (High Level)

Stage 1 (`nanoxml-methods.json`) contains:

- `method_list.averages.called`
- `method_list.averages.calls`
- `method_list.method[]` entries with:
  - id, name, class, returntype
  - top called-by and calls ids
  - inferred `use` object (`type`, `example`)
  - inferred `swum` object (`verb`, `object`)

Stage 2 (`nanoxml-method-summaries.json`) contains per-method natural language summaries.

## Troubleshooting

- `javac: command not found` or `java: command not found`
  - Install JDK 14+ and add it to PATH.
- `package org.eclipse.jdt.core.dom does not exist`
  - Verify classpath points to JDT jars (for this workspace: `../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*`).
- `Method JSON not found`
  - Run Stage 1 first and check the output path.
- `Env file not found`
  - Ensure `.env` exists and pass the correct path.
- `Missing required env key: <KEY>_MODEL`
  - Add `<KEY>_MODEL` to `.env` for the model key you passed on the command line.
- `Model key required as 6th argument. Example: MISTRAL, GPT, CLAUDE, QWEN`
  - Single-project mode needs the model key as the 6th argument — see "Run Stage 2" above.
- `LLM API error (4xx/5xx)`
  - Verify API key, URL, model name, and network access.

## Main Classes

- `MethodFeatureExtractor`: Static parser + method graph/data exporter.
- `LlmSummaryGenerator`: Prompt builder + API caller + summary exporter.
- `SwumRecord`: Lightweight SWUM-style extraction from method names.
- `JsonWriter`: Stage 1 JSON serializer.
