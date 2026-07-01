# sumslice-llm

`sumslice-llm` is a Java command-line project that analyzes Java source code and generates method summaries with an LLM.

It runs in two stages:

1. **Static analysis stage** (`MethodFeatureExtractor`)
   - Parses Java files with Eclipse JDT AST.
   - Extracts method metadata (name, class, return type, arguments).
   - Infers lightweight SWUM-style action/object labels from method names.
   - Builds call/called-by relationships.
   - Writes structured JSON and intermediate artifacts.
2. **LLM summarization stage** (`LlmSummaryGenerator`)
   - Reads generated method JSON.
   - Builds prompts from `resources/prompts.json`.
   - Calls an OpenRouter-compatible chat completion API.
   - Writes method summaries JSON.

## Project Structure

- `src/`
  - `MethodFeatureExtractor.java`: Static analysis entry point.
  - `LlmSummaryGenerator.java`: LLM summarization entry point.
  - `MethodInfo.java`, `SwumRecord.java`, `JsonWriter.java`: supporting model/output classes.
- `input/corpus/`: Input Java corpora.
- `input/ground-truth/`: Reference summaries for evaluation.
- `resources/prompts.json`: System prompt + user prompt template.
- `output/`: Generated artifacts.
- `.env`: API configuration (required for LLM stage).
- `bin/`: Compiled `.class` files.

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
  .env \
  -1 \
  resources/prompts.json \
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
  .env -1 resources/prompts.json GPT \
  input/ground-truth/nanoXML-example-summaries.json
```

### All projects, one model (recommended for the full corpus)

This loops every project under `output/corpus/`, auto-matches each project's ground-truth file in `input/ground-truth/`, and writes `output/corpus/<project>/<project>-<model>-summaries.json`:

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator \
  ALL QWEN output/corpus .env -1 resources/prompts.json input/ground-truth
```

Run again with a different model key to generate the other sets:

```bash
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator ALL GPT    output/corpus .env -1 resources/prompts.json input/ground-truth
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator ALL CLAUDE output/corpus .env -1 resources/prompts.json input/ground-truth
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator ALL MISTRAL output/corpus .env -1 resources/prompts.json input/ground-truth
```

Arguments for `ALL` mode:

1. Literal `ALL`.
2. Model key (required) — `QWEN`, `GPT`, `CLAUDE`, or `MISTRAL`.
3. Directory containing one subfolder per project, each with a `*-methods.json` file (default: `output/all`).
4. `.env` path (default: `.env`).
5. Max methods per project (default: `-1`, all).
6. Prompt config JSON path (default: `resources/prompts.json`).
7. Ground-truth directory (default: `input/ground-truth`) — files are matched to project folders by name prefix.

## End-to-End Example (bash)

```bash
mkdir -p bin
javac -cp "../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" -d bin src/*.java
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" MethodFeatureExtractor input/corpus/nanoxml output/corpus/nanoxml
java -cp "bin:../ORIGINAL_SUMSLICE/SumsliceXMLGenerator/lib/*" LlmSummaryGenerator \
  output/corpus/nanoxml/nanoxml-methods.json \
  output/corpus/nanoxml/nanoxml-qwen-summaries.json \
  .env -1 resources/prompts.json QWEN \
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
  - `.env`
  - max methods `-1`
  - `resources/prompts.json`
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
