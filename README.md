# LLM Software Artefact Summariser

A research repository evaluating how well Large Language Models (via OpenRouter — GPT, Claude, Mistral, Qwen) can summarise different kinds of software artefacts compared to human-written references and to non-LLM baselines. Each subproject targets a different artefact type but follows the same shape: **generate → evaluate (BERTScore + TF-IDF cosine) → rank (LLM rubric judge) → statistical analysis**.

## Subprojects

| Project | Artefact summarised | Language / stack | Docs |
|---------|---------------------|-------------------|------|
| [`DPS_LLM/`](DPS_LLM/README.md) | Java classes implementing GoF design patterns (150 files, 9 patterns, 3 source repos) | Java (Maven) + Python | [README](DPS_LLM/README.md) |
| [`PR_LLM/`](PR_LLM/README.md) | Pull requests (commit messages + inline code comments → PR description) | Python | [README](PR_LLM/README.md) |
| [`SUMSLICE_LLM/`](SUMSLICE_LLM/README.md) | Individual Java methods, using static analysis (Eclipse JDT AST) for context | Java + Python | [README](SUMSLICE_LLM/README.md) |

Each subproject's README documents its own project structure, prerequisites, `.env` configuration, and full pipeline commands — start there for anything project-specific.

## Shared resources

```
LLM-software-artefact-summariser/
├── resources/
│   └── prompts.json      # System/user prompt templates, shared across all three projects
├── DPS_LLM/
├── PR_LLM/
├── SUMSLICE_LLM/
├── LICENSE
└── README.md              # This file
```

`resources/prompts.json` lives at the repo root (not inside any individual project) because all three pipelines read from it. Each project's own scripts resolve the path relative to their own location (typically `../resources/prompts.json` from a `python/` or `src/` subdirectory) — see the relevant subproject README for the exact default.

Each subproject also expects its own `.env` file (API keys, model IDs, OpenRouter endpoint) — see each subproject's Configuration section for the required keys. `.env` files are git-ignored; never commit real API keys.

## Common pipeline shape

Although the three projects target different artefacts, they share a methodology:

1. **Generate** — call one or more LLMs (GPT, Claude, Mistral, Qwen) via OpenRouter to produce a summary/description of the artefact, using prompts from `resources/prompts.json`.
2. **Score against reference** — compare generated text to a human-written reference using TF-IDF cosine similarity and BERTScore (P/R/F1).
3. **Rank** — an LLM judge ranks the competing outputs (different models, or LLM vs. non-LLM baselines) across several quality criteria (accuracy, conciseness, adequacy, context-awareness, etc.).
4. **Statistical analysis** — where applicable, paired significance tests (Wilcoxon, Friedman + Bonferroni post-hoc) confirm whether differences between methods/models are significant.

Generated output (`output/`, `evaluation-results/` / `evaluation_results/`) and raw project checkouts are large and can go stale relative to the current code — check each subproject's README/output directory timestamps before trusting committed results, and re-run the relevant stage if in doubt.

## Prerequisites (union of all subprojects)

| Tool | Needed by |
|------|-----------|
| JDK 14+ | DPS_LLM, SUMSLICE_LLM |
| Maven 3.6+ | DPS_LLM |
| Python 3.8+ | All three (evaluation/ranking scripts) |
| OpenRouter API key | All three (LLM generation + rubric judging) |

See each subproject's README for exact Python dependency lists and installation steps.

## License

MIT — see [LICENSE](LICENSE).
