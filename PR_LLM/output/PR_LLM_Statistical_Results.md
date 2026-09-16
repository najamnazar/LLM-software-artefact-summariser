# PR-LLM Statistical Significance Tests

Mirrors DPS-LLM's ALL_RESULTS.md section 3.7. The deterministic tool baseline is tested alongside the LLMs. Ratings are 1st-to-last place judge rankings converted to points (1st=5 ... last=1); non-parametric tests used throughout since the data is ordinal.

## Friedman Omnibus Tests (per criterion, k=5 systems)

| Criterion | N Items | K Models | Friedman chi2 | p-value | Kendall's W | Significant |
|---|---|---|---|---|---|---|
| accuracy | 150 | 5 | 155.3333 | 1.464e-32 | 0.2589 | Yes |
| adequacy | 149 | 5 | 173.7289 | 1.656e-36 | 0.2915 | Yes |
| clarity | 150 | 5 | 175.6587 | 6.379e-37 | 0.2928 | Yes |
| conciseness | 150 | 5 | 209.1253 | 4.098e-44 | 0.3485 | Yes |
| context_awareness | 150 | 5 | 212.2293 | 8.807e-45 | 0.3537 | Yes |

## Holm-Corrected Pairwise Wilcoxon Post-Hoc (only for significant Friedman criteria)

| Criterion | Model A | Model B | Mean A | Mean B | p (raw) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|
| accuracy | Claude | GPT | 2.300 | 2.440 | 0.2711 | 0.5423 | No |
| accuracy | Claude | Mistral | 2.300 | 3.620 | 1.102e-10 | 7.713e-10 | Yes |
| accuracy | Claude | Qwen | 2.300 | 2.547 | 0.08286 | 0.2486 | No |
| accuracy | Claude | Tool | 2.300 | 4.093 | 1.848e-15 | 1.479e-14 | Yes |
| accuracy | GPT | Mistral | 2.440 | 3.620 | 8.999e-10 | 5.4e-09 | Yes |
| accuracy | GPT | Qwen | 2.440 | 2.547 | 0.4594 | 0.5423 | No |
| accuracy | GPT | Tool | 2.440 | 4.093 | 2.638e-20 | 2.638e-19 | Yes |
| accuracy | Mistral | Qwen | 3.620 | 2.547 | 8.927e-08 | 4.463e-07 | Yes |
| accuracy | Mistral | Tool | 3.620 | 4.093 | 0.02961 | 0.1184 | No |
| accuracy | Qwen | Tool | 2.547 | 4.093 | 4.917e-18 | 4.425e-17 | Yes |
| adequacy | Claude | GPT | 2.436 | 2.349 | 0.5424 | 1 | No |
| adequacy | Claude | Mistral | 2.436 | 3.591 | 7.938e-08 | 3.969e-07 | Yes |
| adequacy | Claude | Qwen | 2.436 | 2.409 | 0.9437 | 1 | No |
| adequacy | Claude | Tool | 2.436 | 4.215 | 1.773e-15 | 1.419e-14 | Yes |
| adequacy | GPT | Mistral | 2.349 | 3.591 | 2.851e-11 | 1.995e-10 | Yes |
| adequacy | GPT | Qwen | 2.349 | 2.409 | 0.8811 | 1 | No |
| adequacy | GPT | Tool | 2.349 | 4.215 | 5.566e-23 | 5.566e-22 | Yes |
| adequacy | Mistral | Qwen | 3.591 | 2.409 | 8.252e-09 | 4.951e-08 | Yes |
| adequacy | Mistral | Tool | 3.591 | 4.215 | 0.004611 | 0.01845 | Yes |
| adequacy | Qwen | Tool | 2.409 | 4.215 | 2.408e-21 | 2.167e-20 | Yes |
| clarity | Claude | GPT | 2.467 | 2.267 | 0.1769 | 0.7077 | No |
| clarity | Claude | Mistral | 2.467 | 3.893 | 4.144e-11 | 2.072e-10 | Yes |
| clarity | Claude | Qwen | 2.467 | 2.400 | 0.6011 | 1 | No |
| clarity | Claude | Tool | 2.467 | 3.973 | 2.669e-14 | 1.868e-13 | Yes |
| clarity | GPT | Mistral | 2.267 | 3.893 | 5.739e-17 | 4.591e-16 | Yes |
| clarity | GPT | Qwen | 2.267 | 2.400 | 0.4672 | 1 | No |
| clarity | GPT | Tool | 2.267 | 3.973 | 4.601e-20 | 4.141e-19 | Yes |
| clarity | Mistral | Qwen | 3.893 | 2.400 | 2.02e-11 | 1.212e-10 | Yes |
| clarity | Mistral | Tool | 3.893 | 3.973 | 0.8392 | 1 | No |
| clarity | Qwen | Tool | 2.400 | 3.973 | 8.134e-21 | 8.134e-20 | Yes |
| conciseness | Claude | GPT | 2.500 | 2.660 | 0.2821 | 0.8462 | No |
| conciseness | Claude | Mistral | 2.500 | 2.760 | 0.07738 | 0.4643 | No |
| conciseness | Claude | Qwen | 2.500 | 2.427 | 0.6019 | 1 | No |
| conciseness | Claude | Tool | 2.500 | 4.653 | 1.727e-20 | 1.382e-19 | Yes |
| conciseness | GPT | Mistral | 2.660 | 2.760 | 0.5855 | 1 | No |
| conciseness | GPT | Qwen | 2.660 | 2.427 | 0.1102 | 0.4643 | No |
| conciseness | GPT | Tool | 2.660 | 4.653 | 4.372e-21 | 3.935e-20 | Yes |
| conciseness | Mistral | Qwen | 2.760 | 2.427 | 0.08186 | 0.4643 | No |
| conciseness | Mistral | Tool | 2.760 | 4.653 | 1.502e-14 | 1.051e-13 | Yes |
| conciseness | Qwen | Tool | 2.427 | 4.653 | 1.797e-22 | 1.797e-21 | Yes |
| context_awareness | Claude | GPT | 2.747 | 2.287 | 0.001966 | 0.007863 | Yes |
| context_awareness | Claude | Mistral | 2.747 | 4.587 | 1.438e-19 | 1.294e-18 | Yes |
| context_awareness | Claude | Qwen | 2.747 | 2.333 | 0.01524 | 0.04572 | Yes |
| context_awareness | Claude | Tool | 2.747 | 3.047 | 0.06426 | 0.1285 | No |
| context_awareness | GPT | Mistral | 2.287 | 4.587 | 1.561e-23 | 1.561e-22 | Yes |
| context_awareness | GPT | Qwen | 2.287 | 2.333 | 0.983 | 0.983 | No |
| context_awareness | GPT | Tool | 2.287 | 3.047 | 7.359e-08 | 4.415e-07 | Yes |
| context_awareness | Mistral | Qwen | 4.587 | 2.333 | 6.362e-19 | 5.09e-18 | Yes |
| context_awareness | Mistral | Tool | 4.587 | 3.047 | 6.954e-12 | 4.868e-11 | Yes |
| context_awareness | Qwen | Tool | 2.333 | 3.047 | 5.121e-07 | 2.56e-06 | Yes |

## Spearman Sensitivity: Automated Metrics vs. Judge-Rubric Points

### vs. BERT F1

| Model | accuracy | adequacy | clarity | conciseness | context_awareness |
|---|---|---|---|---|---|
| Claude | -0.1614 | -0.1342 | -0.0472 | 0.1364 | -0.1940 |
| GPT | 0.2304 | 0.1157 | -0.0055 | -0.1474 | 0.1679 |
| Mistral | -0.1898 | -0.1359 | -0.0769 | -0.0353 | -0.1987 |
| Qwen | -0.0144 | -0.0453 | -0.0629 | -0.0298 | -0.0603 |
| Tool | 0.3657 | 0.3811 | 0.2940 | 0.2726 | 0.2796 |

### vs. Cosine Similarity

| Model | accuracy | adequacy | clarity | conciseness | context_awareness |
|---|---|---|---|---|---|
| Claude | -0.1393 | -0.1481 | -0.0953 | 0.0093 | -0.1662 |
| GPT | 0.1738 | 0.0364 | -0.0384 | -0.1532 | 0.0597 |
| Mistral | -0.0167 | -0.0091 | 0.0925 | -0.0198 | 0.1064 |
| Qwen | -0.2000 | -0.0486 | -0.0794 | 0.0369 | -0.1095 |
| Tool | 0.2763 | 0.3586 | 0.2856 | 0.1919 | 0.2668 |
