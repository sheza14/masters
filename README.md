# Budget Curve for Metaheuristic Link Prediction

**Research title:** Where Does the Swarm Stop Paying Off? A Budget Curve for Metaheuristic Link Prediction

**Authors**
- Sheza Khader Askander — Department of Artificial Intelligence and Data Science, Chaitanya Bharathi Institute of Technology, Gandipet, Hyderabad, India
- Pulipati Srilatha — Associate Professor, Department of Artificial Intelligence and Data Science, Chaitanya Bharathi Institute of Technology, Gandipet, Hyderabad, India

## Project overview

This repository evaluates how fitness-evaluation budgets affect metaheuristic feature-weight optimization for link prediction. It compares Particle Swarm Optimization (PSO), the Firefly Algorithm (FA), and Artificial Bee Colony (ABC) using four topological similarity features: Common Neighbors (CN), Jaccard Coefficient, Adamic–Adar (AA), and Resource Allocation (RA).

The experiments use Zachary's Karate Club network and the largest connected component of the Netscience collaboration network. The tested budgets are 50, 500, 5,000, and 20,000 fitness evaluations, with ten optimizer seeds per configuration.

## Repository structure

- `data/` — Netscience GML dataset.
- `src/` — data preparation, feature extraction, fitness function, baselines, and optimizer implementations.
- `experiments/` — experiment runners and diagnostic scripts.
- `results/` — run-level CSV outputs currently present in the repository.
- `docs/` — experiment notes, result interpretation, and reproducibility information.

## Final reported findings

The revised research report records 240 runs (two datasets × three algorithms × four budgets × ten optimizer seeds). In those final reported results, ABC obtained the highest mean test ROC-AUC on both datasets, and increasing the budget from 50 to 20,000 evaluations produced negligible changes. See [RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Important data-integrity warning

The CSV files currently in `results/` have not yet been reconciled with the final values in the revised report. Do not cite or treat those existing CSV files as the final 240-run dataset until they are replaced with the matching final run-level exports. The current optimizer source files and experiment settings should also be checked against the final experiment configuration before rerunning.

## Reproducibility

Install dependencies listed in `requirements.txt`, then inspect the experiment constants and code before running:

```bash
python experiments/run_pso.py
python experiments/run_firefly.py
python experiments/run_abc.py
```

The scripts write run-level outputs under `results/`. Back up archived results before rerunning experiments.

## Interpretation limitations

- The outer graph split is fixed; the ten seeds measure optimizer-seed variation, not independent graph-split variation.
- Negative edges are uniformly sampled, so they may be easier than deliberately selected hard negatives.
- Fitness-evaluation count is the computational-budget measure. Wall-clock runtime and hardware details were not captured.
- Comparisons with standalone classical baselines should be treated as provisional until identity-weight AA and RA checks are confirmed under the same scaling and evaluation pipeline.

## Research paper

The revised Word report should be placed in `docs/` after downloading it from this conversation. Add the rendered PDF and final figures alongside it when available.