# Budget Curve for Metaheuristic Link Prediction

**Research title:** Where Does the Swarm Stop Paying Off? A Budget Curve for Metaheuristic Link Prediction

**Authors**
- Sheza Khader Askander — Department of Artificial Intelligence and Data Science, Chaitanya Bharathi Institute of Technology, Gandipet, Hyderabad, India
- Pulipati Srilatha — Associate Professor, Department of Artificial Intelligence and Data Science, Chaitanya Bharathi Institute of Technology, Gandipet, Hyderabad, India

## Project overview

This repository evaluates how fitness-evaluation budgets affect metaheuristic feature-weight optimization for link prediction. It compares Particle Swarm Optimization (PSO), the Firefly Algorithm (FA), and Artificial Bee Colony (ABC) using four topological similarity features:

1. Common Neighbors (CN)
2. Jaccard Coefficient
3. Adamic–Adar (AA)
4. Resource Allocation (RA)

The experiments use Zachary's Karate Club network and the largest connected component of the Netscience collaboration network. The tested budgets are 50, 500, 5,000, and 20,000 fitness evaluations, with ten optimizer seeds per configuration.

## Repository structure

- `data/` — Netscience GML dataset.
- `src/` — data preparation, feature extraction, fitness function, baselines, and optimizer implementations.
- `experiments/` — experiment runners and diagnostic scripts.
- `results/` — run-level CSV outputs and available summaries.
- `docs/` — experiment notes, result interpretation, and reproducibility information.

## Main experimental findings

Across the reported 240 optimizer runs, ABC obtained the highest mean test ROC-AUC on both datasets. Increasing the budget from 50 to 20,000 evaluations produced negligible changes in the reported test performance. See [the results summary](docs/RESULTS_SUMMARY.md) for the reported values and interpretation.

## Reproducibility

Install the dependencies listed in `requirements.txt`, then run the corresponding experiment scripts from the repository root:

```bash
python experiments/run_pso.py
python experiments/run_firefly.py
python experiments/run_abc.py
```

The scripts write run-level results under `results/`. Check the constants at the top of each runner before rerunning experiments, and do not overwrite the archived results unless that is intended.

## Important interpretation notes

- The outer train/test graph split is fixed; the ten seeds measure optimizer-seed variation, not independent graph-split variation.
- Negative edges are uniformly sampled, so they may be easier than deliberately selected hard negatives.
- Fitness-evaluation count is the computational-budget measure. Wall-clock runtime and hardware details should be reported separately if measured.
- The comparison with standalone classical baselines should be treated as provisional until identity-weight AA and RA checks are confirmed under the same scaling and evaluation pipeline.

## Research paper

The revised research paper and its rendered PDF should be added under `docs/` when the binary files are uploaded.