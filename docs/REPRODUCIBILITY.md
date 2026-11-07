# Reproducibility Notes

## Data preparation and splits

- Karate Club: NetworkX Zachary karate club graph.
- Netscience: GML network reduced to its largest connected component.
- Outer test split: 30% of original edges held out as positive test examples; an equal number of uniformly sampled non-edges used as negative examples.
- Outer split seed: 42.
- Inner validation split: 30% of the outer training edges, with an equal number of negative validation pairs.
- Inner split seed: 123.
- Optimizer seeds: 0 through 9.
- Validation features are calculated on the inner training graph. Final test features are calculated on the full outer training graph.
- Feature scaling: MinMaxScaler fitted on validation features and applied to test features.
- Optimizer weight bounds: [0, 1].

## Network sizes and split counts

| Dataset | Original nodes | Original edges | Outer train edges | Test positive | Test negative | Inner train edges | Validation positive | Validation negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Karate Club | 34 | 78 | 55 | 23 | 23 | 39 | 16 | 16 |
| Netscience LCC | 379 | 914 | 640 | 274 | 274 | 448 | 192 | 192 |

## Optimizer configurations

| Algorithm | Parameters |
|---|---|
| PSO | Swarm size 10; inertia 0.7; cognitive coefficient 1.5; social coefficient 1.5 |
| Firefly | Population 10; alpha 0.2; beta0 1; gamma 1 |
| ABC | Colony size 10; food sources 5; abandonment limit 5 |

## Fitness-evaluation budgets

The four budgets are 50, 500, 5,000, and 20,000 objective/fitness evaluations. The evaluation counter, rather than iteration count, is used as the budget measure.

## Software and runtime

The project uses Python and the dependencies listed in requirements.txt. Hardware specifications, exact package versions, and wall-clock runtimes were not captured in the experiment summary and should be added if available.