# Experimental Results Summary

## Experimental design

- Datasets: Zachary's Karate Club and Netscience largest connected component (LCC).
- Algorithms: PSO, Firefly Algorithm, and Artificial Bee Colony (ABC).
- Fitness-evaluation budgets: 50, 500, 5,000, and 20,000.
- Optimizer seeds: 0–9.
- Total: 2 datasets × 3 algorithms × 4 budgets × 10 seeds = 240 runs.
- Metric: held-out test ROC-AUC.
- All reported runs used the requested evaluation budget.

## Mean test ROC-AUC ± standard deviation

### Karate Club

| Algorithm | 50 | 500 | 5,000 | 20,000 |
|---|---:|---:|---:|---:|
| PSO | 0.7533 ± 0.0045 | 0.7533 ± 0.0045 | 0.7533 ± 0.0045 | 0.7533 ± 0.0045 |
| Firefly | 0.7533 ± 0.0045 | 0.7533 ± 0.0045 | 0.7533 ± 0.0045 | 0.7533 ± 0.0045 |
| ABC | **0.7544 ± 0.0051** | **0.7544 ± 0.0051** | **0.7544 ± 0.0051** | **0.7544 ± 0.0051** |

### Netscience LCC

| Algorithm | 50 | 500 | 5,000 | 20,000 |
|---|---:|---:|---:|---:|
| PSO | 0.8597 ± 0.00001 | 0.8597 ± 0.00001 | 0.8597 ± 0.00001 | 0.8597 ± 0.00001 |
| Firefly | 0.8735 ± 0.00005 | 0.8735 ± 0.00005 | 0.8735 ± 0.00005 | 0.8735 ± 0.00005 |
| ABC | **0.8801 ± 0.00020** | 0.8800 ± 0.00008 | 0.8800 ± 0.00010 | 0.8800 ± 0.00010 |

## Classical baseline scores

| Dataset | CN | Jaccard | AA | RA | Best baseline |
|---|---:|---:|---:|---:|---|
| Karate Club | 0.6597 | 0.6134 | **0.6720** | 0.6701 | AA |
| Netscience LCC | 0.8511 | 0.8507 | 0.8537 | **0.8538** | RA |

**Caution:** Confirm that standalone AA/RA baselines and optimized models use an equivalent scaling and evaluation pipeline. Test identity weights [0,0,1,0] and [0,0,0,1] through the same pipeline before making a definitive improvement claim.

## Interpretation

ABC achieved the highest reported mean test ROC-AUC on both networks. Increasing the evaluation budget from 50 to 20,000 did not produce a meaningful improvement in the reported scores. On Karate Club, the means were invariant across budgets; on Netscience, ABC showed only very small numerical variation.

A possible explanation is that CN, Jaccard, AA, and RA all use shared-neighbour structural information. Feature redundancy is a hypothesis, not a demonstrated cause unless supported by a feature-correlation analysis.

## Limitations to state

1. The outer graph split is fixed; optimizer seeds do not measure split-to-split variation.
2. Negative examples are uniformly sampled and may be relatively easy.
3. Evaluation count is used as the budget measure; wall-clock runtime was not collected.
4. No statistical significance claim across independent graph splits is supported by these runs.