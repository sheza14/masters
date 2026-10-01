import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))

from data_loader import load_netscience, split_graph, split_validation_graph
from features import calculate_features
from fitness import FitnessFunction


def prepare_netscience():
    """
    Uses the same data-splitting protocol as the main experiments.
    """

    graph = load_netscience()

    # Fixed 30% final test split
    (
        original_graph,
        training_graph,
        positive_test_edges,
        negative_test_edges,
    ) = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    # Fixed validation split inside the 70% training graph
    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges,
    ) = split_validation_graph(
        training_graph,
        validation_ratio=0.10,
        seed=123
    )

    validation_edges = (
        positive_validation_edges +
        negative_validation_edges
    )

    validation_labels = (
        [1] * len(positive_validation_edges) +
        [0] * len(negative_validation_edges)
    )

    validation_features = calculate_features(
        validation_training_graph,
        validation_edges
    )

    return validation_features, validation_labels


def correlation_test(features):
    print("\n" + "=" * 70)
    print("1. FEATURE CORRELATION")
    print("=" * 70)

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nCorrelation matrix:")
    print(features.corr())

    print("\n")


def random_weight_test(features, labels, num_tests=100):
    print("=" * 70)
    print("2. RANDOM WEIGHT FITNESS LANDSCAPE")
    print("=" * 70)

    fitness = FitnessFunction(
        features=features,
        labels=labels,
        budget=num_tests
    )

    rng = np.random.default_rng(42)

    results = []

    for i in range(num_tests):
        weights = rng.uniform(0.0, 1.0, size=4)

        auc = fitness.evaluate(weights)

        results.append({
            "test": i + 1,
            "cn": weights[0],
            "jaccard": weights[1],
            "aa": weights[2],
            "ra": weights[3],
            "auc": auc
        })

    results_df = pd.DataFrame(results)

    print("\nFirst 20 random weight evaluations:")
    print(results_df.head(20).to_string(index=False))

    print("\n" + "-" * 70)

    unique_auc = results_df["auc"].nunique()
    min_auc = results_df["auc"].min()
    max_auc = results_df["auc"].max()

    print(f"Number of random weight combinations : {num_tests}")
    print(f"Number of unique AUC values          : {unique_auc}")
    print(f"Minimum AUC                           : {min_auc:.6f}")
    print(f"Maximum AUC                           : {max_auc:.6f}")
    print(f"AUC range                             : {max_auc - min_auc:.6f}")

    print("\nAUC frequency:")
    print(results_df["auc"].value_counts().sort_index())

    return results_df


def feature_range_test(features):
    print("\n" + "=" * 70)
    print("3. FEATURE RANGES")
    print("=" * 70)

    summary = pd.DataFrame({
        "min": features.min(),
        "max": features.max(),
        "range": features.max() - features.min(),
        "mean": features.mean()
    })

    print(summary.to_string())


def main():
    print("\n")
    print("=" * 70)
    print("NETSCIENCE FITNESS LANDSCAPE DIAGNOSTIC")
    print("=" * 70)

    features, labels = prepare_netscience()

    print(f"\nNumber of validation samples: {len(labels)}")
    print(f"Number of features: {features.shape[1]}")

    feature_range_test(features)
    correlation_test(features)

    results = random_weight_test(
        features,
        labels,
        num_tests=100
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 70)

    unique_auc = results["auc"].nunique()
    auc_range = results["auc"].max() - results["auc"].min()

    if unique_auc == 1:
        print(
            "\nWARNING: All 100 random weight combinations produced "
            "the exact same AUC."
        )
        print(
            "This indicates an extremely flat ranking/fitness landscape."
        )

    elif auc_range < 0.001:
        print(
            "\nThe landscape is very flat."
        )
        print(
            f"Only {unique_auc} unique AUC values were observed "
            f"and the total AUC range was {auc_range:.6f}."
        )

    else:
        print(
            "\nThe fitness landscape is not completely flat."
        )
        print(
            f"{unique_auc} unique AUC values were observed."
        )
        print(
            f"AUC range = {auc_range:.6f}"
        )

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()