
import networkx as nx
from sklearn.metrics import roc_auc_score

from src.data_loader import (
    load_karate,
    load_netscience,
    split_graph
)
from src.features import calculate_features


def evaluate_dataset(name, graph):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Create the fixed outer train/test split
    (
        original_graph,
        training_graph,
        positive_test_edges,
        negative_test_edges
    ) = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    # Combine positive and negative test pairs
    test_edges = positive_test_edges + negative_test_edges

    # Create labels
    labels = (
        [1] * len(positive_test_edges)
        + [0] * len(negative_test_edges)
    )

    # Calculate features using only the outer training graph
    features = calculate_features(
        training_graph,
        test_edges
    )

    # Calculate AUC for each classical feature
    auc_scores = {}

    for feature in ["cn", "jaccard", "aa", "ra"]:

        auc = roc_auc_score(
            labels,
            features[feature]
        )

        auc_scores[feature] = auc

        print(f"{feature.upper():10s}: {auc:.4f}")

    # Select the best individual classical heuristic
    best_feature = max(
        auc_scores,
        key=auc_scores.get
    )

    best_auc = auc_scores[best_feature]

    print("\nClassical Baseline")
    print("------------------")
    print(f"Best feature: {best_feature.upper()}")
    print(f"Baseline AUC: {best_auc:.4f}")

    return auc_scores


if __name__ == "__main__":

    # Karate Club
    karate = load_karate()

    evaluate_dataset("Karate Club", karate)

    # Netscience
    netscience = load_netscience()

    evaluate_dataset("Netscience", netscience)
