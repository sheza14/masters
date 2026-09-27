import networkx as nx
from sklearn.metrics import roc_auc_score

from data_loader import load_karate, load_netscience, split_graph
from features import calculate_features


def evaluate_dataset(name, graph):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    # Create fixed train/test split
    original, training, positives, negatives = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    # Combine positive and negative test pairs
    test_edges = positives + negatives

    # Labels
    labels = (
        [1] * len(positives)
        + [0] * len(negatives)
    )

    # Calculate four features using TRAINING graph
    features = calculate_features(
        training,
        test_edges
    )

    # Calculate AUC for every feature
    auc_scores = {}

    for feature in features.columns:

        auc = roc_auc_score(
            labels,
            features[feature]
        )

        auc_scores[feature] = auc

        print(f"{feature.upper():10s}: {auc:.4f}")

    # Find best classical feature
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

    evaluate_dataset(
        "Karate Club",
        karate
    )

    # Netscience
    netscience = load_netscience()

    evaluate_dataset(
        "Netscience",
        netscience
    )