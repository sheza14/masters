import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from data_loader import split_graph
from features import calculate_features


def calculate_weighted_scores(features, weights):
    """
    Calculate weighted score using:
    CN, Jaccard, AA, RA
    """

    weights = np.array(weights)

    scores = (
        weights[0] * features["cn"]
        + weights[1] * features["jaccard"]
        + weights[2] * features["aa"]
        + weights[3] * features["ra"]
    )

    return scores


def calculate_auc(features, labels, weights):

    scores = calculate_weighted_scores(
        features,
        weights
    )

    return roc_auc_score(labels, scores)


if __name__ == "__main__":

    # Load Karate Club graph
    graph = nx.karate_club_graph()

    # Same fixed split used throughout the experiment
    original, training, positives, negatives = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    # Test edges
    test_edges = positives + negatives

    # Labels
    labels = [1] * len(positives) + [0] * len(negatives)

    # Calculate features using training graph
    features = calculate_features(
        training,
        test_edges
    )

    # Equal weights
    weights = [1.0, 1.0, 1.0, 1.0]

    # Calculate AUC
    auc = calculate_auc(
        features,
        labels,
        weights
    )

    print("Weighted Link Prediction Test")
    print("-----------------------------")
    print("Weights:", weights)
    print(f"ROC-AUC: {auc:.4f}")

    print("\nClassical baseline:")
    print("AA = 0.6720")