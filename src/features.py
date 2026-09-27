import networkx as nx
import pandas as pd


def calculate_features(graph, edges):
    """
    Calculate four link-prediction features:
    1. Common Neighbors (CN)
    2. Jaccard Coefficient
    3. Adamic-Adar (AA)
    4. Resource Allocation (RA)

    Features are calculated using the training graph only.
    """

    # Common Neighbors
    cn_scores = []

    for u, v in edges:
        common = list(nx.common_neighbors(graph, u, v))
        cn_scores.append(len(common))

    # Jaccard Coefficient
    jaccard_scores = []

    for u, v, score in nx.jaccard_coefficient(graph, edges):
        jaccard_scores.append(score)

    # Adamic-Adar
    aa_scores = []

    for u, v, score in nx.adamic_adar_index(graph, edges):
        aa_scores.append(score)

    # Resource Allocation
    ra_scores = []

    for u, v, score in nx.resource_allocation_index(graph, edges):
        ra_scores.append(score)

    # Put everything into a DataFrame
    features = pd.DataFrame({
        "cn": cn_scores,
        "jaccard": jaccard_scores,
        "aa": aa_scores,
        "ra": ra_scores
    })

    return features


if __name__ == "__main__":

    # Load Karate Club graph
    graph = nx.karate_club_graph()

    # Import our splitting function
    from data_loader import split_graph

    original, training, positives, negatives = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    # Combine positive and negative test edges
    test_edges = positives + negatives

    # Calculate features using TRAINING graph
    features = calculate_features(
        training,
        test_edges
    )

    print("\nFeature table:")
    print(features)

    print("\nFeature shape:")
    print(features.shape)