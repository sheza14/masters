import networkx as nx
import pandas as pd


def calculate_features(graph, edges):
    cn_scores = []

    for u, v in edges:
        common = list(nx.common_neighbors(graph, u, v))
        cn_scores.append(len(common))

    jaccard_scores = []

    for u, v, score in nx.jaccard_coefficient(graph, edges):
        jaccard_scores.append(score)

    aa_scores = []

    for u, v, score in nx.adamic_adar_index(graph, edges):
        aa_scores.append(score)

    ra_scores = []

    for u, v, score in nx.resource_allocation_index(graph, edges):
        ra_scores.append(score)

    features = pd.DataFrame({
        "cn": cn_scores,
        "jaccard": jaccard_scores,
        "aa": aa_scores,
        "ra": ra_scores
    })

    return features