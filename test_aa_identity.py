
import networkx as nx
from sklearn.metrics import roc_auc_score

from src.data_loader import split_graph, split_validation_graph
from src.features import calculate_features

G = nx.karate_club_graph()

_, train_graph, _, _ = split_graph(
    G, test_ratio=0.30, seed=42
)

for ratio in [0.10, 0.20, 0.30, 0.40]:

    val_graph, pos_val, neg_val = split_validation_graph(
        train_graph,
        validation_ratio=ratio,
        seed=123
    )

    edges = pos_val + neg_val
    labels = [1] * len(pos_val) + [0] * len(neg_val

)

    features = calculate_features(val_graph, edges)

    auc = roc_auc_score(labels, features["aa"])

    print(
        f"Validation ratio: {ratio:.2f} | "
        f"AA AUC: {auc:.4f} | "
        f"Graph edges: {val_graph.number_of_edges()}"
    )
