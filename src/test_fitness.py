import networkx as nx

from data_loader import split_graph
from features import calculate_features
from fitness import FitnessFunction


# Load Karate Club
graph = nx.karate_club_graph()

# Use the same fixed split
original, training, positives, negatives = split_graph(
    graph,
    test_ratio=0.30,
    seed=42
)

# Test edges
test_edges = positives + negatives

# Labels
labels = (
    [1] * len(positives)
    + [0] * len(negatives)
)

# Calculate features using training graph
features = calculate_features(
    training,
    test_edges
)

# Create fitness function
fitness = FitnessFunction(
    features,
    labels,
    budget=10
)

# Test weights
weights = [1.0, 1.0, 1.0, 1.0]

# Evaluate once
auc = fitness.evaluate(weights)

print("Fitness Function Test")
print("---------------------")
print("Weights:", weights)
print(f"ROC-AUC: {auc:.4f}")
print("Evaluations used:", fitness.evaluations)
print("Budget:", fitness.budget)