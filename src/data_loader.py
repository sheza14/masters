import networkx as nx
import random


def load_karate():
    """Load the Karate Club graph."""
    return nx.karate_club_graph()


def split_graph(graph, test_ratio=0.30, seed=42):
    """
    Split graph edges into:
    - training graph
    - positive test edges
    - negative test edges
    """

    random.seed(seed)

    # Make a copy of the original graph
    original_graph = graph.copy()

    # List all original edges
    edges = list(original_graph.edges())

    # Number of positive test edges
    num_test_edges = int(len(edges) * test_ratio)

    # Randomly select positive test edges
    positive_test_edges = random.sample(edges, num_test_edges)

    # Create training graph
    training_graph = original_graph.copy()

    # Remove test edges from training graph
    training_graph.remove_edges_from(positive_test_edges)

    # Generate possible negative edges
    possible_negative_edges = []

    nodes = list(original_graph.nodes())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u = nodes[i]
            v = nodes[j]

            # Only choose pairs that were NOT edges
            # in the original graph
            if not original_graph.has_edge(u, v):
                possible_negative_edges.append((u, v))

    # Select same number of negatives as positives
    negative_test_edges = random.sample(
        possible_negative_edges,
        num_test_edges
    )

    return (
        original_graph,
        training_graph,
        positive_test_edges,
        negative_test_edges
    )


if __name__ == "__main__":

    graph = load_karate()

    original, training, positives, negatives = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    print("Original graph")
    print("Nodes:", original.number_of_nodes())
    print("Edges:", original.number_of_edges())

    print("\nTraining graph")
    print("Nodes:", training.number_of_nodes())
    print("Edges:", training.number_of_edges())

    print("\nPositive test edges:", len(positives))
    print("Negative test edges:", len(negatives))

    print("\nExample positive edges:")
    print(positives[:5])

    print("\nExample negative edges:")
    print(negatives[:5])