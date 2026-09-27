import networkx as nx
import random


def load_karate():
    """Load the Zachary Karate Club graph."""
    return nx.karate_club_graph()


def load_netscience(path="data/netscience.gml"):
    """
    Load the Netscience co-authorship network
    and return its largest connected component.
    """

    graph = nx.read_gml(path)

    # Convert to undirected graph
    graph = graph.to_undirected()

    # Find largest connected component
    largest_component_nodes = max(
        nx.connected_components(graph),
        key=len
    )

    # Create independent graph
    largest_component = graph.subgraph(
        largest_component_nodes
    ).copy()

    return largest_component


def split_graph(graph, test_ratio=0.30, seed=42):
    """
    Create the fixed final train/test split.

    30% of original edges are held out as
    the final positive test edges.

    Negative test edges are node pairs that
    were NOT edges in the original graph.

    This split is fixed and reused throughout
    the experiment.
    """

    random.seed(seed)

    # Keep original graph
    original_graph = graph.copy()

    # Get all edges
    edges = list(original_graph.edges())

    # Number of positive test edges
    num_test_edges = int(
        len(edges) * test_ratio
    )

    # Select positive test edges
    positive_test_edges = random.sample(
        edges,
        num_test_edges
    )

    # Create training graph
    training_graph = original_graph.copy()

    # Remove test edges
    training_graph.remove_edges_from(
        positive_test_edges
    )

    # Generate possible negative edges
    possible_negative_edges = []

    nodes = list(original_graph.nodes())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):

            u = nodes[i]
            v = nodes[j]

            # Must not be an edge in ORIGINAL graph
            if not original_graph.has_edge(u, v):
                possible_negative_edges.append(
                    (u, v)
                )

    # Same number of negatives as positives
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


def split_validation_graph(
    training_graph,
    validation_ratio=0.10,
    seed=123
):
    """
    Create a validation split INSIDE the 70% training graph.

    The final 30% test set is never used here.

    Returns:

    validation_training_graph
    positive_validation_edges
    negative_validation_edges
    """

    random.seed(seed)

    # Copy the 70% training graph
    graph = training_graph.copy()

    # Get training edges
    edges = list(graph.edges())

    # Number of validation positives
    num_validation_edges = int(
        len(edges) * validation_ratio
    )

    # Select validation positive edges
    positive_validation_edges = random.sample(
        edges,
        num_validation_edges
    )

    # Remove validation positives
    validation_training_graph = graph.copy()

    validation_training_graph.remove_edges_from(
        positive_validation_edges
    )

    # Generate possible negative validation pairs
    possible_negative_edges = []

    nodes = list(graph.nodes())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):

            u = nodes[i]
            v = nodes[j]

            # Negative must not be an edge in
            # the graph BEFORE validation removal
            if not graph.has_edge(u, v):
                possible_negative_edges.append(
                    (u, v)
                )

    # Same number of negative and positive examples
    negative_validation_edges = random.sample(
        possible_negative_edges,
        num_validation_edges
    )

    return (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    )


if __name__ == "__main__":

    # ==================================================
    # 1. Karate Club
    # ==================================================

    print("=" * 50)
    print("KARATE CLUB")
    print("=" * 50)

    karate = load_karate()

    print("Original graph")
    print("Nodes:", karate.number_of_nodes())
    print("Edges:", karate.number_of_edges())

    (
        original,
        training,
        positives,
        negatives
    ) = split_graph(
        karate,
        test_ratio=0.30,
        seed=42
    )

    print("\nTraining graph")
    print("Nodes:", training.number_of_nodes())
    print("Edges:", training.number_of_edges())

    print("\nPositive test edges:", len(positives))
    print("Negative test edges:", len(negatives))

    # Validation split
    (
        validation_training,
        validation_positives,
        validation_negatives
    ) = split_validation_graph(
        training,
        validation_ratio=0.10,
        seed=123
    )

    print("\nValidation split")
    print(
        "Validation training edges:",
        validation_training.number_of_edges()
    )

    print(
        "Positive validation edges:",
        len(validation_positives)
    )

    print(
        "Negative validation edges:",
        len(validation_negatives)
    )


    # ==================================================
    # 2. Netscience
    # ==================================================

    print("\n" + "=" * 50)
    print("NETSCIENCE")
    print("=" * 50)

    netscience = load_netscience()

    print("Largest connected component")
    print("Nodes:", netscience.number_of_nodes())
    print("Edges:", netscience.number_of_edges())

    (
        original_ns,
        training_ns,
        positives_ns,
        negatives_ns
    ) = split_graph(
        netscience,
        test_ratio=0.30,
        seed=42
    )

    print("\nTraining graph")
    print("Nodes:", training_ns.number_of_nodes())
    print("Edges:", training_ns.number_of_edges())

    print("\nPositive test edges:", len(positives_ns))
    print("Negative test edges:", len(negatives_ns))

    # Validation split
    (
        validation_training_ns,
        validation_positives_ns,
        validation_negatives_ns
    ) = split_validation_graph(
        training_ns,
        validation_ratio=0.10,
        seed=123
    )

    print("\nValidation split")
    print(
        "Validation training edges:",
        validation_training_ns.number_of_edges()
    )

    print(
        "Positive validation edges:",
        len(validation_positives_ns)
    )

    print(
        "Negative validation edges:",
        len(validation_negatives_ns)
    )