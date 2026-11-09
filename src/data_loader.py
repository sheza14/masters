
import networkx as nx
import random


def load_karate():
    """Load the Karate Club dataset."""
    return nx.karate_club_graph()


def load_netscience(path="data/netscience.gml"):
    """Load Netscience and return its largest connected component."""
    graph = nx.read_gml(path)

    # Convert directed graph to undirected if necessary
    if graph.is_directed():
        graph = graph.to_undirected()

    largest_component = max(
        nx.connected_components(graph),
        key=len
    )

    graph = graph.subgraph(largest_component).copy()

    return graph


def get_all_non_edges(graph):
    """Return all node pairs that are not connected by an edge."""
    return list(nx.non_edges(graph))


def split_graph(graph, test_ratio=0.30, seed=42):
    """
    Outer split:
    70% edges for training and 30% for final testing.

    Negative test edges are sampled from non-edges
    of the original graph.
    """
    rng = random.Random(seed)

    original_graph = graph.copy()

    edges = list(original_graph.edges())
    rng.shuffle(edges)

    test_size = int(len(edges) * test_ratio)

    positive_test_edges = edges[:test_size]

    training_graph = original_graph.copy()
    training_graph.remove_edges_from(positive_test_edges)

    all_negative_edges = get_all_non_edges(original_graph)

    if len(all_negative_edges) < test_size:
        raise ValueError("Not enough negative edges for testing.")

    negative_test_edges = rng.sample(
        all_negative_edges,
        test_size
    )

    return (
        original_graph,
        training_graph,
        positive_test_edges,
        negative_test_edges
    )


def split_validation_graph(
    training_graph,
    original_graph,
    excluded_negative_edges=None,
    validation_ratio=0.30,
    seed=123
):
    """
    Inner split:
    Split the outer training graph into:
    70% inner training and 30% validation.

    Validation negatives are sampled from original graph
    non-edges, excluding final test negatives.
    """
    rng = random.Random(seed)

    excluded_negative_edges = set(
        excluded_negative_edges or []
    )

    edges = list(training_graph.edges())
    rng.shuffle(edges)

    validation_size = int(len(edges) * validation_ratio)

    positive_validation_edges = edges[:validation_size]

    validation_training_graph = training_graph.copy()
    validation_training_graph.remove_edges_from(
        positive_validation_edges
    )

    all_negative_edges = get_all_non_edges(original_graph)

    available_negative_edges = [
        edge for edge in all_negative_edges
        if edge not in excluded_negative_edges
        and (edge[1], edge[0]) not in excluded_negative_edges
    ]

    if len(available_negative_edges) < validation_size:
        raise ValueError(
            "Not enough negative edges for validation."
        )

    negative_validation_edges = rng.sample(
        available_negative_edges,
        validation_size
    )

    return (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    )


if __name__ == "__main__":

    print("Testing data loader...\n")

    datasets = {
        "Karate Club": load_karate(),
        "Netscience": load_netscience()
    }

    for name, graph in datasets.items():

        (
            original_graph,
            training_graph,
            positive_test,
            negative_test
        ) = split_graph(
            graph,
            test_ratio=0.30,
            seed=42
        )

        (
            validation_training_graph,
            positive_validation,
            negative_validation
        ) = split_validation_graph(
            training_graph=training_graph,
            original_graph=original_graph,
            excluded_negative_edges=negative_test,
            validation_ratio=0.30,
            seed=123
        )

        print(f"Dataset: {name}")
        print(f"Original nodes: {original_graph.number_of_nodes()}")
        print(f"Original edges: {original_graph.number_of_edges()}")
        print(f"Outer training edges: {training_graph.number_of_edges()}")
        print(f"Final test positive edges: {len(positive_test)}")
        print(f"Final test negative edges: {len(negative_test)}")
        print(
            f"Inner training edges: "
            f"{validation_training_graph.number_of_edges()}"
        )
        print(f"Validation positive edges: {len(positive_validation)}")
        print(f"Validation negative edges: {len(negative_validation)}")
        print("-" * 40)
