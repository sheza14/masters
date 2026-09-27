import sys
import os
import csv

import networkx as nx

# Add src folder to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_PATH)

from data_loader import (
    split_graph,
    split_validation_graph
)

from features import calculate_features

from fitness import FitnessFunction

from firefly import (
    Firefly,
    evaluate_test_auc
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

DATASETS = [
    "Karate Club",
    "Netscience"
]

BUDGETS = [
    50,
    500,
    5000,
    20000
]

SEEDS = list(range(10))

TEST_SPLIT_SEED = 42
VALIDATION_SPLIT_SEED = 123

TEST_RATIO = 0.30
VALIDATION_RATIO = 0.10

POPULATION_SIZE = 10

ALPHA = 0.2
BETA0 = 1.0
GAMMA = 1.0


# ============================================================
# OUTPUT FILE
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "firefly_results.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name):

    if dataset_name == "Karate Club":

        return nx.karate_club_graph()

    elif dataset_name == "Netscience":

        path = os.path.join(
            PROJECT_ROOT,
            "data",
            "netscience.gml"
        )

        graph = nx.read_gml(path)

        graph = graph.to_undirected()

        largest_component_nodes = max(
            nx.connected_components(graph),
            key=len
        )

        graph = graph.subgraph(
            largest_component_nodes
        ).copy()

        return graph

    else:

        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )


# ============================================================
# PREPARE EXPERIMENT DATA
# ============================================================

def prepare_experiment_data(graph):

    # --------------------------------------------------------
    # FINAL 30% TEST SPLIT
    # --------------------------------------------------------

    (
        original_graph,
        training_graph,
        positive_test_edges,
        negative_test_edges
    ) = split_graph(
        graph,
        test_ratio=TEST_RATIO,
        seed=TEST_SPLIT_SEED
    )

    # --------------------------------------------------------
    # INTERNAL VALIDATION SPLIT
    # --------------------------------------------------------

    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    ) = split_validation_graph(
        training_graph,
        validation_ratio=VALIDATION_RATIO,
        seed=VALIDATION_SPLIT_SEED
    )

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    validation_edges = (
        positive_validation_edges
        + negative_validation_edges
    )

    validation_labels = (
        [1] * len(positive_validation_edges)
        + [0] * len(negative_validation_edges)
    )

    validation_features = calculate_features(
        validation_training_graph,
        validation_edges
    )

    # --------------------------------------------------------
    # FINAL TEST DATA
    # --------------------------------------------------------

    test_edges = (
        positive_test_edges
        + negative_test_edges
    )

    test_labels = (
        [1] * len(positive_test_edges)
        + [0] * len(negative_test_edges)
    )

    test_features = calculate_features(
        training_graph,
        test_edges
    )

    return (
        validation_features,
        validation_labels,
        test_features,
        test_labels
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_one_experiment(
    dataset_name,
    budget,
    seed,
    validation_features,
    validation_labels,
    test_features,
    test_labels
):

    print("\n" + "=" * 60)

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Budget: {budget}"
    )

    print(
        f"Seed: {seed}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # FITNESS FUNCTION
    # --------------------------------------------------------

    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    # --------------------------------------------------------
    # FIREFLY
    # --------------------------------------------------------

    firefly = Firefly(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        population_size=POPULATION_SIZE,
        alpha=ALPHA,
        beta0=BETA0,
        gamma=GAMMA
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = firefly.optimize()

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    final_test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    print(
        f"Best validation AUC: "
        f"{best_validation_auc:.4f}"
    )

    print(
        f"Final test AUC: "
        f"{final_test_auc:.4f}"
    )

    print(
        f"Evaluations used: "
        f"{evaluations}"
    )

    return {
        "dataset": dataset_name,
        "budget": budget,
        "seed": seed,
        "best_validation_auc": best_validation_auc,
        "final_test_auc": final_test_auc,
        "evaluations": evaluations,
        "weight_cn": best_weights[0],
        "weight_jaccard": best_weights[1],
        "weight_aa": best_weights[2],
        "weight_ra": best_weights[3]
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_result(result):

    file_exists = os.path.exists(
        RESULTS_FILE
    )

    fieldnames = [
        "dataset",
        "budget",
        "seed",
        "best_validation_auc",
        "final_test_auc",
        "evaluations",
        "weight_cn",
        "weight_jaccard",
        "weight_aa",
        "weight_ra"
    ]

    with open(
        RESULTS_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(result)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    total_experiments = (
        len(DATASETS)
        * len(BUDGETS)
        * len(SEEDS)
    )

    print("=" * 60)
    print("FIREFLY BUDGET CURVE EXPERIMENT")
    print("=" * 60)

    print(
        f"Total experiments: "
        f"{total_experiments}"
    )

    print(
        f"Datasets: {DATASETS}"
    )

    print(
        f"Budgets: {BUDGETS}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print("=" * 60)

    run_number = 0

    for dataset_name in DATASETS:

        print("\n")
        print("#" * 60)

        print(
            f"LOADING DATASET: "
            f"{dataset_name}"
        )

        print("#" * 60)

        graph = load_dataset(
            dataset_name
        )

        # Prepare the fixed data once.
        (
            validation_features,
            validation_labels,
            test_features,
            test_labels
        ) = prepare_experiment_data(
            graph
        )

        for budget in BUDGETS:

            for seed in SEEDS:

                run_number += 1

                print(
                    f"\nRUN "
                    f"{run_number}/"
                    f"{total_experiments}"
                )

                result = run_one_experiment(
                    dataset_name,
                    budget,
                    seed,
                    validation_features,
                    validation_labels,
                    test_features,
                    test_labels
                )

                # Save immediately after each run
                save_result(result)


    print("\n")
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(
        f"Total completed runs: "
        f"{total_experiments}"
    )

    print("\nResults saved to:")

    print(
        RESULTS_FILE
    )