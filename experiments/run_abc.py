import os
import sys
import csv

import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add src folder to Python path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from artificial_bee_colony import (
    ArtificialBeeColony,
    evaluate_test_auc
)
from fitness import FitnessFunction
from features import calculate_features
from data_loader import (
    split_graph,
    split_validation_graph,
    load_netscience
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

DATASETS = [
    "Karate Club",
    "Netscience"
]

BUDGETS = [50, 500, 5000, 20000]

SEEDS = list(range(10))

TEST_RATIO = 0.30
VALIDATION_RATIO = 0.30

TEST_SPLIT_SEED = 42
VALIDATION_SPLIT_SEED = 123

COLONY_SIZE = 10
LIMIT = 5


# ============================================================
# OUTPUT FILE
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "abc_results.csv"
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

        return load_netscience(path)

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )


# ============================================================
# PREPARE EXPERIMENT DATA
# ============================================================

def prepare_experiment_data(graph):

    # Final 30% test split.
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

    # Internal 30% validation split.
    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    ) = split_validation_graph(
        training_graph=training_graph,
        original_graph=original_graph,
        excluded_negative_edges=negative_test_edges,
        validation_ratio=VALIDATION_RATIO,
        seed=VALIDATION_SPLIT_SEED
    )

    # Prepare validation data.
    validation_edges = (
        positive_validation_edges
        + negative_validation_edges
    )

    validation_labels = (
        [1] * len(positive_validation_edges)
        + [0] * len(negative_validation_edges)
    )

    validation_features_raw = calculate_features(
        validation_training_graph,
        validation_edges
    )

    # Prepare final test data.
    test_edges = (
        positive_test_edges
        + negative_test_edges
    )

    test_labels = (
        [1] * len(positive_test_edges)
        + [0] * len(negative_test_edges)
    )

    test_features_raw = calculate_features(
        training_graph,
        test_edges
    )

    # Fit scaler only on validation features.
    scaler = MinMaxScaler()

    validation_features_scaled = pd.DataFrame(
        scaler.fit_transform(validation_features_raw),
        columns=validation_features_raw.columns
    )

    # Apply the same scaler to test features.
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features_raw),
        columns=test_features_raw.columns
    )

    return (
        validation_features_scaled,
        validation_labels,
        test_features_scaled,
        test_labels
    )
# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_one_experiment(
    validation_features,
    validation_labels,
    test_features,
    test_labels,
    budget,
    seed
):

    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    abc = ArtificialBeeColony(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        colony_size=COLONY_SIZE,
        limit=LIMIT
    )

    best_weights, best_validation_auc, evaluations = abc.optimize()

    if evaluations != budget:
        raise RuntimeError(
            f"Expected {budget} evaluations, got {evaluations}."
        )

    # Evaluate the selected weights on the untouched test set.
    final_test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    return {
        "dataset": None,
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
# SAVE RESULT
# ============================================================

def save_result(result):

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

    file_exists = (
        os.path.exists(RESULT_FILE)
        and os.path.getsize(RESULT_FILE) > 0
    )

    with open(
        RESULT_FILE,
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ARTIFICIAL BEE COLONY EXPERIMENT")
    print("=" * 70)

    all_data = {}

    # Prepare fixed data once for each dataset.
    for dataset_name in DATASETS:

        print(f"\nPreparing dataset: {dataset_name}")

        graph = load_dataset(dataset_name)

        all_data[dataset_name] = prepare_experiment_data(graph)

        (
            validation_features,
            validation_labels,
            test_features,
            test_labels
        ) = all_data[dataset_name]

        print("Validation samples:", len(validation_labels))
        print("Test samples:", len(test_labels))

    total_runs = (
        len(DATASETS)
        * len(BUDGETS)
        * len(SEEDS)
    )

    completed_runs = 0

    print(f"\nTotal planned runs: {total_runs}")

    # Run all experiments.
    for dataset_name in DATASETS:

        (
            validation_features,
            validation_labels,
            test_features,
            test_labels
        ) = all_data[dataset_name]

        for budget in BUDGETS:

            for seed in SEEDS:

                print("\n" + "-" * 70)
                print(f"Dataset: {dataset_name}")
                print(f"Budget: {budget}")
                print(f"Seed: {seed}")

                result = run_one_experiment(
                    validation_features,
                    validation_labels,
                    test_features,
                    test_labels,
                    budget,
                    seed
                )

                result["dataset"] = dataset_name

                save_result(result)

                completed_runs += 1

                print(
                    f"Validation AUC: "
                    f"{result['best_validation_auc']:.4f}"
                )

                print(
                    f"Test AUC: "
                    f"{result['final_test_auc']:.4f}"
                )

                print(f"Evaluations: {result['evaluations']}")
                print(f"Completed: {completed_runs}/{total_runs}")

    print("\n" + "=" * 70)
    print("ALL ABC EXPERIMENTS COMPLETED")
    print("=" * 70)

    print(f"Total completed runs: {completed_runs}")
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()