
import os
import sys
import csv

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)


from data_loader import (
    split_graph,
    split_validation_graph,
    load_netscience
)

from features import calculate_features
from fitness import FitnessFunction
from pso import PSO, evaluate_test_auc


# ============================================================
# FINAL EXPERIMENT SETTINGS
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
VALIDATION_RATIO = 0.30

SWARM_SIZE = 10

FEATURE_COLUMNS = [
    "cn",
    "jaccard",
    "aa",
    "ra"
]


# ============================================================
# OUTPUT FILE
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "pso_results.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def get_dataset(name):

    if name == "Karate Club":
        return nx.karate_club_graph()

    elif name == "Netscience":

        netscience_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "netscience.gml"
        )

        return load_netscience(netscience_path)

    else:
        raise ValueError(
            f"Unknown dataset: {name}"
        )


# ============================================================
# PREPARE EXPERIMENT DATA
# ============================================================


def prepare_experiment_data(graph):

    # OUTER TEST SPLIT: 30%
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

    # INNER VALIDATION SPLIT: 30%
    (
        inner_training_graph,
        validation_positive,
        validation_negative
    ) = split_validation_graph(
        training_graph=training_graph,
        original_graph=original_graph,
        excluded_negative_edges=negative_test_edges,
        validation_ratio=VALIDATION_RATIO,
        seed=VALIDATION_SPLIT_SEED
    )

    # VALIDATION DATA
    validation_edges = (
        validation_positive + validation_negative
    )

    validation_labels = (
        [1] * len(validation_positive)
        + [0] * len(validation_negative)
    )

    validation_features_raw = calculate_features(
        inner_training_graph,
        validation_edges
    )

    # FINAL TEST DATA
    test_edges = (
        positive_test_edges + negative_test_edges
    )

    test_labels = (
        [1] * len(positive_test_edges)
        + [0] * len(negative_test_edges)
    )

    test_features_raw = calculate_features(
        training_graph,
        test_edges
    )

    # FEATURE SCALING
    scaler = MinMaxScaler()

    validation_features_scaled = pd.DataFrame(
        scaler.fit_transform(
            validation_features_raw[FEATURE_COLUMNS]
        ),
        columns=FEATURE_COLUMNS
    )

    test_features_scaled = pd.DataFrame(
        scaler.transform(
            test_features_raw[FEATURE_COLUMNS]
        ),
        columns=FEATURE_COLUMNS
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

def run_experiment(
    dataset_name,
    budget,
    seed,
    validation_features,
    validation_labels,
    test_features,
    test_labels
):

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Budget: {budget}")
    print(f"Seed: {seed}")
    print("=" * 60)

    # Fresh fitness function for every run
    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget
    )

    pso = PSO(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        swarm_size=SWARM_SIZE
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = pso.optimize()

    # --------------------------------------------------------
    # VERIFY BUDGET
    # --------------------------------------------------------

    if evaluations != budget:
        raise RuntimeError(
            f"Expected {budget} evaluations, "
            f"but used {evaluations}."
        )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    final_test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    print(f"Best validation AUC: {best_validation_auc:.4f}")
    print(f"Final test AUC: {final_test_auc:.4f}")
    print(f"Evaluations used: {evaluations}")
    print(f"Best weights: {best_weights}")

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
        os.path.exists(RESULTS_FILE)
        and os.path.getsize(RESULTS_FILE) > 0
    )

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

def main():

    print("\n" + "=" * 60)
    print("PSO LINK PREDICTION EXPERIMENT")
    print("=" * 60)

    total_runs = (
        len(DATASETS)
        * len(BUDGETS)
        * len(SEEDS)
    )

    completed_runs = 0

    for dataset_name in DATASETS:

        print(f"\nPreparing dataset: {dataset_name}")

        graph = get_dataset(dataset_name)

        (
            validation_features,
            validation_labels,
            test_features,
            test_labels
        ) = prepare_experiment_data(graph)

        for budget in BUDGETS:

            for seed in SEEDS:

                result = run_experiment(
                    dataset_name,
                    budget,
                    seed,
                    validation_features,
                    validation_labels,
                    test_features,
                    test_labels
                )

                save_result(result)

                completed_runs += 1

                print(
                    f"\nProgress: {completed_runs}/{total_runs}"
                )

    print("\n" + "=" * 60)
    print("ALL PSO EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(f"Results saved to:\n{RESULTS_FILE}")


if __name__ == "__main__":
    main()
