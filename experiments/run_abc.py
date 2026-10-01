import os
import sys
import csv

import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


# ============================================================
# ADD SRC FOLDER TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_PATH = os.path.join(
    PROJECT_ROOT,
    "src"
)

sys.path.insert(0, SRC_PATH)


from artificial_bee_colony import ArtificialBeeColony
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

BUDGETS = [
    50,
    500,
    5000,
    20000
]

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

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

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

    # --------------------------------------------------------
    # FINAL 30% TEST SPLIT
    # --------------------------------------------------------

    (
        _,
        training_graph,
        positive_test_edges,
        negative_test_edges
    ) = split_graph(
        graph,
        test_ratio=TEST_RATIO,
        seed=TEST_SPLIT_SEED
    )

    # --------------------------------------------------------
    # INTERNAL 30% VALIDATION SPLIT
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

    validation_features_raw = calculate_features(
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

    test_features_raw = calculate_features(
        training_graph,
        test_edges
    )

    # --------------------------------------------------------
    # FEATURE SCALING
    # --------------------------------------------------------
    # Fit ONLY on validation features.
    # Use the SAME scaler for test features.
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    validation_features_scaled = pd.DataFrame(
        scaler.fit_transform(
            validation_features_raw
        ),
        columns=validation_features_raw.columns
    )

    test_features_scaled = pd.DataFrame(
        scaler.transform(
            test_features_raw
        ),
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

    # --------------------------------------------------------
    # FITNESS FUNCTION
    # --------------------------------------------------------

    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    # --------------------------------------------------------
    # ARTIFICIAL BEE COLONY
    # --------------------------------------------------------

    abc = ArtificialBeeColony(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        colony_size=COLONY_SIZE,
        limit=LIMIT
    )

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = abc.optimize()

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    weights = best_weights

    scores = (
        weights[0] * test_features["cn"]
        + weights[1] * test_features["jaccard"]
        + weights[2] * test_features["aa"]
        + weights[3] * test_features["ra"]
    )

    final_test_auc = roc_auc_score(
        test_labels,
        scores
    )

    return {
        "dataset": None,
        "budget": budget,
        "seed": seed,
        "best_validation_auc": best_validation_auc,
        "final_test_auc": final_test_auc,
        "evaluations": evaluations,
        "weight_cn": weights[0],
        "weight_jaccard": weights[1],
        "weight_aa": weights[2],
        "weight_ra": weights[3]
    }


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    file_exists = os.path.exists(
        RESULT_FILE
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

    # --------------------------------------------------------
    # PREPARE FIXED DATA ONCE PER DATASET
    # --------------------------------------------------------

    for dataset_name in DATASETS:

        print(
            f"\nPreparing dataset: "
            f"{dataset_name}"
        )

        graph = load_dataset(
            dataset_name
        )

        all_data[dataset_name] = (
            prepare_experiment_data(graph)
        )

        (
            validation_features,
            validation_labels,
            test_features,
            test_labels
        ) = all_data[dataset_name]

        print(
            "Validation samples:",
            len(validation_labels)
        )

        print(
            "Test samples:",
            len(test_labels)
        )

    # --------------------------------------------------------
    # TOTAL RUNS
    # --------------------------------------------------------

    total_runs = (
        len(DATASETS)
        * len(BUDGETS)
        * len(SEEDS)
    )

    completed_runs = 0

    print(
        f"\nTotal planned runs: "
        f"{total_runs}"
    )

    # --------------------------------------------------------
    # RUN ALL EXPERIMENTS
    # --------------------------------------------------------

    for dataset_name in DATASETS:

        (
            validation_features,
            validation_labels,
            test_features,
            test_labels
        ) = all_data[dataset_name]

        for budget in BUDGETS:

            for seed in SEEDS:

                print(
                    "\n" + "-" * 70
                )

                print(
                    f"Dataset: {dataset_name}"
                )

                print(
                    f"Budget: {budget}"
                )

                print(
                    f"Seed: {seed}"
                )

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

                print(
                    f"Evaluations: "
                    f"{result['evaluations']}"
                )

                print(
                    f"Completed: "
                    f"{completed_runs}/{total_runs}"
                )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL ABC EXPERIMENTS COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total completed runs: "
        f"{completed_runs}"
    )

    print(
        f"Results saved to: "
        f"{RESULT_FILE}"
    )

if __name__ == "__main__":
    main()