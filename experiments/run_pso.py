import sys
import os
import csv

# Add src folder to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src"
        )
    )
)

import networkx as nx
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from data_loader import (
    split_graph,
    split_validation_graph
)

from features import calculate_features

from fitness import FitnessFunction

from pso import (
    PSO,
    evaluate_test_auc
)


# ==========================================================
# EXPERIMENT SETTINGS
# ==========================================================

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

# 30% validation from the 70% training graph
VALIDATION_RATIO = 0.30

# 30% of the original graph is reserved for final testing
TEST_RATIO = 0.30


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(dataset_name):

    if dataset_name == "Karate Club":

        return nx.karate_club_graph()

    elif dataset_name == "Netscience":

        from data_loader import load_netscience

        return load_netscience()

    else:

        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_experiment_data(graph):

    # ------------------------------------------------------
    # Step 1: Fixed 30% final test split
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Step 2: Validation split inside 70% training graph
    # ------------------------------------------------------

    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    ) = split_validation_graph(
        training_graph,
        validation_ratio=VALIDATION_RATIO,
        seed=VALIDATION_SPLIT_SEED
    )

    # ------------------------------------------------------
    # Step 3: Validation data
    # ------------------------------------------------------

    validation_edges = (
        positive_validation_edges
        + negative_validation_edges
    )

    validation_labels = (
        [1] * len(positive_validation_edges)
        + [0] * len(negative_validation_edges)
    )

    # Calculate RAW validation features
    validation_features_raw = calculate_features(
        validation_training_graph,
        validation_edges
    )

    # ------------------------------------------------------
    # Step 4: Final test data
    # ------------------------------------------------------

    test_edges = (
        positive_test_edges
        + negative_test_edges
    )

    test_labels = (
        [1] * len(positive_test_edges)
        + [0] * len(negative_test_edges)
    )

    # Calculate RAW test features
    test_features_raw = calculate_features(
        training_graph,
        test_edges
    )

    # ------------------------------------------------------
    # Step 5: Consistent Min-Max scaling
    # ------------------------------------------------------
    #
    # The scaler is fitted ONLY on validation features.
    #
    # The SAME scaler is then used to transform the
    # final test features.
    #
    # We do NOT fit another scaler on the test data.
    # ------------------------------------------------------

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


# ==========================================================
# RUN ONE EXPERIMENT
# ==========================================================

def run_one_experiment(
    dataset_name,
    graph,
    budget,
    seed
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

    # Prepare validation and test data
    (
        validation_features,
        validation_labels,
        test_features,
        test_labels
    ) = prepare_experiment_data(graph)

    # ------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------

    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    # ------------------------------------------------------
    # PSO
    # ------------------------------------------------------

    pso = PSO(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        swarm_size=10
    )

    # ------------------------------------------------------
    # Optimization
    # ------------------------------------------------------

    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = pso.optimize()

    # ------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------

    test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print(
        f"Best validation AUC: "
        f"{best_validation_auc:.4f}"
    )

    print(
        f"Final test AUC: "
        f"{test_auc:.4f}"
    )

    print(
        f"Evaluations used: "
        f"{evaluations}"
    )

    print(
        "Best weights:",
        best_weights
    )

    # ------------------------------------------------------
    # Return result
    # ------------------------------------------------------

    return {
        "dataset": dataset_name,
        "budget": budget,
        "seed": seed,
        "best_validation_auc": best_validation_auc,
        "final_test_auc": test_auc,
        "evaluations": evaluations,
        "weight_cn": best_weights[0],
        "weight_jaccard": best_weights[1],
        "weight_aa": best_weights[2],
        "weight_ra": best_weights[3]
    }


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

def main():

    # ------------------------------------------------------
    # Project root
    # ------------------------------------------------------

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    # ------------------------------------------------------
    # Results folder
    # ------------------------------------------------------

    results_folder = os.path.join(
        project_root,
        "results"
    )

    os.makedirs(
        results_folder,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Output file
    # ------------------------------------------------------

    output_file = os.path.join(
        results_folder,
        "pso_results.csv"
    )

    # ------------------------------------------------------
    # CSV columns
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Experiment counters
    # ------------------------------------------------------

    all_results = []

    total_runs = (
        len(DATASETS)
        * len(BUDGETS)
        * len(SEEDS)
    )

    current_run = 0

    # ------------------------------------------------------
    # Header
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PSO BUDGET CURVE EXPERIMENT")
    print("=" * 60)

    print(
        f"Total experiments: {total_runs}"
    )

    print(
        "Datasets:",
        DATASETS
    )

    print(
        "Budgets:",
        BUDGETS
    )

    print(
        "Seeds:",
        SEEDS
    )

    print(
        f"Validation ratio: {VALIDATION_RATIO}"
    )

    print(
        f"Test ratio: {TEST_RATIO}"
    )

    print("=" * 60)

    # ======================================================
    # DATASET LOOP
    # ======================================================

    for dataset_name in DATASETS:

        print("\n")
        print("#" * 60)

        print(
            f"LOADING DATASET: {dataset_name}"
        )

        print(
            "#" * 60
        )

        graph = load_dataset(
            dataset_name
        )

        # ==================================================
        # BUDGET LOOP
        # ==================================================

        for budget in BUDGETS:

            # ==============================================
            # SEED LOOP
            # ==============================================

            for seed in SEEDS:

                current_run += 1

                print(
                    f"\nRUN {current_run}/{total_runs}"
                )

                result = run_one_experiment(
                    dataset_name,
                    graph,
                    budget,
                    seed
                )

                all_results.append(
                    result
                )

                # --------------------------------------------------
                # Save after every completed run
                # --------------------------------------------------

                with open(
                    output_file,
                    "w",
                    newline=""
                ) as csv_file:

                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=fieldnames
                    )

                    writer.writeheader()

                    writer.writerows(
                        all_results
                    )

    # ======================================================
    # FINISHED
    # ======================================================

    print("\n")
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(
        f"Total completed runs: "
        f"{len(all_results)}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )

    print("=" * 60)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()