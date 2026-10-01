import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from fitness import FitnessFunction
from data_loader import (
    split_graph,
    split_validation_graph
)
from features import calculate_features


class Firefly:
    """
    Firefly Algorithm for 4-dimensional
    link-prediction weight optimization.

    Dimensions:
        0 = CN
        1 = Jaccard
        2 = AA
        3 = RA

    The Firefly Algorithm uses VALIDATION AUC
    during optimization.

    The final TEST AUC is calculated only after
    optimization is complete.

    Search space:
        [-1.0, 1.0]
    """

    def __init__(
        self,
        fitness_function,
        budget,
        seed=42,
        population_size=10,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0
    ):

        self.fitness_function = fitness_function
        self.budget = budget
        self.seed = seed

        self.population_size = population_size

        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

        self.dimensions = 4

        self.rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # Initial firefly positions
        # Search space = [-1, 1]
        # ----------------------------------------------------

        self.positions = self.rng.uniform(
            -1.0,
            1.0,
            size=(
                population_size,
                self.dimensions
            )
        )

        # Store fitness values
        self.fitness_values = np.full(
            population_size,
            -np.inf
        )

        self.best_position = None
        self.best_score = -np.inf

    def evaluate_population(self):
        """
        Evaluate fireflies while respecting
        the exact fitness-evaluation budget.
        """

        for i in range(self.population_size):

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            score = self.fitness_function.evaluate(
                self.positions[i]
            )

            self.fitness_values[i] = score

            # Strict improvement only
            if score > self.best_score:

                self.best_score = score

                self.best_position = (
                    self.positions[i].copy()
                )

    def move_fireflies(self):
        """
        Move less bright fireflies toward
        brighter fireflies.
        """

        for i in range(self.population_size):

            for j in range(self.population_size):

                # Firefly j is brighter than firefly i
                if (
                    self.fitness_values[j]
                    > self.fitness_values[i]
                ):

                    distance = np.linalg.norm(
                        self.positions[i]
                        - self.positions[j]
                    )

                    attractiveness = (
                        self.beta0
                        * np.exp(
                            -self.gamma
                            * distance
                            * distance
                        )
                    )

                    random_step = (
                        self.alpha
                        * (
                            self.rng.random(
                                self.dimensions
                            )
                            - 0.5
                        )
                    )

                    self.positions[i] += (
                        attractiveness
                        * (
                            self.positions[j]
                            - self.positions[i]
                        )
                        + random_step
                    )

                    # ------------------------------------------------
                    # Keep weights inside [-1, 1]
                    # ------------------------------------------------

                    self.positions[i] = np.clip(
                        self.positions[i],
                        -1.0,
                        1.0
                    )

    def optimize(self):
        """
        Run Firefly optimization until the
        exact fitness-evaluation budget is reached.
        """

        while (
            self.fitness_function.evaluations
            < self.budget
        ):

            # Evaluate current population
            self.evaluate_population()

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            # Move fireflies
            self.move_fireflies()

            # Reset fitness values because positions changed
            self.fitness_values = np.full(
                self.population_size,
                -np.inf
            )

        return (
            self.best_position,
            self.best_score,
            self.fitness_function.evaluations
        )


def evaluate_test_auc(
    features,
    labels,
    weights
):
    """
    Calculate final TEST ROC-AUC using the
    best weights found by Firefly.
    """

    weights = np.asarray(
        weights,
        dtype=float
    )

    scores = (
        weights[0] * features["cn"]
        + weights[1] * features["jaccard"]
        + weights[2] * features["aa"]
        + weights[3] * features["ra"]
    )

    return roc_auc_score(
        labels,
        scores
    )


def prepare_data(graph):
    """
    Prepare leakage-free validation and test data.

    Final test set:
        30% of original edges

    Validation set:
        30% of the remaining 70% training edges

    Firefly optimization sees ONLY validation data.

    Final test evaluation uses the untouched
    30% test set.

    Note:
        Feature scaling is handled by the experiment
        runner (experiments/run_firefly.py).
    """

    (
        original,
        training_graph,
        positive_test_edges,
        negative_test_edges
    ) = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    ) = split_validation_graph(
        training_graph,
        validation_ratio=0.30,
        seed=123
    )

    # --------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------

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

    # --------------------------------------------------
    # FINAL TEST DATA
    # --------------------------------------------------

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


if __name__ == "__main__":

    # --------------------------------------------------
    # SINGLE FIREFLY TEST
    # --------------------------------------------------

    graph = nx.karate_club_graph()

    budget = 20000
    seed = 1

    (
        validation_features,
        validation_labels,
        test_features,
        test_labels
    ) = prepare_data(graph)

    # Create validation fitness function
    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    # Create Firefly optimizer
    firefly = Firefly(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        population_size=10,
        alpha=0.2,
        beta0=1.0,
        gamma=1.0
    )

    # Run optimization
    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = firefly.optimize()

    # Final evaluation on untouched test set
    test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    print("\nFirefly Leakage-Free Test")
    print("=" * 50)

    print("Dataset: Karate Club")
    print("Budget:", budget)
    print("Seed:", seed)

    print("\nBest weights:")
    print(best_weights)

    print(
        f"\nBest validation AUC: "
        f"{best_validation_auc:.4f}"
    )

    print(
        f"Final test AUC: "
        f"{test_auc:.4f}"
    )

    print(
        "Evaluations used:",
        evaluations
    )