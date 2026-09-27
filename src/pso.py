import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from fitness import FitnessFunction
from data_loader import (
    split_graph,
    split_validation_graph
)
from features import calculate_features


class PSO:
    """
    Particle Swarm Optimization for 4-dimensional
    link-prediction weight optimization.

    Dimensions:
        0 = CN
        1 = Jaccard
        2 = AA
        3 = RA

    The PSO uses VALIDATION AUC as its fitness.

    The final TEST AUC is calculated only after
    optimization is complete.
    """

    def __init__(
        self,
        fitness_function,
        budget,
        seed=42,
        swarm_size=10,
        inertia=0.7,
        cognitive=1.5,
        social=1.5
    ):

        self.fitness_function = fitness_function
        self.budget = budget
        self.seed = seed

        self.swarm_size = swarm_size

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        # Four feature weights
        self.dimensions = 4

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Initial particle positions
        self.positions = self.rng.uniform(
            0.0,
            1.0,
            size=(
                swarm_size,
                self.dimensions
            )
        )

        # Initial velocities
        self.velocities = self.rng.uniform(
            -0.1,
            0.1,
            size=(
                swarm_size,
                self.dimensions
            )
        )

        # Personal best
        self.personal_best_positions = (
            self.positions.copy()
        )

        self.personal_best_scores = np.full(
            swarm_size,
            -np.inf
        )

        # Global best
        self.global_best_position = None
        self.global_best_score = -np.inf

    def optimize(self):

        """
        Run PSO until the exact fitness evaluation
        budget is reached.
        """

        while (
            self.fitness_function.evaluations
            < self.budget
        ):

            # Evaluate particles
            for i in range(self.swarm_size):

                # Stop exactly at budget
                if (
                    self.fitness_function.evaluations
                    >= self.budget
                ):
                    break

                score = self.fitness_function.evaluate(
                    self.positions[i]
                )

                # Personal best
                if score > self.personal_best_scores[i]:

                    self.personal_best_scores[i] = score

                    self.personal_best_positions[i] = (
                        self.positions[i].copy()
                    )

                # Global best
                if score > self.global_best_score:

                    self.global_best_score = score

                    self.global_best_position = (
                        self.positions[i].copy()
                    )

            # Stop if budget exhausted
            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            # Update particles
            for i in range(self.swarm_size):

                r1 = self.rng.random(
                    self.dimensions
                )

                r2 = self.rng.random(
                    self.dimensions
                )

                self.velocities[i] = (
                    self.inertia
                    * self.velocities[i]

                    + self.cognitive
                    * r1
                    * (
                        self.personal_best_positions[i]
                        - self.positions[i]
                    )

                    + self.social
                    * r2
                    * (
                        self.global_best_position
                        - self.positions[i]
                    )
                )

                # Move particle
                self.positions[i] += (
                    self.velocities[i]
                )

                # Keep weights between 0 and 1
                self.positions[i] = np.clip(
                    self.positions[i],
                    0.0,
                    1.0
                )

        return (
            self.global_best_position,
            self.global_best_score,
            self.fitness_function.evaluations
        )


def evaluate_test_auc(
    features,
    labels,
    weights
):
    """
    Calculate final TEST ROC-AUC using the
    best weights found by PSO.
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

    # --------------------------------------------------------
    # Fixed final 30% test split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validation split inside the 70% training graph
    # --------------------------------------------------------

    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    ) = split_validation_graph(
        training_graph,
        validation_ratio=0.10,
        seed=123
    )

    # --------------------------------------------------------
    # Validation data
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
    # Final test data
    #
    # IMPORTANT:
    # The full 70% training graph is used here.
    # The final test edges were never used by PSO.
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


if __name__ == "__main__":

    # ========================================================
    # TEST PSO
    # ========================================================

    graph = nx.karate_club_graph()

    # Use one budget for testing
    budget = 20000

    # Optimizer seed
    seed = 42

    # Prepare validation and final test data
    (
        validation_features,
        validation_labels,
        test_features,
        test_labels
    ) = prepare_data(graph)

    # --------------------------------------------------------
    # Fitness uses VALIDATION data
    # --------------------------------------------------------

    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    # --------------------------------------------------------
    # Run PSO
    # --------------------------------------------------------

    pso = PSO(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        swarm_size=10
    )

    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = pso.optimize()

    # --------------------------------------------------------
    # Final TEST evaluation
    # --------------------------------------------------------

    test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nPSO Leakage-Free Test")
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