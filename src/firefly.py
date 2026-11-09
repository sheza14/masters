
import numpy as np
from sklearn.metrics import roc_auc_score


class Firefly:
    """
    Firefly Algorithm for 4-dimensional link-prediction
    weight optimization.

    Feature order:
        0 = CN
        1 = Jaccard
        2 = AA
        3 = RA

    Optimization uses validation AUC only.
    Final test AUC is calculated separately by the runner.

    Search space: [0.0, 1.0]
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
        self.lower_bound = 0.0
        self.upper_bound = 1.0

        self.rng = np.random.default_rng(seed)

        # Initialize fireflies inside [0, 1]
        self.positions = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(population_size, self.dimensions)
        )

        self.fitness_values = np.full(
            population_size,
            -np.inf
        )

        self.best_position = None
        self.best_score = -np.inf

    def evaluate_population(self):
        """Evaluate fireflies without exceeding the budget."""

        for i in range(self.population_size):

            if self.fitness_function.evaluations >= self.budget:
                break

            score = self.fitness_function.evaluate(
                self.positions[i]
            )

            self.fitness_values[i] = score

            if score > self.best_score:
                self.best_score = score
                self.best_position = self.positions[i].copy()

    def move_fireflies(self):
        """
        Move less bright fireflies towards brighter fireflies.

        Positions and fitness values are copied before movement
        so comparisons use the same population snapshot.
        """

        old_positions = self.positions.copy()
        old_fitness = self.fitness_values.copy()

        new_positions = old_positions.copy()

        for i in range(self.population_size):

            for j in range(self.population_size):

                if old_fitness[j] > old_fitness[i]:

                    distance = np.linalg.norm(
                        old_positions[i] - old_positions[j]
                    )

                    attractiveness = (
                        self.beta0
                        * np.exp(
                            -self.gamma * distance * distance
                        )
                    )

                    random_step = (
                        self.alpha
                        * (
                            self.rng.random(self.dimensions)
                            - 0.5
                        )
                    )

                    new_positions[i] += (
                        attractiveness
                        * (
                            old_positions[j] - new_positions[i]
                        )
                        + random_step
                    )

            # Keep every weight within [0, 1]
            new_positions[i] = np.clip(
                new_positions[i],
                self.lower_bound,
                self.upper_bound
            )

        self.positions = new_positions

    def optimize(self):
        """
        Run optimization until the exact evaluation budget
        is reached.
        """

        while self.fitness_function.evaluations < self.budget:

            self.evaluate_population()

            if self.fitness_function.evaluations >= self.budget:
                break

            self.move_fireflies()

            # New positions need fresh fitness evaluations
            self.fitness_values = np.full(
                self.population_size,
                -np.inf
            )

        if self.best_position is None:
            raise RuntimeError(
                "Firefly did not evaluate any solution."
            )

        if self.fitness_function.evaluations != self.budget:
            raise RuntimeError(
                "Firefly did not use the exact evaluation budget."
            )

        return (
            self.best_position,
            self.best_score,
            self.fitness_function.evaluations
        )


def evaluate_test_auc(features, labels, weights):
    """
    Calculate final test ROC-AUC using the best
    weights obtained during validation optimization.
    """

    weights = np.asarray(weights, dtype=float)

    if len(weights) != 4:
        raise ValueError("Exactly four weights are required.")

    scores = (
        weights[0] * features["cn"]
        + weights[1] * features["jaccard"]
        + weights[2] * features["aa"]
        + weights[3] * features["ra"]
    )

    return roc_auc_score(labels, scores)
