
import numpy as np
from sklearn.metrics import roc_auc_score


class PSO:
    """
    Particle Swarm Optimization for link prediction.

    Feature order:
        0 = CN
        1 = Jaccard
        2 = AA
        3 = RA

    Fitness: Validation ROC-AUC
    Search bounds: [0, 1]
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

        if budget <= 0:
            raise ValueError("Budget must be positive.")

        if swarm_size <= 0:
            raise ValueError("Swarm size must be positive.")

        self.fitness_function = fitness_function
        self.budget = budget
        self.seed = seed
        self.swarm_size = swarm_size

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.dimensions = 4

        self.lower_bound = 0.0
        self.upper_bound = 1.0

        self.rng = np.random.default_rng(seed)

        # Initialize particle positions
        self.positions = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(swarm_size, self.dimensions)
        )

        # Initialize particle velocities
        self.velocities = self.rng.uniform(
            -0.1,
            0.1,
            size=(swarm_size, self.dimensions)
        )

        # Personal best positions and scores
        self.personal_best_positions = self.positions.copy()

        self.personal_best_scores = np.full(
            swarm_size,
            -np.inf
        )

        # Global best
        self.global_best_position = None
        self.global_best_score = -np.inf

    def optimize(self):
        """
        Run PSO until the exact fitness evaluation budget
        is reached.
        """

        while self.fitness_function.evaluations < self.budget:

            # Evaluate particles
            for i in range(self.swarm_size):

                if self.fitness_function.evaluations >= self.budget:
                    break

                score = self.fitness_function.evaluate(
                    self.positions[i]
                )

                # Update personal best
                if score > self.personal_best_scores[i]:

                    self.personal_best_scores[i] = score

                    self.personal_best_positions[i] = (
                        self.positions[i].copy()
                    )

                # Update global best
                if score > self.global_best_score:

                    self.global_best_score = score

                    self.global_best_position = (
                        self.positions[i].copy()
                    )

            # Stop when the exact budget is reached
            if self.fitness_function.evaluations >= self.budget:
                break

            # Update velocities and positions
            for i in range(self.swarm_size):

                r1 = self.rng.random(self.dimensions)
                r2 = self.rng.random(self.dimensions)

                self.velocities[i] = (
                    self.inertia * self.velocities[i]
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

                self.positions[i] += self.velocities[i]

                # Keep particle weights inside [0, 1]
                self.positions[i] = np.clip(
                    self.positions[i],
                    self.lower_bound,
                    self.upper_bound
                )

        if self.global_best_position is None:
            raise RuntimeError(
                "PSO did not evaluate any solution."
            )

        if self.fitness_function.evaluations != self.budget:
            raise RuntimeError(
                "PSO did not use the exact evaluation budget."
            )

        return (
            self.global_best_position,
            self.global_best_score,
            self.fitness_function.evaluations
        )


def evaluate_test_auc(features, labels, weights):
    """
    Calculate final test ROC-AUC using the best
    weights found during validation optimization.
    """

    weights = np.asarray(weights, dtype=float)

    if len(weights) != 4:
        raise ValueError("Exactly four weights are required.")

    scores = (
        weights[0] * features["cn"].to_numpy()
        + weights[1] * features["jaccard"].to_numpy()
        + weights[2] * features["aa"].to_numpy()
        + weights[3] * features["ra"].to_numpy()
    )

    return roc_auc_score(labels, scores)
