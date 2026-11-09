import numpy as np
from sklearn.metrics import roc_auc_score


class ArtificialBeeColony:
    """
    Artificial Bee Colony optimization for link prediction.

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
        colony_size=10,
        limit=5
    ):
        if budget <= 0:
            raise ValueError("Budget must be positive.")

        if colony_size < 2:
            raise ValueError("Colony size must be at least 2.")

        if limit <= 0:
            raise ValueError("Limit must be positive.")

        self.fitness_function = fitness_function
        self.budget = budget
        self.seed = seed
        self.colony_size = colony_size
        self.limit = limit

        self.dimensions = 4
        self.lower_bound = 0.0
        self.upper_bound = 1.0

        self.rng = np.random.default_rng(seed)

        # Half of the colony are food sources.
        self.food_sources = colony_size // 2

        self.positions = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(self.food_sources, self.dimensions)
        )

        self.fitness_values = np.full(
            self.food_sources,
            -np.inf
        )

        self.trial_counters = np.zeros(
            self.food_sources,
            dtype=int
        )

        self.best_position = None
        self.best_score = -np.inf

    def evaluate_position(self, index):
        if self.fitness_function.evaluations >= self.budget:
            return False

        score = self.fitness_function.evaluate(
            self.positions[index]
        )

        self.fitness_values[index] = score

        if score > self.best_score:
            self.best_score = score
            self.best_position = self.positions[index].copy()

        return True

    def evaluate_initial_population(self):
        for i in range(self.food_sources):
            if self.fitness_function.evaluations >= self.budget:
                break

            self.evaluate_position(i)

    def generate_neighbor(self, index):
        candidates = [
            i for i in range(self.food_sources)
            if i != index
        ]

        k = self.rng.choice(candidates)

        phi = self.rng.uniform(
            -1.0,
            1.0,
            size=self.dimensions
        )

        new_position = (
            self.positions[index]
            + phi * (
                self.positions[index] - self.positions[k]
            )
        )

        return np.clip(
            new_position,
            self.lower_bound,
            self.upper_bound
        )

    def greedy_selection(self, index, candidate):
        if self.fitness_function.evaluations >= self.budget:
            return False

        candidate_score = self.fitness_function.evaluate(candidate)

        current_score = self.fitness_values[index]

        if candidate_score > current_score:
            self.positions[index] = candidate.copy()
            self.fitness_values[index] = candidate_score
            self.trial_counters[index] = 0

            if candidate_score > self.best_score:
                self.best_score = candidate_score
                self.best_position = candidate.copy()
        else:
            self.trial_counters[index] += 1

        return True

    def employed_bee_phase(self):
        for i in range(self.food_sources):
            if self.fitness_function.evaluations >= self.budget:
                break

            candidate = self.generate_neighbor(i)
            self.greedy_selection(i, candidate)

    def calculate_probabilities(self):
        minimum = np.min(self.fitness_values)

        adjusted = (
            self.fitness_values - minimum + 1e-12
        )

        total = np.sum(adjusted)

        if total <= 0:
            return np.ones(self.food_sources) / self.food_sources

        return adjusted / total

    def onlooker_bee_phase(self):
        probabilities = self.calculate_probabilities()

        i = 0
        attempts = 0
        max_attempts = self.food_sources * 3

        while (
            i < self.food_sources
            and self.fitness_function.evaluations < self.budget
            and attempts < max_attempts
        ):
            attempts += 1

            selected = self.rng.random()
            cumulative = 0.0
            selected_index = 0

            for j, probability in enumerate(probabilities):
                cumulative += probability

                if selected <= cumulative:
                    selected_index = j
                    break

            candidate = self.generate_neighbor(selected_index)

            self.greedy_selection(selected_index, candidate)

            i += 1

    def scout_bee_phase(self):
        for i in range(self.food_sources):
            if self.fitness_function.evaluations >= self.budget:
                break

            if self.trial_counters[i] >= self.limit:
                new_position = self.rng.uniform(
                    self.lower_bound,
                    self.upper_bound,
                    size=self.dimensions
                )

                self.positions[i] = new_position
                self.trial_counters[i] = 0

                self.evaluate_position(i)

    def optimize(self):
        # Evaluate initial food sources.
        self.evaluate_initial_population()

        # Continue until the exact budget is consumed.
        while self.fitness_function.evaluations < self.budget:
            self.employed_bee_phase()

            if self.fitness_function.evaluations >= self.budget:
                break

            self.onlooker_bee_phase()

            if self.fitness_function.evaluations >= self.budget:
                break

            self.scout_bee_phase()

        if self.best_position is None:
            raise RuntimeError("ABC did not evaluate any solution.")

        if self.fitness_function.evaluations != self.budget:
            raise RuntimeError(
                "ABC did not use the exact evaluation budget."
            )

        return (
            self.best_position,
            self.best_score,
            self.fitness_function.evaluations
        )


def evaluate_test_auc(features, labels, weights):
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