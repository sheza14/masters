import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from fitness import FitnessFunction
from data_loader import split_graph, split_validation_graph
from features import calculate_features


class ArtificialBeeColony:

    def __init__(
        self,
        fitness_function,
        budget,
        seed=42,
        colony_size=10,
        limit=5
    ):

        self.fitness_function = fitness_function
        self.budget = budget
        self.seed = seed
        self.colony_size = colony_size
        self.limit = limit
        self.dimensions = 4

        # Search bounds
        self.lower_bound = -1.0
        self.upper_bound = 1.0

        self.rng = np.random.default_rng(seed)

        # Half employed bees + half onlooker bees
        self.food_sources = colony_size // 2

        # ----------------------------------------------------
        # Initial food-source positions
        # Search space = [-1, 1]
        # ----------------------------------------------------

        self.positions = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(
                self.food_sources,
                self.dimensions
            )
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

            self.best_position = (
                self.positions[index].copy()
            )

        return True

    def evaluate_initial_population(self):

        for i in range(self.food_sources):

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            self.evaluate_position(i)

    def generate_neighbor(self, index):

        if self.food_sources <= 1:

            return self.positions[index].copy()

        # Select a different food source
        candidates = [
            i
            for i in range(self.food_sources)
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
                self.positions[index]
                - self.positions[k]
            )
        )

        # ----------------------------------------------------
        # Keep weights inside [-1, 1]
        # ----------------------------------------------------

        new_position = np.clip(
            new_position,
            self.lower_bound,
            self.upper_bound
        )

        return new_position

    def greedy_selection(
        self,
        index,
        candidate
    ):

        if (
            self.fitness_function.evaluations
            >= self.budget
        ):
            return False

        candidate_score = (
            self.fitness_function.evaluate(
                candidate
            )
        )

        current_score = (
            self.fitness_values[index]
        )

        # Strict improvement
        if candidate_score > current_score:

            self.positions[index] = candidate

            self.fitness_values[index] = (
                candidate_score
            )

            self.trial_counters[index] = 0

            if candidate_score > self.best_score:

                self.best_score = candidate_score

                self.best_position = (
                    candidate.copy()
                )

        else:

            self.trial_counters[index] += 1

        return True

    def employed_bee_phase(self):

        for i in range(self.food_sources):

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            candidate = self.generate_neighbor(i)

            self.greedy_selection(
                i,
                candidate
            )

    def calculate_probabilities(self):

        # Convert AUC values into positive
        # selection probabilities

        minimum = np.min(
            self.fitness_values
        )

        adjusted = (
            self.fitness_values
            - minimum
            + 1e-12
        )

        total = np.sum(adjusted)

        if total <= 0:

            return (
                np.ones(self.food_sources)
                / self.food_sources
            )

        return adjusted / total

    def onlooker_bee_phase(self):

        probabilities = (
            self.calculate_probabilities()
        )

        i = 0
        attempts = 0

        max_attempts = (
            self.food_sources * 3
        )

        while (
            i < self.food_sources
            and
            self.fitness_function.evaluations
            < self.budget
            and
            attempts < max_attempts
        ):

            attempts += 1

            selected = self.rng.random()

            cumulative = 0.0
            selected_index = 0

            for j, probability in enumerate(
                probabilities
            ):

                cumulative += probability

                if selected <= cumulative:

                    selected_index = j
                    break

            candidate = self.generate_neighbor(
                selected_index
            )

            self.greedy_selection(
                selected_index,
                candidate
            )

            i += 1

    def scout_bee_phase(self):

        for i in range(self.food_sources):

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            if (
                self.trial_counters[i]
                >= self.limit
            ):

                # Generate new position
                # inside [-1, 1]

                new_position = self.rng.uniform(
                    self.lower_bound,
                    self.upper_bound,
                    size=self.dimensions
                )

                self.positions[i] = new_position

                self.trial_counters[i] = 0

                self.evaluate_position(i)

    def optimize(self):

        # ----------------------------------------------------
        # Initial evaluation
        # ----------------------------------------------------

        self.evaluate_initial_population()

        # ----------------------------------------------------
        # Main ABC loop
        # ----------------------------------------------------

        while (
            self.fitness_function.evaluations
            < self.budget
        ):

            # Employed bees
            self.employed_bee_phase()

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            # Onlooker bees
            self.onlooker_bee_phase()

            if (
                self.fitness_function.evaluations
                >= self.budget
            ):
                break

            # Scout bees
            self.scout_bee_phase()

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

    # ----------------------------------------------------
    # Fixed final 30% test split
    # ----------------------------------------------------

    (
        _,
        training_graph,
        positive_test_edges,
        negative_test_edges
    ) = split_graph(
        graph,
        test_ratio=0.30,
        seed=42
    )

    # ----------------------------------------------------
    # Internal 30% validation split
    # ----------------------------------------------------

    (
        validation_training_graph,
        positive_validation_edges,
        negative_validation_edges
    ) = split_validation_graph(
        training_graph,
        validation_ratio=0.30,
        seed=123
    )

    # ----------------------------------------------------
    # Validation data
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # Final test data
    # ----------------------------------------------------

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

    graph = nx.karate_club_graph()

    budget = 20000
    seed = 42

    (
        validation_features,
        validation_labels,
        test_features,
        test_labels
    ) = prepare_data(graph)

    fitness = FitnessFunction(
        validation_features,
        validation_labels,
        budget=budget
    )

    abc = ArtificialBeeColony(
        fitness_function=fitness,
        budget=budget,
        seed=seed,
        colony_size=10,
        limit=5
    )

    (
        best_weights,
        best_validation_auc,
        evaluations
    ) = abc.optimize()

    test_auc = evaluate_test_auc(
        test_features,
        test_labels,
        best_weights
    )

    print("\nABC Leakage-Free Test")
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