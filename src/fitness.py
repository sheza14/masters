import numpy as np
from sklearn.metrics import roc_auc_score


class FitnessFunction:
    """
    Fitness function for metaheuristic link prediction.

    The optimizer evaluates candidate weights using
    VALIDATION data only.

    Every call to evaluate() counts as exactly
    one fitness evaluation.
    """

    def __init__(
        self,
        features,
        labels,
        budget
    ):

        self.features = features
        self.labels = labels
        self.budget = budget

        # Actual fitness evaluations performed
        self.evaluations = 0

    def evaluate(self, weights):

        # Check evaluation budget
        if self.evaluations >= self.budget:
            raise RuntimeError(
                "Evaluation budget exhausted."
            )

        # Count this actual fitness call
        self.evaluations += 1

        weights = np.asarray(
            weights,
            dtype=float
        )

        # Weighted link-prediction score
        scores = (
            weights[0] * self.features["cn"]
            + weights[1] * self.features["jaccard"]
            + weights[2] * self.features["aa"]
            + weights[3] * self.features["ra"]
        )

        # Validation ROC-AUC
        auc = roc_auc_score(
            self.labels,
            scores
        )

        return auc