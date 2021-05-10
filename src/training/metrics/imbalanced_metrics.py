"""
Metrics for imbalanced datasets
"""
from typing import Dict, List, Optional

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


class ImbalancedMetrics:
    """
    Metrics for imbalanced datasets
    """

    def __init__(self, class_weights: Optional[List[float]] = None) -> None:
        self.class_weights = class_weights
        self.y_true: List[int] = []
        self.y_pred: List[int] = []
        self.sample_weight: List[float] = []

    @staticmethod
    def get_metrics() -> List[str]:
        return ["f1_score", "mcc"]

    def reset(self) -> None:
        """
        reset states
        :return:
        """
        self.y_true = []
        self.y_pred = []
        self.sample_weight = []

    def update(self, y_true: List[int], y_pred: List[int]) -> None:
        """
        add samples
        :param y_true:
        :param y_pred:
        :return:
        """
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        if self.class_weights is not None:
            self.sample_weight.extend([self.class_weights[c] for c in y_true])

    def f1_score(self) -> float:
        weights = self.sample_weight if self.class_weights is not None else None
        return f1_score(self.y_true, self.y_pred, average="weighted", sample_weight=weights, zero_division=0)

    def mcc(self) -> float:
        weights = self.sample_weight if self.class_weights is not None else None
        return matthews_corrcoef(self.y_true, self.y_pred, sample_weight=weights)

    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    def eval(self) -> Dict[str, float]:
        return {"accuracy": self.accuracy(), "f1_score": self.f1_score(), "mcc": self.mcc()}
