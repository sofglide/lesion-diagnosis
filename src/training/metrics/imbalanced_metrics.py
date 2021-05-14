"""
Metrics for imbalanced datasets
"""
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


class ImbalancedMetrics:
    """
    Metrics for imbalanced datasets
    """

    def __init__(
        self,
    ) -> None:
        self.y_true: List[int] = []
        self.y_pred: List[int] = []

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

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """
        add samples
        :param y_true:
        :param y_pred:
        :return:
        """
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())

    def f1_score(self) -> float:
        return f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)

    def mcc(self) -> float:
        return matthews_corrcoef(self.y_true, self.y_pred)

    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    def eval(self) -> Dict[str, float]:
        return {"accuracy": self.accuracy(), "f1_score": self.f1_score(), "mcc": self.mcc()}
