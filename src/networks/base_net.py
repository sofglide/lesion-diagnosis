"""
transfer learning from densenet
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from torch import Tensor, nn


class BaseNet(ABC, nn.Module):
    """
    Densenet model, image size (3, 450, 600)
    """

    def __init__(
        self, num_classes: int = 1000, fine_tune: bool = False, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """

        :param num_classes:
        :param fine_tune:
        """
        super().__init__()

        self.model, self.classifier = self._get_model(num_classes)
        self.params = params
        self.set_fine_tune(fine_tune)

    @staticmethod
    @abstractmethod
    def _get_model(num_classes: int, params: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        pass

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """
        input size
        :return:
        """
        ...

    def set_fine_tune(self, fine_tune: bool) -> None:
        """
        set/unset fine tuning
        :param fine_tune:
        :return:
        """
        if self.classifier is None:
            return
        for param in self.model.parameters():
            param.requires_grad = fine_tune
        if not fine_tune:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation
        :param x:
        :return:
        """
        return self.model(x)
