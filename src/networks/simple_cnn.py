"""
Neural Network
"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch import Tensor

INPUT_SIZE = (450, 600)
DEFAULT_PARAMS = {
    "conv": [(10, 3, 0, 0), (20, 3, 2, 0.25), (40, 3, 2, 0.25)],
    "fc": [(200, 0.2), (100, 0.2), (50, 0.1)],
}


class SimpleCNN(nn.Module):
    """
    CNN model, image size (3, 450, 600)
    """

    def __init__(self, num_classes: int = 1000, params: Optional[Dict[str, Any]] = None) -> None:
        """

        :param num_classes:
        :param params:
        """
        super().__init__()
        if not params:
            params = DEFAULT_PARAMS
        self.model = get_model_layers(params, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation
        :param x:
        :return:
        """
        return self.model(x)


def get_model_layers(model_params: Dict[str, Any], n_features: int) -> nn.Sequential:
    """
    Builds sequential model from layer parameters
    :param model_params:
    :param n_features:
    :return:
    """
    current_size = list(INPUT_SIZE)
    previous_depth = 3
    layers: List[Tuple[str, nn.Module]] = []
    for i, (depth, kernel, pool, dropout) in enumerate(model_params["conv"]):
        layers.append((f"conv2d_{i + 1}", nn.Conv2d(previous_depth, depth, kernel_size=kernel, stride=1)))
        layers.append((f"relu2d_{i + 1}", nn.ReLU()))
        if pool:
            layers.append((f"pool2d_{i + 1}", nn.MaxPool2d(kernel_size=pool)))
        if dropout:
            layers.append((f"dropout2d_{i + 1}", nn.Dropout2d(dropout)))

        _update_current_size(current_size, kernel, pool)
        previous_depth = depth

    layers.append(("flatten", nn.Flatten()))
    current_length = previous_depth * current_size[0] * current_size[1]
    i = -1
    for i, (hidden_size, dropout) in enumerate(model_params["fc"]):
        layers.append((f"linear_{i + 1}", nn.Linear(current_length, hidden_size)))
        layers.append((f"relu1d_{i + 1}", nn.ReLU()))
        if dropout:
            layers.append((f"dropout1d_{i + 1}", nn.Dropout(dropout)))
        current_length = hidden_size

    layers.append((f"linear_{i + 2}", nn.Linear(current_length, n_features)))
    return nn.Sequential(OrderedDict(layers))


def _update_current_size(current_size: List[int], kernel: int, pool: int) -> None:
    """
    Updates images size after transformation by conv layer
    :param current_size:
    :param kernel:
    :param pool:
    :return:
    """
    for i, _ in enumerate(current_size):
        current_size[i] -= 2 * (kernel // 2)
    if pool:
        for i, _ in enumerate(current_size):
            current_size[i] //= pool
