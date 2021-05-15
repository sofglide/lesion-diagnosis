"""
Neural Network
"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from networks.base_net import BaseNet

INPUT_SIZE = (450, 600)
DEFAULT_PARAMS = {
    "conv": [(10, 3, 0, 0), (20, 3, 2, 0.25), (40, 3, 2, 0.25)],
    "fc": [(200, 0.2), (100, 0.2), (50, 0.1)],
}


class SimpleCNN(BaseNet):
    """
    CNN model, image size (3, 450, 600)
    """

    @property
    def input_size(self) -> Tuple[int, int]:
        return INPUT_SIZE

    @staticmethod
    def _get_model(num_classes: int, params: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, Optional[nn.Module]]:
        if not params:
            params = DEFAULT_PARAMS
        model = get_model_layers(params, num_classes)
        return model, None


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
    for i, (depth, kernel, pool, dropout) in enumerate(model_params["conv"], 1):
        layers.append((f"conv2d_{i}", nn.Conv2d(previous_depth, depth, kernel_size=kernel, stride=1)))
        layers.append((f"relu2d_{i}", nn.ReLU()))
        if pool:
            layers.append((f"pool2d_{i}", nn.MaxPool2d(kernel_size=pool)))
        if dropout:
            layers.append((f"dropout2d_{i}", nn.Dropout2d(dropout)))

        _update_current_size(current_size, kernel, pool)
        previous_depth = depth

    layers.append(("flatten", nn.Flatten()))
    current_length = previous_depth * current_size[0] * current_size[1]
    i = 0
    for i, (hidden_size, dropout) in enumerate(model_params["fc"], 1):
        layers.append((f"linear_{i}", nn.Linear(current_length, hidden_size)))
        layers.append((f"relu1d_{i}", nn.ReLU()))
        if dropout:
            layers.append((f"dropout1d_{i}", nn.Dropout(dropout)))
        current_length = hidden_size

    layers.append((f"linear_{i + 1}", nn.Linear(current_length, n_features)))
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
