"""
metrics and losses
"""
from typing import Dict, Optional

from torch import nn, optim


def get_optimizer(model: nn.Module, optimizer_params: Optional[Dict[str, float]] = None) -> optim.Optimizer:
    """
    Training optimizer
    :param model:
    :param optimizer_params:
    :return:
    """
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    if optimizer_params is None:
        optimizer_params = dict()
    return optim.Adam(params_to_update, lr=optimizer_params.get("lr", 0.001))
