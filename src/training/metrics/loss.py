"""
loss functions
"""
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from training.metrics.loss_functions.focal_loss import FocalLoss
from utils.computing_device import get_device


def get_criterion(loss_params: Optional[Dict[str, Any]] = None, weight: Optional[np.ndarray] = None) -> nn.Module:
    """
    Loss criterion
    :param loss_params:
    :param weight:
    :return:
    """
    weight_tensor = torch.FloatTensor(weight) if weight is not None else None
    if loss_params is None:
        loss_params = {"function": "cross_entropy"}

    loss_function = loss_params.pop("function", "cross_entropy")

    if loss_function == "cross_entropy":
        criterion: nn.Module = nn.CrossEntropyLoss(weight=weight_tensor, **loss_params)
    elif loss_function == "focal_loss":
        criterion = FocalLoss(**loss_params)
    else:
        raise NotImplementedError(f"unknown loss function {loss_function}")

    if get_device() == "cuda":
        criterion = criterion.cuda()
    return criterion
