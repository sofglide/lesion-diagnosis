"""
loss functions
"""
from typing import Any, Dict

import torch
from torch import nn

from training.metrics.loss_functions.focal_loss import FocalLoss
from utils.computing_device import get_device


def get_criterion(loss_params: Dict[str, Any]) -> nn.Module:
    """
    Loss criterion
    :param loss_params:
    :return:
    """
    loss_function = loss_params.pop("function", "cross_entropy")
    if "weight" in loss_params:
        loss_params["weight"] = torch.FloatTensor(loss_params["weight"])

    if loss_function == "cross_entropy":
        criterion: nn.Module = nn.CrossEntropyLoss(**loss_params)
    elif loss_function == "focal_loss":
        criterion = FocalLoss(**loss_params)
    else:
        raise NotImplementedError(f"unknown loss function {loss_function}")

    if get_device() == "cuda":
        criterion = criterion.cuda()
    return criterion
