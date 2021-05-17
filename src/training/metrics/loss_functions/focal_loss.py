"""
focal loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.computing_device import get_device


class FocalLoss(nn.Module):
    """
    focal loss
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        constructor
        :param alpha:
        :param gamma:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def _get_weight(self, p: Tensor, t: Tensor) -> Tensor:
        pt = p * t + (1 - p) * (1 - t)
        w = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (w * (1 - pt).pow(self.gamma)).detach()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        focal loss
        :param prediction:
        :param target:
        :return:
        """
        input_prob = torch.sigmoid(prediction)
        if prediction.size(1) == 1:
            input_prob = torch.cat([1 - input_prob, input_prob], axis=1)  # type: ignore
            num_class = 2
        else:
            num_class = input_prob.size(1)
        binary_target = torch.eye(num_class)[target.long()]
        if get_device() == "cuda":
            binary_target = binary_target.cuda()
        binary_target = binary_target.contiguous()
        weight = self._get_weight(input_prob, binary_target)
        return F.binary_cross_entropy(input_prob, binary_target, weight, reduction="mean")
