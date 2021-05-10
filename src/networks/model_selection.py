"""
model selection
"""
from typing import Any, Dict, Optional

from torch import nn
from torch.backends import cudnn

from networks.densenet import Densenet
from networks.hybrid import Hybrid
from networks.resnet import Resnet
from networks.simple_cnn import SimpleCNN
from utils.computing_device import get_device
from utils.logging import get_logger

logger = get_logger(__name__)


def get_model(network: str, num_classes: int, model_params: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    creates the selected model
    :param network:
    :param num_classes:
    :param model_params:
    :return:
    """
    logger.info("==> Building model..")
    device = get_device()
    if network == "SimpleCNN":
        net: nn.Module = SimpleCNN(num_classes=num_classes, params=model_params)
    elif network == "Resnet":
        net = Resnet(num_classes)
    elif network == "Densenet":
        net = Densenet(num_classes)
    elif network == "Hybrid":
        net = Hybrid(num_classes, params=model_params)
    else:
        raise ValueError(f"unknown network {network}")
    net = net.to(device)
    if device == "cuda":
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    return net
