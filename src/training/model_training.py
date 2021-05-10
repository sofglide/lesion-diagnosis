"""
model training
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from config import config
from training.logging import get_metrics_string
from training.metrics.imbalanced_metrics import ImbalancedMetrics
from training.metrics.loss import get_criterion
from training.optimization import get_optimizer
from utils.computing_device import get_device
from utils.logging import get_logger

logger = get_logger(__name__)


def train_model(
    model: nn.Module,
    epoch: int,
    batch_size: int,
    train_loader: DataLoader,
    weight: Optional[np.ndarray] = None,
    optimizer_params: Optional[Dict[str, float]] = None,
    loss_params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    train model
    :param model:
    :param epoch:
    :param batch_size:
    :param train_loader:
    :param weight:
    :param optimizer_params:
    :param loss_params:
    :return:
    """
    logger.info("Epoch: %d" % epoch)
    status_message = config.get_status_msg()
    device = get_device()
    model.train(mode=True)
    batch_idx = 0
    train_loss = 0
    metrics_val: Dict[str, float] = {}
    train_metrics = ImbalancedMetrics()
    n_batches = len(train_loader.dataset) // batch_size  # type: ignore

    optimizer = get_optimizer(model, optimizer_params)
    criterion = get_criterion(loss_params, weight)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)

        train_metrics.update(targets.cpu(), predicted.cpu())
        metrics_val = train_metrics.eval()

        metric_str = get_metrics_string(metrics_val)
        logger.info(status_message.format(epoch, batch_idx + 1, n_batches, loss / (batch_idx + 1), metric_str))

    return train_loss / (batch_idx + 1), metrics_val
