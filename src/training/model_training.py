"""
model training
"""
from typing import Any, Dict, Optional, Tuple

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
    phase: str,
    epoch: int,
    train_loader: DataLoader,
    loss_params: Dict[str, Any],
    optimizer_params: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    train model
    :param model:
    :param phase:
    :param epoch:
    :param train_loader:
    :param optimizer_params:
    :param loss_params:
    :return:
    """
    status_message = config.get_status_msg()
    device = get_device()
    model.train(mode=True)
    train_loss = 0
    metrics_val: Dict[str, float] = {}
    train_metrics = ImbalancedMetrics()
    n_batches = len(train_loader)

    optimizer = get_optimizer(model, optimizer_params)
    criterion = get_criterion(loss_params)

    for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)

        train_metrics.update(targets.detach(), predicted.detach())
        metrics_val = train_metrics.eval()

        metric_str = get_metrics_string(metrics_val)
        logger.info(status_message.format(phase, epoch, batch_idx, n_batches, loss / batch_idx, metric_str))

    return train_loss / len(train_loader), metrics_val
