"""
model validation
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import config
from saving.checkpoints import save_checkpoint
from training.logging import get_metrics_string
from training.metrics.imbalanced_metrics import ImbalancedMetrics
from training.metrics.loss import get_criterion
from utils.computing_device import get_device
from utils.logging import get_logger

logger = get_logger(__name__)


def validate_model(
    model: nn.Module,
    epoch: int,
    batch_size: int,
    val_loader: DataLoader,
    best_metrics: Dict[str, float],
    objective_metric: str,
    weight: Optional[np.ndarray] = None,
    loss_params: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    validate model
    :param model:
    :param epoch:
    :param batch_size:
    :param val_loader:
    :param best_metrics:
    :param objective_metric:
    :param weight:
    :param loss_params:
    :return:
    """
    status_message = config.get_status_msg()
    exp_dir = config.get_exp_dir()
    device = get_device()
    model.eval()
    batch_idx = 0
    valid_loss = 0
    validation_metrics = ImbalancedMetrics()
    n_batches = len(val_loader.dataset) // batch_size  # type: ignore
    criterion = get_criterion(loss_params, weight)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)

            validation_metrics.update(targets.cpu(), predicted.cpu())
            metrics_val = validation_metrics.eval()

            metric_str = get_metrics_string(metrics_val)
            logger.info(status_message.format(epoch, batch_idx + 1, n_batches, loss / (batch_idx + 1), metric_str))

    # Save checkpoint.
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
    }
    state.update(metrics_val)
    if metrics_val[objective_metric] > best_metrics[objective_metric]:
        logger.info(f"Found better {objective_metric}, saving...")
        save_checkpoint(state, exp_dir, backup_as_best=True)
        best_metrics = metrics_val.copy()
    else:
        save_checkpoint(state, exp_dir, backup_as_best=False)

    return valid_loss / (batch_idx + 1), metrics_val, best_metrics
