"""
entry point for model training
"""
from typing import Any, Dict, Optional

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from config import config
from saving.plotting import create_loss_plot, create_metrics_plots
from training.model_training import train_model
from training.model_validation import validate_model
from utils.logging import get_logger

logger = get_logger(__name__)


def start_training(
    net: nn.Module,
    num_epochs: int,
    batch_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    objective_metric: str,
    early_stopping: Dict[str, float],
    optimizer_params: Optional[Dict[str, float]] = None,
    loss: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    train model and save artifacts
    :param net:
    :param num_epochs:
    :param batch_size:
    :param train_loader:
    :param val_loader:
    :param objective_metric:
    :param early_stopping:
    :param optimizer_params:
    :param loss:
    :return:
    """
    logger.info("model training: START")
    early_stop_count = early_stopping.copy()
    early_stop_count.update({"max_count": early_stop_count["count"], "epochs_since_last_decrease": 0})

    epochs, train_losses, validation_losses = [], [], []
    train_metrics_all, valid_metrics_all = [], []
    best_valid_metrics = {objective_metric: -np.inf}
    previous_valid_loss = np.inf

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_model(
            net,
            epoch,
            batch_size,
            train_loader=train_loader,
            optimizer_params=optimizer_params,
            loss_params=loss,
        )

        valid_loss, valid_metrics, best_valid_metrics = validate_model(
            net,
            epoch,
            batch_size,
            val_loader=val_loader,
            best_metrics=best_valid_metrics,
            objective_metric=objective_metric,
            loss_params=loss,
        )

        epochs.append(epoch)
        train_losses.append(train_loss)
        validation_losses.append(valid_loss)
        train_metrics_all.append(train_metrics)
        valid_metrics_all.append(valid_metrics)

        create_loss_plot(epochs, train_losses, validation_losses)
        create_metrics_plots(epochs, train_metrics_all, valid_metrics_all)
        np.save(config.get_npy_file(), [train_losses, validation_losses])

        update_early_stop_count(valid_loss, previous_valid_loss, early_stop_count)
        previous_valid_loss = valid_loss
        if early_stop_count["count"] == 0:
            logger.info(
                f"early stopping at epoch {epoch}: "
                f"val loss not decreasing for the past {early_stop_count['epochs_since_last_decrease']} epochs"
            )
            break
    logger.info("model training: END")
    return best_valid_metrics


def update_early_stop_count(current_loss: float, previous_loss: float, early_stopping: Dict[str, float]) -> None:
    """
    early stopping counter
    :param current_loss:
    :param previous_loss:
    :param early_stopping:
    :return:
    """
    if current_loss > (1 + early_stopping["ratio"]) * previous_loss:
        early_stopping["count"] -= 1
        early_stopping["epochs_since_last_decrease"] += 1
    elif current_loss < previous_loss:
        early_stopping["count"] = early_stopping["max_count"]
        early_stopping["epochs_since_last_decrease"] += 0
    else:
        early_stopping["epochs_since_last_decrease"] += 1
