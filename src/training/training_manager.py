"""
entry point for model training
"""
from typing import Any, Dict, Mapping, Optional

import numpy as np
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from config import config
from networks.base_net import BaseNet
from saving.plotting import create_loss_plot, create_metrics_plots
from training.model_training import train_model
from training.model_validation import validate_model
from utils.logging import get_logger

logger = get_logger(__name__)


def start_training(
    net: BaseNet,
    epochs_extraction: int,
    epochs_tuning: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    objective_metric: str,
    loss: Dict[str, Any],
    optimizer_params: Mapping[str, Optional[Dict[str, float]]],
) -> Dict[str, float]:
    """
    train model and save artifacts
    :param net:
    :param epochs_extraction:
    :param epochs_tuning:
    :param train_loader:
    :param val_loader:
    :param objective_metric:
    :param loss:
    :param optimizer_params:
    :return:
    """
    logger.info("model training: START")

    epoch_indices, train_losses, validation_losses = [], [], []
    train_metrics_all, valid_metrics_all = [], []
    best_valid_metrics = {objective_metric: -np.inf}

    loss["weight"] = _get_class_weights(train_loader)

    phase_epochs = {"extraction": epochs_extraction, "tuning": epochs_tuning}

    for phase, n_epochs in phase_epochs.items():
        logger.info(f"Phase: {phase}")
        net.set_fine_tune(phase == "tuning")
        for epoch in range(1, n_epochs + 1):
            logger.info(f"Epoch: {epoch}")

            train_loss, train_metrics = train_model(
                net, phase, epoch, train_loader=train_loader, loss_params=loss, optimizer_params=optimizer_params[phase]
            )

            valid_loss, valid_metrics, best_valid_metrics = validate_model(
                net,
                phase,
                epoch,
                val_loader=val_loader,
                best_metrics=best_valid_metrics,
                objective_metric=objective_metric,
                loss_params=loss,
            )

            epoch_indices.append(epoch)
            train_losses.append(train_loss)
            validation_losses.append(valid_loss)
            train_metrics_all.append(train_metrics)
            valid_metrics_all.append(valid_metrics)

            create_loss_plot(epoch_indices, train_losses, validation_losses)
            create_metrics_plots(epoch_indices, train_metrics_all, valid_metrics_all)
            np.save(config.get_npy_file(), [train_losses, validation_losses])

    logger.info("model training: END")
    return best_valid_metrics


def _get_phase_optimizer_params(optimizer_params: Optional[Dict[str, float]], phase: str) -> Optional[Dict[str, float]]:
    if optimizer_params is None:
        return None
    return {key: val for key, val in optimizer_params.items() if key.endswith(f"_{phase}")}


def _get_class_weights(loader: DataLoader) -> np.ndarray:
    """
    compute class weights
    :param loader:
    :return:
    """
    labels = loader.dataset.metadata["dx"].cat.codes  # type: ignore
    classes = np.unique(labels)
    return compute_class_weight(class_weight="balanced", classes=classes, y=labels)
