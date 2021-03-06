"""
entry point for running experiments
"""
import json
import re
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from config import config
from data_processing.data_loading import get_data_loaders
from networks.model_selection import get_model
from saving.checkpoints import load_best_checkpoint
from saving.data_params import save_mappings
from saving.predictions import save_predictions
from training.training_manager import start_training
from utils.logging import get_logger

logger = get_logger(__name__)


def run_experiment(
    val_fraction: float,
    batch_size: int,
    network: str,
    model: Union[str, Dict[str, Any]],
    optimizer_params: Mapping[str, Optional[Dict[str, float]]],
    loss: Dict[str, Any],
    epochs_extraction: int,
    epochs_tuning: int,
    objective_metric: str,
    seed: int,
) -> Dict[str, float]:
    """
    run a single experiment
    :param val_fraction:
    :param batch_size:
    :param network:
    :param model:
    :param optimizer_params:
    :param loss:
    :param epochs_extraction:
    :param epochs_tuning:
    :param objective_metric:
    :param seed:
    :return:
    """
    logger.info("experiment: START")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    image_size = config.get_model_size(network)
    network_copies = re.search(r"_(\d+)$", network)
    if network_copies:
        image_copies = int(network_copies[1])
    else:
        image_copies = 1
    train_loader, val_loader, num_classes = get_data_loaders(
        batch_size=batch_size,
        val_fraction=val_fraction,
        image_size=image_size,
        image_copies=image_copies,
        random_seed=seed,
    )

    # noinspection PyUnresolvedReferences
    save_mappings(train_loader.dataset.class_map_dict)  # type: ignore

    model_params_dict = json.loads(model) if isinstance(model, str) else model

    net = get_model(network, num_classes, model_params=model_params_dict)

    best_valid_metrics = start_training(
        net,
        epochs_extraction,
        epochs_tuning,
        train_loader,
        val_loader,
        objective_metric=objective_metric,
        loss=loss,
        optimizer_params=optimizer_params,
    )
    best_model_params = load_best_checkpoint()["model"]
    net.load_state_dict(best_model_params)

    save_predictions(model=net, train_loader=train_loader, val_loader=val_loader)
    logger.info(f"best validation {objective_metric} for experiment: {best_valid_metrics}")
    logger.info("experiment: END")

    return best_valid_metrics
