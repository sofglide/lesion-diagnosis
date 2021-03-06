"""
general experiment setup
"""
import logging
from datetime import datetime
from typing import Any, Dict

from config import config
from utils.experiment_config import ExperimentConfig
from utils.logging import log_arguments_to_file, setup_logger


def setup_experiment_env(
    *,
    val_fraction: float,
    batch_size: int,
    network: str,
    model_params: Dict[str, Any],
    lr_extraction: float,
    lr_tuning: float,
    loss: Dict[str, Any],
    epochs_extraction: int,
    epochs_tuning: int,
    objective_metric: str,
    seed: int,
) -> None:
    """
    setup general configuration for experiment
    :param val_fraction:
    :param batch_size:
    :param network:
    :param model_params:
    :param lr_extraction:
    :param lr_tuning:
    :param loss:
    :param epochs_extraction:
    :param epochs_tuning:
    :param objective_metric:
    :param seed:
    :return:
    """
    experiment_config = ExperimentConfig(
        network=network,
        epochs_extraction=epochs_extraction,
        epochs_tuning=epochs_tuning,
        batch_size=batch_size,
        val_fraction=val_fraction,
        model_params=model_params,
        lr_extraction=lr_extraction,
        lr_tuning=lr_tuning,
        loss=loss,
        objective_metric=objective_metric,
        seed=seed,
    )

    setup_experiment_dir(experiment_config.network)
    config.set_log_to_file(True)
    log_arguments_to_file(experiment_config)

    for logger_name in config.get_loggers():
        logger = logging.getLogger(logger_name)
        setup_logger(logger)


def setup_experiment_dir(exp_name: str) -> None:
    """
    Setup experiment directory
    :param exp_name:
    :return:
    """
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    experiment_dir = exp_name + "_" + now
    config.set_exp_dir(experiment_dir)
