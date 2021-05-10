"""
general experiment setup
"""
import logging
from datetime import datetime
from typing import Any, Dict

from config import config
from utils.logging import log_arguments_to_file, setup_logger


def setup_experiment_env(
    *,
    exp_name: str,
    data_dir: str,
    log_level: str,
    val_fraction: float,
    batch_size: int,
    network: str,
    model_params: Dict[str, Any],
    lr: float,
    loss: Dict[str, Any],
    num_epochs: int,
    objective_metric: str,
    early_stop_count: int,
    early_stop_ratio: float,
    seed: int,
) -> None:
    """
    setup general configuration for experiment
    :param exp_name:
    :param data_dir:
    :param log_level:
    :param val_fraction:
    :param batch_size:
    :param network:
    :param model_params:
    :param lr:
    :param loss:
    :param num_epochs:
    :param objective_metric:
    :param early_stop_count:
    :param early_stop_ratio:
    :param seed:
    :return:
    """
    experiment_dict = {
        "network": network,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "val_fraction": val_fraction,
        "model_params": model_params,
        "lr": lr,
        "loss": loss,
        "objective_metric": objective_metric,
        "early_stop_count": early_stop_count,
        "early_stop_ratio": early_stop_ratio,
        "seed": seed,
    }
    setup_experiment_dir(exp_name, experiment_dict)
    config.set_data_dir(data_dir)
    config.set_log_to_file(True)
    config.set_log_level(log_level)
    log_arguments_to_file(experiment_dict)

    for logger_name in config.get_loggers():
        logger = logging.getLogger(logger_name)
        setup_logger(logger)


def setup_experiment_dir(exp_name: str, exp_dict: Dict[str, Any]) -> None:
    """
    Setup experiment directory
    :param exp_name:
    :param exp_dict:
    :return:
    """
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    params_in_dir_name = ["network", "lr", "batch_size", "objective_metric"]
    experiment_dir = (
        exp_name + "_" + now + "_" + "_".join(f"{k}_{v}" for k, v in exp_dict.items() if k in params_in_dir_name)
    )
    config.set_exp_dir(experiment_dir)
