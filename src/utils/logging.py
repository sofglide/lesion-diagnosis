"""Utility functions."""
import json
import logging
from typing import Any, Dict

from config import config


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for module represented by name
    :param name:
    :return:
    """
    logger = logging.getLogger(name)
    config.add_logger(name)
    return logger


def setup_logger(logger: logging.Logger) -> None:
    """
    Prepare logging for the provided logger
    :param logger:
    :return:
    """
    log_level = config.get_log_level()
    logger.setLevel(log_level)
    logger.handlers = []
    log_format = config.get_log_format()
    log_formatter = logging.Formatter(fmt=log_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    if config.get_log_to_file():
        add_log_file_to_logger(logger, log_formatter)


def add_log_file_to_logger(logger: logging.Logger, log_formatter: logging.Formatter) -> None:
    """
    Add logfile to logger
    :param logger:
    :param log_formatter:
    :return:
    """
    log_path = config.get_log_file()
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)


def log_arguments_to_file(experiment_dict: Dict[str, Any]) -> None:
    """
    Logs experiments arguments to file
    :param experiment_dict:
    :return:
    """
    with open(config.get_args_file(), "w") as fp:
        json.dump(experiment_dict, fp, indent=2)
