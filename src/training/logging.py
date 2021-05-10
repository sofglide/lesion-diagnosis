"""
training logging
"""
from typing import Dict

from utils.logging import get_logger

logger = get_logger(__name__)


def get_metrics_string(metrics_val: Dict[str, float]) -> str:
    """
    transform metrics values to string for logging
    :param metrics_val:
    :return:
    """
    return ", ".join(f"{k}: {v:.04f}" for k, v in metrics_val.items())
