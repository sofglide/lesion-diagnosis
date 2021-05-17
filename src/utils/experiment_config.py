"""
experiment config
"""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    """
    experiment config
    """

    network: str
    epochs_extraction: int
    epochs_tuning: int
    batch_size: int
    val_fraction: float
    model_params: Dict[str, Any]
    lr_extraction: float
    lr_tuning: float
    loss: Dict[str, Any]
    objective_metric: str
    seed: int
