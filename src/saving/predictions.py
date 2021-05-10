"""
calculating predictions at the end of the training
"""
from typing import Optional

import pandas as pd
import torch
from torch import nn

from config import config
from data_processing.data_loading import get_data_loaders
from utils.computing_device import get_device
from utils.logging import get_logger

logger = get_logger(__name__)


def save_predictions(
    model: nn.Module,
    batch_size: int,
    val_fraction: float,
    image_size: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> None:
    """
    Save predictions as csv dataframes
    :param model:
    :param batch_size:
    :param val_fraction:
    :param image_size:
    :param random_seed:
    :return:
    """
    logger.info("prediction saving: START")
    predictions_message = config.get_prediction_msg()
    target_dir = config.get_exp_dir()
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=batch_size, val_fraction=val_fraction, image_size=image_size, random_seed=random_seed
    )

    load_model_best_params(model)

    device = get_device()
    with torch.no_grad():
        for phase, data_loader in zip(["train", "val"], [train_loader, val_loader]):
            n_batches = len(data_loader.dataset) // batch_size  # type: ignore
            true_labels, predicted_labels = [], []
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                true_labels.extend(targets.tolist())
                predicted_labels.extend(outputs.max(1).indices.tolist())

                logger.info(predictions_message.format(batch_idx + 1, n_batches))

            results = pd.DataFrame(data={"reference": predicted_labels, "predicted": true_labels})

            results.to_csv(target_dir / f"predictions_{phase}.csv", index=False)
    logger.info("prediction saving: END")


def load_model_best_params(model: nn.Module) -> None:
    """
    Load model best params
    :param model:
    :return:
    """
    best_model_path = config.get_best_model_file()
    model_state_dict = torch.load(best_model_path)["model"]
    model.load_state_dict(model_state_dict)
    model.eval()
