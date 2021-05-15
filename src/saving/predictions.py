"""
calculating predictions at the end of the training
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import config
from utils.computing_device import get_device
from utils.logging import get_logger

logger = get_logger(__name__)


def save_predictions(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
    """
    Save predictions as csv dataframes
    :param model:
    :param train_loader:
    :param val_loader:
    :return:
    """
    logger.info("prediction saving: START")
    predictions_message = config.get_prediction_msg()
    target_dir = config.get_exp_dir()
    load_model_best_params(model)

    device = get_device()
    with torch.no_grad():
        for phase, data_loader in zip(["train", "val"], [train_loader, val_loader]):
            n_batches = len(train_loader)
            true_labels, predicted_labels = [], []
            for batch_idx, (inputs, targets) in enumerate(data_loader, 1):
                inputs = inputs.to(device)
                outputs = model(inputs)
                true_labels.extend(targets.tolist())
                predicted_labels.extend(outputs.max(1).indices.tolist())

                logger.info(predictions_message.format(batch_idx, n_batches))

            results = pd.DataFrame(data={"reference": true_labels, "predicted": predicted_labels})

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
