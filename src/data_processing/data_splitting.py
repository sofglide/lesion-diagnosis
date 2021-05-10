"""
Training and validation data splitting
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from config import config
from data_processing.ham10000 import HAM10000
from data_processing.metadata_loading import read_metadata
from utils.logging import get_logger

logger = get_logger(__name__)


def split_data(
    val_fraction: float, image_size: Optional[int] = None, random_seed: Optional[int] = None
) -> Tuple[HAM10000, HAM10000]:
    """

    :param val_fraction:
    :param image_size:
    :param random_seed:
    :return:
    """
    logger.info("==> Preparing data..")
    data_dir = config.get_data_dir()
    train_ids, val_ids = create_train_val_split(data_dir, val_fraction=val_fraction, random_seed=random_seed)

    distribution_moments = get_train_data_moments(train_ids, image_size)

    train_set = HAM10000(
        data_dir, train_ids, is_eval=False, distrib_moments=distribution_moments, image_size=image_size
    )
    val_set = HAM10000(data_dir, val_ids, is_eval=True, distrib_moments=distribution_moments, image_size=image_size)
    return train_set, val_set


def create_train_val_split(
    data_dir: Path, val_fraction: float, random_seed: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Split data into training and validation sets, based on given fractions
    Images are grouped by 'lesion_id' and only 1 image from each lesion_id is taken
    :param data_dir:
    :param val_fraction:
    :param random_seed:
    :return:
    """
    test_fraction = config.get_test_fraction()
    test_seed = config.get_test_seed()

    metadata = read_metadata(data_dir)
    image_ids = metadata.groupby("lesion_id").sample(n=1, random_state=random_seed).index.to_numpy()
    labels = metadata.loc[image_ids, "dx"].to_numpy()

    log_split_distribution(labels, partition="all")

    train_val_xy, test_xy = create_stratified_split(  # test_xy is not used, in this phase, kept for later testing
        image_ids, labels, test_size=test_fraction, random_state=test_seed
    )

    log_split_distribution(test_xy["y"], partition="test")

    train_xy, val_xy = create_stratified_split(
        train_val_xy["X"],
        train_val_xy["y"],
        test_size=val_fraction,
        random_state=random_seed,
    )

    log_split_distribution(train_xy["y"], partition="train")
    log_split_distribution(val_xy["y"], partition="valid")

    return train_xy["X"].tolist(), val_xy["X"].tolist()


def create_stratified_split(
    data_x: np.ndarray,
    data_y: np.ndarray,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    creates stratified splitting of data based on label
    :param data_x:
    :param data_y:
    :param test_size:
    :param random_state:
    :return:
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(data_x, data_y))
    return {"X": data_x[train_idx], "y": data_y[train_idx]}, {"X": data_x[test_idx], "y": data_y[test_idx]}


def log_split_distribution(labels: np.ndarray, partition: str) -> None:
    """
    Log split label distribution
    :param labels:
    :param partition:
    :return:
    """
    label_distribution = pd.Series(labels).value_counts(normalize=True).sort_index().to_dict()
    distribution_str = {key: f"{val*100:2.0f} %" for key, val in label_distribution.items()}
    logger.info(f"partition '{partition:6s}': {distribution_str}")


def get_train_data_moments(train_ids: List[str], image_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Get train data moments for normalization
    :param train_ids:
    :param image_size:
    :return:
    """
    data_dir = config.get_data_dir()
    normalization_train_set = HAM10000(data_dir, train_ids, is_eval=True, image_size=image_size)
    normalization_train_loader: DataLoader = DataLoader(normalization_train_set, batch_size=128, num_workers=1)
    batch_shape = next(iter(normalization_train_loader))[0].shape
    norm_ratio = batch_shape[2] * batch_shape[3]

    count, image_sum, image_squared_sum = torch.zeros(3), torch.zeros(3), torch.zeros(3)
    for images, _ in normalization_train_loader:
        count += images.shape[0] * norm_ratio
        image_sum += images.sum(dim=[0, 2, 3])
        image_squared_sum += torch.pow(images, 2).sum(dim=[0, 2, 3])

    data_moments = {
        "mean": (image_sum / count).numpy(),
        "std": torch.sqrt(image_squared_sum / count - torch.pow(image_sum / count, 2)).numpy(),
    }

    logger.info(f"training set channels: {data_moments}")
    return data_moments
