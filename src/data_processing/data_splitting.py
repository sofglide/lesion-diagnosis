"""
Training and validation data splitting
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from config import config
from data_processing.ham10000 import HAM10000
from data_processing.metadata_loading import read_metadata
from utils.logging import get_logger

logger = get_logger(__name__)


def split_data(
    val_fraction: float, image_size: Tuple[int, int], random_seed: Optional[int] = None
) -> Tuple[HAM10000, HAM10000]:
    """

    :param val_fraction:
    :param image_size:
    :param random_seed:
    :return:
    """
    logger.info("==> Preparing data..")
    metadata = read_metadata()
    train_ids, val_ids = create_train_val_split(metadata, val_fraction=val_fraction, random_seed=random_seed)

    img_mean_std = get_img_mean_std(train_ids, image_size)

    train_set = HAM10000(metadata, train_ids, image_size=image_size, is_eval=False, image_mean_std=img_mean_std)
    val_set = HAM10000(metadata, val_ids, image_size=image_size, is_eval=True, image_mean_std=img_mean_std)
    return train_set, val_set


def create_train_val_split(
    metadata: pd.DataFrame, val_fraction: float, random_seed: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Split data into training and validation sets, based on given fractions
    Images are grouped by 'lesion_id' and only 1 image from each lesion_id is taken
    :param metadata:
    :param val_fraction:
    :param random_seed:
    :return:
    """
    test_fraction = config.get_test_fraction()
    test_seed = config.get_test_seed()

    is_duplicated = metadata["lesion_id"].duplicated(keep=False)
    image_ids = metadata.index.to_numpy()
    labels = metadata.loc[image_ids, "dx"].to_numpy()

    log_split_distribution(labels, partition="all")

    train_val_xy, test_xy = create_stratified_split(  # test_xy is not used, in this phase, kept for later testing
        image_ids[~is_duplicated], labels[~is_duplicated], test_size=test_fraction, random_state=test_seed
    )
    log_split_distribution(test_xy["y"], partition="test")

    train_xy, val_xy = create_stratified_split(
        train_val_xy["X"],
        train_val_xy["y"],
        test_size=val_fraction,
        random_state=random_seed,
    )

    train_xy["X"] = np.concatenate([train_xy["X"], image_ids[is_duplicated]])
    train_xy["y"] = np.concatenate([train_xy["y"], labels[is_duplicated]])

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
    logger.info(f"partition '{partition:6s}': size: {len(labels)} | distribution: {distribution_str}")


def get_img_mean_std(train_ids: List[str], image_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Get train data mean and std for normalization
    :param train_ids:
    :param image_size:
    :return:
    """
    data_dir = config.get_data_dir()

    images = []
    means, stdevs = [], []
    for img in tqdm(train_ids):
        img = Image.open((data_dir / img).with_suffix(".jpg"))
        if all(image_size):
            img = img.resize(image_size)
        img_np = np.array(img)
        images.append(img_np)

    images_np = np.stack(images, axis=3).astype(np.float32) / 255.0

    for i in range(3):
        pixels = images_np[:, :, i, :].ravel()
        means.append(pixels.mean())
        stdevs.append(pixels.std())

    data_mean_std = {"mean": np.array(means), "std": np.array(stdevs)}

    logger.info(f"training set channels: {data_mean_std}")
    return data_mean_std
