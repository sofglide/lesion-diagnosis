"""
mapping classes to labels
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.utils import compute_class_weight

from data_processing.metadata_loading import read_metadata


def get_class_map_dict(data_dir: Path) -> Dict[str, int]:
    """
    Get dict to map label strings to label indices
    :param data_dir:
    :return:
    """
    metadata = read_metadata(data_dir)
    classes_list = metadata["dx"].unique().tolist()
    classes_list = sorted(classes_list)
    class_map_dict = {}
    for i, cls in enumerate(classes_list):
        class_map_dict[cls] = i

    return class_map_dict


def get_reverse_class_map_dict(class_map_dict: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in class_map_dict.items()}


def get_class_weights(labels: List[str], num_classes: int, class_map_dict: Dict[str, int]) -> np.ndarray:
    """
    Compute class weight for imbalanced dataset
    :param labels:
    :param num_classes:
    :param class_map_dict:
    :return:
    """
    reverse_class_dict = get_reverse_class_map_dict(class_map_dict)
    class_labels = [reverse_class_dict[i] for i in range(num_classes)]
    return compute_class_weight("balanced", classes=class_labels, y=labels)
