"""
saving weights and mappings
"""
import json
from typing import Dict

import numpy as np

from config import config


def save_weights_and_mappings(class_weights: np.ndarray, class_map_dict: Dict[str, int]) -> None:
    """
    save class weights and label mappings
    :param class_weights:
    :param class_map_dict:
    :return:
    """
    np.save(config.get_class_weight_file(), class_weights)
    with open(config.get_class_map_file(), "w") as fp:
        json.dump(class_map_dict, fp)
