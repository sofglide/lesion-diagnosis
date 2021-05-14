"""
saving mappings
"""
import json
from typing import Dict

from config import config


def save_mappings(class_map_dict: Dict[str, int]) -> None:
    """
    save class label mappings
    :param class_map_dict:
    :return:
    """
    with open(config.get_class_map_file(), "w") as fp:
        json.dump(class_map_dict, fp)
