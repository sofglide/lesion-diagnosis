"""
mapping classes to labels
"""
from typing import Dict

import pandas as pd


def get_class_map_dict(labels: pd.Series) -> Dict[str, int]:
    """
    Get dict to map label strings to label indices
    :param labels:
    :return:
    """
    return dict({category: index for index, category in enumerate(labels.cat.categories)})
