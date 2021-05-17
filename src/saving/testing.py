"""
testing set
"""
import pandas as pd

from config import config


def dump_test_set(labels: pd.Series) -> None:
    test_set_file = config.get_test_set_file()
    labels.to_csv(test_set_file)
