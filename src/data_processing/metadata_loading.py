"""
metadata loading
"""
from pathlib import Path

import pandas as pd


def read_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Read meta data file using Pandas
    :param data_dir:
    :return:
    """
    meta_data = pd.read_csv(data_dir / "HAM10000_metadata.csv", index_col="image_id")
    return meta_data
