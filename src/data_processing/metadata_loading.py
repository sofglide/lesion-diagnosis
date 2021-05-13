"""
metadata loading
"""

import pandas as pd

from config import config


def read_metadata() -> pd.DataFrame:
    """
    Read meta data file using Pandas
    :return:
    """
    data_dir = config.get_data_dir()
    metadata = pd.read_csv(data_dir / "HAM10000_metadata.csv", index_col="image_id", dtype={"dx": "category"})
    return metadata
