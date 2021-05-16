"""
paring experiment folder
"""
from pathlib import Path
from typing import List

import pandas as pd

from config import config
from experiment_analysis.experiment import Experiment


def list_experiments(experiment_dir: Path) -> List[Path]:
    return list(experiment_dir.iterdir())


def load_experiments(path: Path) -> pd.DataFrame:
    """

    :param path:
    :return:
    """
    experiment_paths = list_experiments(path)
    experiments_list = [Experiment(exp) for exp in experiment_paths]
    properties = ["metric", "f1_score", "mcc", "network", "batch_size", "lr", "name"]
    experiments_dict = {prop: [getattr(exp, prop) for exp in experiments_list] for prop in properties}
    experiments_df = pd.DataFrame(data=experiments_dict)
    return experiments_df


if __name__ == "__main__":
    experiments_dir = Path("./experiments")
    experiments = load_experiments(experiments_dir)
