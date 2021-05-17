"""
experiment loading
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from config import config
from networks.model_selection import get_model
from utils.logging import get_logger

logger = get_logger(__name__)


class Experiment:
    """
    experiment results as class
    """

    def __init__(self, path: Path) -> None:
        """

        :param path:
        """
        self.directory = path
        self.name = path.name
        self.train, self.valid = load_experiment_predictions(path)
        self.arguments = load_experiment_arguments(path)
        self.model: Optional[nn.Module] = None
        with open(path / config.get("DEFAULT", "class_map_file"), "r") as fp:
            self.class_map = json.load(fp)

        self.f1_score, self.mcc = self.load_best_metrics()

    @property
    def network(self) -> str:
        return self.arguments["network"]

    @property
    def batch_size(self) -> int:
        return self.arguments["batch_size"]

    @property
    def lr(self) -> int:
        return self.arguments["lr"]

    @property
    def metric(self) -> str:
        return self.arguments["objective_metric"]

    @property
    def epochs(self) -> int:
        return self.arguments["num_epochs"]

    @property
    def reverse_class_map(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_map.items()}

    def load_model(self) -> None:
        self.model = load_best_model(
            self.directory, net=self.arguments["network"], model_params=self.arguments["model_params"]
        )

    def load_best_metrics(self) -> Tuple[float, float]:
        best_state = torch.load(self.directory / "model_best.pth.tar", map_location="cpu")
        return best_state["f1_score"], best_state["f1_score"]

    def display_plot(self, plot: str, ax: Optional[AxesImage] = None) -> AxesImage:
        """
        display one of the available plots
        :param plot: one of 'loss', 'f1_score', 'mcc'
        :param ax:
        :return:
        """
        image = plt.imread((self.directory / plot).with_suffix(".png"))
        if ax is None:
            ax = plt.imshow(image)
        else:
            ax.imshow(image)
        ax.axis("off")
        return ax

    def classification_report(self, phase: str = "valid") -> str:
        """
        classification report
        :param phase:
        :return:
        """
        if phase == "train":
            data = self.train
        elif phase == "valid":
            data = self.valid
        else:
            raise ValueError(f"unknown phase {phase}")
        return classification_report(
            data["reference"].map(self.reverse_class_map), data["predicted"].map(self.reverse_class_map)
        )

    def confusion_matrix(self, phase: str = "valid", normalize: str = "all") -> pd.DataFrame:
        """
        Confusion matrix
        :param phase:
        :param normalize:
        :return:
        """
        if phase == "train":
            data = self.train
        elif phase == "valid":
            data = self.valid
        else:
            raise ValueError(f"unknown phase {phase}")
        y_true = data["reference"].map(self.reverse_class_map)
        y_pred = data["predicted"].map(self.reverse_class_map)
        cf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
        df_cm = pd.DataFrame(cf_matrix, columns=np.unique(y_true), index=np.unique(y_true))
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        return df_cm

    def plot_confusion_matrix(self, phase: str = "valid", normalize: str = "true") -> None:
        """
        Plot confusion matrix
        :param phase:
        :param normalize:
        :return:
        """
        df_cm = self.confusion_matrix(phase, normalize=normalize)
        plt.figure(figsize=(10, 7))
        sn.set_context(context="notebook", font_scale=1.4)
        sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
        plt.show()


def load_best_model(path: Path, net: str, model_params: Dict[str, Any]) -> nn.Module:
    """
    load best model
    :param path:
    :param net:
    :param model_params:
    :return:
    """
    model = get_model(net, num_classes=config.get_num_classes(), model_params=model_params)
    model_state = torch.load(path / "model_best.pth.tar")
    model.load_state_dict(model_state["model"])
    # FIXME when loading model complains about "Missing key(s) in state_dict"
    return model


def load_experiment_predictions(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    load predictions
    :param path:
    :return:
    """
    train = pd.read_csv(path / "predictions_train.csv")
    valid = pd.read_csv(path / "predictions_val.csv")
    return train, valid


def load_experiment_arguments(path: Path) -> Dict[str, Any]:
    """
    load arguments file
    :param path:
    :return:
    """
    with open(path / "args.log") as fp:
        arguments = json.load(fp)
    return arguments


if __name__ == "__main__":
    experiments_dir = Path("./experiments")
    experiment_list = list(experiments_dir.iterdir())

    experiment = Experiment(experiment_list[0])

    logger.info(experiment.classification_report("train"))

    logger.info(experiment.confusion_matrix("valid"))

    experiment.plot_confusion_matrix("valid")

    experiment.display_plot("loss")

    experiment.load_model()
