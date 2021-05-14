"""
plotting functions
"""
from typing import Dict, List

from matplotlib import pyplot as plt

from config import config
from training.metrics.imbalanced_metrics import ImbalancedMetrics


def create_loss_plot(epochs: List[int], train_losses: List[float], valid_losses: List[float]) -> None:
    """
    Plot losses and save
    :param epochs:
    :param train_losses:
    :param valid_losses:
    :return:
    """
    exp_dir = config.get_exp_dir()
    fig = plt.figure()
    plt.title("Loss plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.plot(epochs, train_losses, "b", marker="o", label="train loss")
    plt.plot(epochs, valid_losses, "r", marker="o", label="valid loss")
    plt.legend()
    plt.savefig(exp_dir / "loss.png")
    plt.close(fig)


def create_metrics_plots(
    epochs: List[int], train_metrics: List[Dict[str, float]], valid_metrics: List[Dict[str, float]]
) -> None:
    """
    Plot losses and save
    :param epochs:
    :param train_metrics:
    :param valid_metrics:
    :return:
    """
    exp_dir = config.get_exp_dir()
    for metric in ImbalancedMetrics.get_metrics():
        fig = plt.figure()
        plt.title(f"{metric} plot")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.plot(epochs, [val[metric] for val in train_metrics], "b", marker="o", label=f"train {metric}")
        plt.plot(epochs, [val[metric] for val in valid_metrics], "r", marker="o", label=f"valid {metric}")
        plt.legend()
        plt.savefig(exp_dir / f"{metric}.png")
        plt.close(fig)
