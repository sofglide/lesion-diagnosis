"""
Train classifier on HAM10000 dataset.
"""
import json
import shutil
from pathlib import Path

import click
from kaggle import KaggleApi

from config import config
from experimentation.single_experiment import run_experiment
from experimentation.tune_config_parsing import tune_parse_config_dict
from experimentation.tune_experiment import run_tune_experiment
from utils.experiment_setup import setup_experiment_env
from utils.logging import get_logger

logger = get_logger(__name__)

model_list = ["SimpleCNN", "Resnet", "Densenet", "Hybrid"]


@click.group()
def execute() -> None:
    pass


@execute.command(help="Run a single experiment")
@click.option("--data-dir", type=click.Path(), default="./data", help="path to data")
@click.option("--val-fraction", type=click.FLOAT, default=0.2, help="fraction of dataset to use for validation")
@click.option("--exp-name", type=click.STRING, default="baseline", help="name of experiment")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO"]), default="INFO", help="log-level to use")
@click.option("--batch-size", type=click.INT, default=32, help="batch-size to use")
@click.option("--network", type=click.Choice(model_list), default="SimpleCNN", help="network architecture")
@click.option("--model-params", type=click.STRING, default="{}", help="network parameters")
@click.option("--lr", type=click.FLOAT, default=0.001, help="optimizer learning rate")
@click.option("--loss", type=click.STRING, default='{"function": "cross_entropy"}', help="loss function")
@click.option("--num-epochs", type=click.INT, default=10, help="number of training epochs")
@click.option(
    "--objective-metric", type=click.Choice(["f1_score", "mcc"]), default="f1_score", help="metric to maximize"
)
@click.option("--early-stop-count", type=click.INT, default=3, help="validation loss increase count to stop experiment")
@click.option(
    "--early-stop-ratio", type=click.FLOAT, default=0.1, help="validation loss increase ratio to increment counter"
)
@click.option("--seed", type=click.INT, default=0, help="random seed for train/valid split")
def single_experiment(
    data_dir: str,
    val_fraction: float,
    exp_name: str,
    log_level: str,
    batch_size: int,
    network: str,
    model_params: str,
    lr: float,
    loss: str,
    num_epochs: int,
    objective_metric: str,
    early_stop_count: int,
    early_stop_ratio: float,
    seed: int,
) -> None:
    """
    run a single experiment
    :param data_dir:
    :param val_fraction:
    :param exp_name:
    :param log_level:
    :param batch_size:
    :param network:
    :param model_params:
    :param lr:
    :param loss:
    :param num_epochs:
    :param objective_metric:
    :param early_stop_count:
    :param early_stop_ratio:
    :param seed:
    :return:
    """
    # pylint: disable=too-many-arguments
    model_params_dict = json.loads(model_params)
    loss_dict = json.loads(loss)

    setup_experiment_env(
        exp_name=exp_name,
        data_dir=data_dir,
        log_level=log_level,
        val_fraction=val_fraction,
        batch_size=batch_size,
        network=network,
        model_params=model_params_dict,
        lr=lr,
        loss=loss_dict,
        num_epochs=num_epochs,
        objective_metric=objective_metric,
        early_stop_count=early_stop_count,
        early_stop_ratio=early_stop_ratio,
        seed=seed,
    )

    run_experiment(
        val_fraction=val_fraction,
        batch_size=batch_size,
        network=network,
        model=model_params_dict,
        optimizer_params={"lr": lr},
        loss=loss_dict,
        num_epochs=num_epochs,
        objective_metric=objective_metric,
        early_stop_count=early_stop_count,
        early_stop_ratio=early_stop_ratio,
        seed=seed,
    )


@execute.command(help="Run a ray tune driven set of experiments")
@click.option("--exp-name", type=click.STRING, default="tune_baseline", help="name of experiment")
@click.option("--config-file", type=click.Path(), help="path to tune config file")
@click.option("--num-samples", type=click.INT, default=1, help="number of tuning samples")
@click.option("--data-dir", type=click.Path(), default="./data", help="path to data")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO"]), default="INFO", help="log-level to use")
def tune_experiment(exp_name: str, config_file: str, num_samples: int, data_dir: str, log_level: str) -> None:
    """
    Run tuning experiment
    :param exp_name:
    :param config_file:
    :param num_samples:
    :param data_dir:
    :param log_level:
    :return:
    """
    with open(config.get_tune_config_dir() / config_file, "r") as fp:
        config_raw = json.load(fp)
    tune_config = tune_parse_config_dict(config_raw)
    tune_config["exp_name"] = exp_name
    tune_config["data_dir"] = str(Path(data_dir).absolute())
    tune_config["log_level"] = log_level
    run_tune_experiment(tune_config, num_samples)


@execute.command(help="Download and prepare data")
@click.option("--data-dir", type=click.Path(), default="./data", help="path to data")
def download_data(data_dir: str) -> None:
    """
    Download data to data directory
    :param data_dir:
    :return:
    """
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(exist_ok=True)

    dataset = config.get_dataset_attributes()
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_cli(dataset["name"], path=data_dir_path, unzip=True)
    selected_files = [data_dir_path / dataset_file for dataset_file in dataset["files"]]

    for obj in data_dir_path.glob("*"):
        if obj not in selected_files:
            if obj.is_dir():
                shutil.rmtree(obj)
            else:
                obj.unlink()

    for obj in selected_files:
        if obj.is_dir():
            for image in obj.glob("*.jpg"):
                image.rename(image.parents[1] / image.name)
            obj.rmdir()


if __name__ == "__main__":
    execute()
