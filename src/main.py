"""
Train classifier on HAM10000 dataset.
"""
import json
import shutil

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
@click.option("--val-fraction", type=click.FLOAT, default=0.2, help="fraction of dataset to use for validation")
@click.option("--batch-size", type=click.INT, default=32, help="batch-size to use")
@click.option("--network", type=click.Choice(model_list), default="SimpleCNN", help="network architecture")
@click.option("--model-params", type=click.STRING, default="{}", help="network parameters")
@click.option("--lr-extraction", type=click.FLOAT, default=0.001, help="optimizer feature extraction learning rate")
@click.option("--lr-tuning", type=click.FLOAT, default=0.0001, help="optimizer fine tuning learning rate")
@click.option("--loss", type=click.STRING, default='{"function": "cross_entropy"}', help="loss function")
@click.option("--epochs-extraction", type=click.INT, default=2, help="training epochs for classification layer")
@click.option("--epochs-tuning", type=click.INT, default=10, help="training epochs for fine tuning")
@click.option(
    "--objective-metric", type=click.Choice(["f1_score", "mcc"]), default="f1_score", help="metric to maximize"
)
@click.option("--seed", type=click.INT, default=0, help="random seed for train/valid split")
def single_experiment(
    val_fraction: float,
    batch_size: int,
    network: str,
    model_params: str,
    lr_extraction: float,
    lr_tuning: float,
    loss: str,
    epochs_extraction: int,
    epochs_tuning: int,
    objective_metric: str,
    seed: int,
) -> None:
    """
    run a single experiment
    :param val_fraction:
    :param batch_size:
    :param network:
    :param model_params:
    :param lr_extraction:
    :param lr_tuning:
    :param loss:
    :param epochs_extraction:
    :param epochs_tuning:
    :param objective_metric:
    :param seed:
    :return:
    """
    # pylint: disable=too-many-arguments
    model_params_dict = json.loads(model_params)
    loss_dict = json.loads(loss)

    setup_experiment_env(
        val_fraction=val_fraction,
        batch_size=batch_size,
        network=network,
        model_params=model_params_dict,
        lr_extraction=lr_extraction,
        lr_tuning=lr_tuning,
        loss=loss_dict,
        epochs_extraction=epochs_extraction,
        epochs_tuning=epochs_tuning,
        objective_metric=objective_metric,
        seed=seed,
    )

    optimizer_params = {"extraction": {"lr": lr_extraction}, "tuning": {"lr": lr_tuning}}
    run_experiment(
        val_fraction=val_fraction,
        batch_size=batch_size,
        network=network,
        model=model_params_dict,
        optimizer_params=optimizer_params,
        loss=loss_dict,
        epochs_extraction=epochs_extraction,
        epochs_tuning=epochs_tuning,
        objective_metric=objective_metric,
        seed=seed,
    )


@execute.command(help="Run a ray tune driven set of experiments")
@click.option("--config-file", type=click.Path(), help="path to tune config file")
@click.option("--num-samples", type=click.INT, default=1, help="number of tuning samples")
def tune_experiment(config_file: str, num_samples: int) -> None:
    """
    Run tuning experiment
    :param config_file:
    :param num_samples:
    :return:
    """
    with open(config.get_tune_config_dir() / config_file, "r") as fp:
        config_raw = json.load(fp)
    tune_config = tune_parse_config_dict(config_raw)
    run_tune_experiment(tune_config, num_samples)


@execute.command(help="Download and prepare data")
def download_data() -> None:
    """
    Download data to data directory
    :return:
    """
    data_dir_path = config.get_data_dir()
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
