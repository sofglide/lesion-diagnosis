"""
run tuning of experiments.
"""
import json
from typing import Any, Dict

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from config import config as system_config
from experimentation.single_experiment import run_experiment
from utils.experiment_setup import setup_experiment_env
from utils.logging import get_logger

logger = get_logger(__name__)


def run_tune_experiment(tune_config: Dict[str, Any], num_samples: int) -> None:
    """
    entry point for running tune experiment
    :param tune_config:
    :param num_samples:
    :return:
    """
    hyperopt = HyperOptSearch(metric=tune_config["objective_metric"], mode="max")
    analysis = tune.run(
        Trainable,
        config=tune_config,
        search_alg=hyperopt,
        num_samples=num_samples,
        stop={"training_iteration": 1},
        local_dir=system_config.get_tune_results_dir(),
        resources_per_trial={"cpu": 1, "gpu": 1},
    )

    results = analysis.get_best_config(metric=tune_config["objective_metric"], mode="max")
    logger.info(f"best config: {results}")


# noinspection PyAbstractClass
class Trainable(tune.Trainable):
    """
    Tune trainable class
    """

    # pylint: disable=W0223
    # noinspection PyAttributeOutsideInit
    def setup(self, config: Dict[str, Any]) -> None:
        """

        :param config:
        :return:
        """
        self.network = config["network"]
        self.model_params = config["model_params"]
        self.lr_extraction = config["lr_extraction"]
        self.lr_tuning = config["lr_tuning"]
        self.loss = config["loss"]
        self.val_fraction = config["val_fraction"]
        self.batch_size = config["batch_size"]
        self.epochs_extraction = config["epochs_extraction"]
        self.epochs_tuning = config["epochs_tuning"]
        self.objective_metric = config["objective_metric"]
        self.seed = config["seed"]
        self.exp_name = config["exp_name"]
        self.data_dir = config["data_dir"]

    def step(self) -> Dict[str, float]:
        """

        :return:
        """
        setup_experiment_env(
            network=self.network,
            model_params=json.loads(self.model_params),
            lr_extraction=self.lr_extraction,
            lr_tuning=self.lr_tuning,
            loss=self.loss,
            val_fraction=self.val_fraction,
            batch_size=self.batch_size,
            epochs_extraction=self.epochs_extraction,
            epochs_tuning=self.epochs_tuning,
            objective_metric=self.objective_metric,
            seed=self.seed,
            exp_name=self.exp_name,
            data_dir=self.data_dir,
        )

        optimizer_params = {"extraction": {"lr": self.lr_extraction}, "tuning": {"lr": self.lr_tuning}}
        metrics_values = run_experiment(
            val_fraction=self.val_fraction,
            batch_size=self.batch_size,
            network=self.network,
            model=self.model_params,
            optimizer_params=optimizer_params,
            loss=self.loss,
            epochs_extraction=self.epochs_extraction,
            epochs_tuning=self.epochs_tuning,
            objective_metric=self.objective_metric,
            seed=self.seed,
        )

        return metrics_values
