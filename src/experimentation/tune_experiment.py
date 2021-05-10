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
        self.lr = config["lr"]
        self.loss = config["loss"]
        self.val_fraction = config["val_fraction"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.objective_metric = config["objective_metric"]
        self.early_stop_count = config["early_stop_count"]
        self.early_stop_ratio = config["early_stop_ratio"]
        self.seed = config["seed"]
        self.exp_name = config["exp_name"]
        self.data_dir = config["data_dir"]
        self.log_level = config["log_level"]

    def step(self) -> Dict[str, float]:
        """

        :return:
        """
        setup_experiment_env(
            network=self.network,
            model_params=json.loads(self.model_params),
            lr=self.lr,
            loss=self.loss,
            val_fraction=self.val_fraction,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            objective_metric=self.objective_metric,
            early_stop_count=self.early_stop_count,
            early_stop_ratio=self.early_stop_ratio,
            seed=self.seed,
            exp_name=self.exp_name,
            data_dir=self.data_dir,
            log_level=self.log_level,
        )

        metrics_values = run_experiment(
            val_fraction=self.val_fraction,
            batch_size=self.batch_size,
            network=self.network,
            model=self.model_params,
            optimizer_params={"lr": self.lr},
            loss=self.loss,
            num_epochs=self.num_epochs,
            objective_metric=self.objective_metric,
            early_stop_count=self.early_stop_count,
            early_stop_ratio=self.early_stop_ratio,
            seed=self.seed,
        )

        return metrics_values
