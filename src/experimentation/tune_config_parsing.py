"""
parsing tune config from input params
"""
import json
from typing import Any, Dict

from ray import tune


def tune_parse_config_dict(config_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse tune config from input
    :param config_raw:
    :return:
    """
    parsed_config = dict()
    parsed_config["network"] = tune.choice(config_raw["network"])
    parsed_config["model_params"] = tune.choice([json.dumps(p) for p in config_raw["model_params"]])
    parsed_config["lr"] = tune.loguniform(*config_raw["lr"])
    parsed_config["loss"] = config_raw["loss"]
    parsed_config["batch_size"] = tune.choice(config_raw["batch_size"])
    parsed_config["val_fraction"] = config_raw["val_fraction"]
    parsed_config["num_epochs"] = config_raw["num_epochs"]
    parsed_config["objective_metric"] = config_raw["objective_metric"]
    parsed_config["early_stop_count"] = config_raw["early_stop_count"]
    parsed_config["early_stop_ratio"] = config_raw["early_stop_ratio"]
    parsed_config["seed"] = config_raw["seed"]
    return parsed_config
