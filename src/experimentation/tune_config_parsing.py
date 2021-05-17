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
    for lr in ["lr_extraction", "lr_tuning"]:
        if len(config_raw[lr]) == 1:
            parsed_config[lr] = config_raw[lr][0]
        else:
            parsed_config[lr] = tune.loguniform(*config_raw[lr])
    parsed_config["loss"] = config_raw["loss"]
    parsed_config["batch_size"] = tune.choice(config_raw["batch_size"])
    parsed_config["val_fraction"] = config_raw["val_fraction"]
    parsed_config["epochs_extraction"] = config_raw["epochs_extraction"]
    parsed_config["epochs_tuning"] = config_raw["epochs_tuning"]
    parsed_config["objective_metric"] = config_raw["objective_metric"]
    parsed_config["seed"] = config_raw["seed"]
    return parsed_config
