"""
Global configuration parsing
"""
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, List, Tuple


class Config(ConfigParser):
    """
    Global Configuration
    """

    def __init__(self) -> None:
        config_file = Path(__file__).parent / "config.ini"
        super().__init__()
        self.read(config_file)
        self.set("DEFAULT", "loggers", "")

    def add_logger(self, name: str) -> None:
        loggers_str = self.get("DEFAULT", "loggers")
        if loggers_str:
            loggers_str += f",{name}"
        else:
            loggers_str = name
        self.set("DEFAULT", "loggers", loggers_str)

    def get_loggers(self) -> List[str]:
        loggers_str = self.get("DEFAULT", "loggers")
        if not loggers_str:
            return []
        else:
            return loggers_str.split(",")

    def get_dataset_attributes(self) -> Dict[str, Any]:
        dataset_name = self.get("DEFAULT", "dataset_name")
        dataset_files = self.get("DEFAULT", "dataset_files").split(",")
        return {"name": dataset_name, "files": dataset_files}

    def get_out_dir(self) -> Path:
        return Path(self.get("DEFAULT", "out_dir")).absolute()

    def get_tune_results_dir(self) -> Path:
        return Path(self.get("DEFAULT", "tune_results_dir"))

    def get_tune_config_dir(self) -> Path:
        return Path(self.get("DEFAULT", "tune_config_dir")).absolute()

    def set_exp_dir(self, exp_dir: str) -> None:
        exp_path = self.get_out_dir() / exp_dir
        exp_path.mkdir(exist_ok=True, parents=True)
        self.set("DEFAULT", "exp_dir", str(exp_path))

    def get_exp_dir(self) -> Path:
        return Path(self.get("DEFAULT", "exp_dir"))

    def get_data_dir(self) -> Path:
        return Path(self.get("DEFAULT", "data_dir"))

    def get_status_msg(self) -> str:
        return self.get("DEFAULT", "status_message")

    def get_prediction_msg(self) -> str:
        return self.get("DEFAULT", "prediction_message")

    def get_log_file(self) -> Path:
        return self.get_exp_dir() / self.get("DEFAULT", "log_file")

    def get_npy_file(self) -> Path:
        return self.get_exp_dir() / self.get("DEFAULT", "npy_file")

    def get_class_map_file(self) -> Path:
        return self.get_exp_dir() / self.get("DEFAULT", "class_map_file")

    def get_args_file(self) -> Path:
        return self.get_exp_dir() / self.get("DEFAULT", "args_file")

    def get_best_model_file(self) -> Path:
        return self.get_exp_dir() / self.get("DEFAULT", "best_model_file")

    def get_log_format(self) -> str:
        return self.get("DEFAULT", "logging_fmt")

    def set_log_to_file(self, log_to_file: bool) -> None:
        self.set("DEFAULT", "log_to_file", str(log_to_file))

    def get_log_to_file(self) -> bool:
        return self.getboolean("DEFAULT", "log_to_file")

    def get_log_level(self) -> str:
        return self.get("DEFAULT", "log_level")

    def get_test_fraction(self) -> float:
        return self.getfloat("training", "test_fraction")

    def get_test_set_file(self) -> Path:
        return self.get_exp_dir() / self.get("DEFAULT", "test_set_file")

    def get_test_seed(self) -> int:
        return self.getint("training", "test_seed")

    def get_model_size(self, model: str) -> Tuple[int, int]:
        image_size = [int(x) for x in self.get("input_size", model.lower()).split(",")]
        return image_size[0], image_size[1]

    def get_num_classes(self) -> int:
        return self.getint("classification", "num_classes")


config = Config()
