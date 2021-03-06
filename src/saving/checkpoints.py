"""
checkpoints saving
"""
import shutil
from pathlib import Path
from typing import Any, Dict

import torch

from config import config


def save_checkpoint(
    state: Dict[str, Any],
    target_dir: Path,
    file_name: str = "checkpoint.pth.tar",
    backup_as_best: bool = False,
) -> None:
    """
    Save checkpoint to disk
    :param state:
    :param target_dir:
    :param file_name:
    :param backup_as_best:
    :return:
    """
    target_model_path = target_dir / file_name

    target_dir.mkdir(exist_ok=True)
    torch.save(state, target_model_path)
    if backup_as_best:
        best_model_path = config.get_best_model_file()
        shutil.copyfile(target_model_path, best_model_path)


def load_checkpoint(target_dir: Path, file_name: str = "checkpoint.pth.tar") -> Dict[str, Any]:
    """
    load checkpint
    :param target_dir:
    :param file_name:
    :return:
    """
    target_model_path = target_dir / file_name
    checkpoint = torch.load(target_model_path)

    return checkpoint


def load_best_checkpoint() -> Dict[str, Any]:
    """
    load best checkpoint
    :return:
    """
    best_model_path = config.get_best_model_file()
    return load_checkpoint(best_model_path.parent, best_model_path.name)
