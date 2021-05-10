"""
Image manipulation
"""
import os
from pathlib import Path
from typing import Dict, List


def get_image_paths_dict(
    data_dir: Path,
) -> Dict[str, Path]:
    """
    Create and return dict that maps image IDs to image paths
    :param data_dir:
    :return:
    """
    image_paths = _read_image_paths(data_dir)

    return {image_path.stem: image_path for image_path in image_paths}


def _read_image_paths(data_dir: Path) -> List[Path]:
    """
    Read image paths from data directory
    :param data_dir:
    :return:
    """
    image_extension_pattern = "*.jpg"
    image_paths = sorted(y for x in os.walk(data_dir) for y in Path(x[0]).glob(image_extension_pattern))
    return image_paths
