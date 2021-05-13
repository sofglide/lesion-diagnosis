"""
Image dataset class
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchdata as td
from PIL import Image

from config import config
from data_processing.class_mapping import get_class_map_dict
from data_processing.image_transforms import get_transforms


class HAM10000(td.Dataset):
    """
    Images dataset class
    uses a cached dataset containing the images before transform
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        sampling_list: List[str],
        image_size: Tuple[int, int],
        is_eval: bool = False,
        image_mean_std: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Dataset constructor uses cached dataset before transform
        :param metadata:
        :param sampling_list: list of image IDs to use
        :param is_eval: if for evaluation, no data augmentation
        :param image_mean_std: training image distribution mean and std
        :param image_size: image size for network input
        """
        super().__init__()

        self.sampling_list = sampling_list

        self.metadata = metadata.loc[sampling_list]
        self.transforms = get_transforms(image_size=image_size, is_eval=is_eval, image_mean_std=image_mean_std)
        self.images = _HAM10000(self.sampling_list).cache().map(self.transforms)

    @property
    def class_map_dict(self) -> Dict[str, int]:
        return get_class_map_dict(self.metadata["dx"])

    @property
    def num_classes(self) -> int:
        """
        Get number of classes
        :return:
        """
        return len(self.metadata["dx"].cat.categories)

    def __len__(self) -> int:
        """
        Get size of dataset
        :return:
        """
        return len(self.sampling_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item.
        :param index:
        :return: tuple with image and label
        """
        image_id = self.sampling_list[index]
        label = torch.tensor(self.metadata["dx"].cat.codes.loc[image_id], dtype=torch.long)  # pylint: disable=E1102
        img = self.images[index]
        return img, label


class _HAM10000(td.Dataset):
    """
    _HAM10000 dataset.
    cached images before transforms
    """

    def __init__(self, sampling_list: List[str]) -> None:
        """

        :param sampling_list:
        """
        super().__init__()

        self.sampling_list = sampling_list
        self.data_dir = config.get_data_dir()

    def __len__(self) -> int:
        """
        Get size of dataset
        :return:
        """
        return len(self.sampling_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get item
        :param index:
        :return:
        """
        image_id = self.sampling_list[index]
        img = Image.open((self.data_dir / image_id).with_suffix(".jpg"))

        return img
