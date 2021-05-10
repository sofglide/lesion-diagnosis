"""
Image dataset class
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchdata as td
from PIL import Image

from data_processing.class_mapping import get_class_map_dict, get_class_weights
from data_processing.image_processing import get_image_paths_dict
from data_processing.image_transforms import get_transforms
from data_processing.metadata_loading import read_metadata


class HAM10000(td.Dataset):
    """
    Images dataset class
    uses a cached dataset containing the images before transform
    """

    def __init__(
        self,
        data_dir: Path,
        sampling_list: List[str],
        is_eval: bool = False,
        distrib_moments: Optional[Dict[str, np.ndarray]] = None,
        image_size: Optional[int] = None,
    ) -> None:
        """
        Dataset constructor uses cached dataset before transform
        :param data_dir: path to images and metadata file
        :param sampling_list: list of image IDs to use
        :param is_eval: if for evaluation, no data augmentation
        :param distrib_moments: training image distribution mean and std
        :param image_size: image size for network input
        """
        super().__init__()

        self.data_dir = data_dir
        self.sampling_list = sampling_list
        self.image_paths_dict = get_image_paths_dict(self.data_dir)
        self.metadata = read_metadata(self.data_dir)
        self.class_map_dict = get_class_map_dict(self.data_dir)
        self.transforms = get_transforms(is_eval=is_eval, distrib_moments=distrib_moments, image_size=image_size)
        self.images = _HAM10000(self.sampling_list, self.image_paths_dict).cache().map(self.transforms)

    def get_labels(self) -> List[str]:
        """
        Get labels of dataset and return them as list
        :return:
        """
        return self.metadata.loc[self.sampling_list, "dx"].tolist()

    def get_weights(self) -> np.ndarray:
        """
        Compute class weight for imbalanced dataset
        :return:
        """
        return get_class_weights(self.get_labels(), self.get_num_classes(), self.class_map_dict)

    def get_num_classes(self) -> int:
        """
        Get number of classes
        :return:
        """
        return len(self.class_map_dict)

    def __len__(self) -> int:
        """
        Get size of dataset
        :return:
        """
        return len(self.sampling_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get item.
        :param index:
        :return: tuple with image and label
        """
        image_id = self.sampling_list[index]
        label = self.class_map_dict[self.metadata.loc[image_id, "dx"]]
        img = self.images[index]
        return img, label


class _HAM10000(td.Dataset):
    """
    _HAM10000 dataset.
    cached images before transforms
    """

    def __init__(
        self,
        sampling_list: List[str],
        image_paths_dict: Dict[str, Path],
    ) -> None:
        """

        :param sampling_list:
        :param image_paths_dict:
        """
        super().__init__()

        self.sampling_list = sampling_list
        self.image_paths_dict = image_paths_dict

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
        img = Image.open(self.image_paths_dict.get(image_id))

        return img
