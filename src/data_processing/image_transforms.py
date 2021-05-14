"""
Image preparation
"""
from typing import Dict, Optional, Tuple

import numpy as np
from torchvision import transforms


def get_transforms(
    image_size: Tuple[int, int], is_eval: bool = False, image_mean_std: Optional[Dict[str, np.ndarray]] = None
) -> transforms.Compose:
    """
    Image augmentation transforms if not for evaluation
    :param is_eval: if True, do not apply image augmentation
    :param image_mean_std:
    :param image_size: if model requires a given size
    :return:
    """
    if all(image_size):
        transforms_list = [transforms.Resize(image_size)]
    else:
        transforms_list = []

    if not is_eval:
        transforms_list.extend(
            [
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)
                # transforms.RandomErasing(
                #     p=0.75,
                #     scale=(0.05, 0.5),
                #     value=1.0,
                # ),
            ]
        )

    transforms_list.append(transforms.ToTensor())
    if image_mean_std is not None:
        transforms_list.append(transforms.Normalize(image_mean_std["mean"], image_mean_std["std"]))

    return transforms.Compose(transforms_list)
