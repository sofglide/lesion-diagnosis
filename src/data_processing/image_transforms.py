"""
Image preparation
"""
from typing import Dict, Optional

import numpy as np
from torchvision import transforms
from torchvision.transforms import Tensor
from torchvision.transforms import functional as F


def get_transforms(
    is_eval: bool = False, distrib_moments: Optional[Dict[str, np.ndarray]] = None, image_size: Optional[int] = None
) -> transforms.Compose:
    """
    Image augmentation transforms if not for evaluation
    :param is_eval: if True, do not apply image augmentation
    :param distrib_moments:
    :param image_size: if model requires a given size
    :return:
    """
    if image_size is not None:
        transforms_list = [SquarePad(), transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    else:
        transforms_list = []
    transforms_list.append(transforms.ToTensor())
    if distrib_moments is not None:
        transforms_list.append(transforms.Normalize(distrib_moments["mean"], distrib_moments["std"]))

    if not is_eval:
        transforms_list.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomErasing(
                    p=0.75,
                    scale=(0.05, 0.5),
                    value=1.0,
                ),
            ]
        )

    return transforms.Compose(transforms_list)


class SquarePad:
    """
    Padding helper for image resizing
    """

    def __call__(self, image: Tensor) -> Tensor:
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, "constant")
