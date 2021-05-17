"""
data exploration
"""
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from config import config


def show_images(image_ids: List[str], cols: int = 1, titles: Optional[List[str]] = None) -> None:
    """
    Display multiple images arranged as a table
    :param image_ids:
    :param cols:
    :param titles:
    :return:
    """
    assert (titles is None) or (len(image_ids) == len(titles))
    n_images = len(image_ids)
    if titles is None:
        titles = image_ids
    fig = plt.figure()
    for n, (image_id, title) in enumerate(zip(image_ids, titles), 1):
        image = load_image(image_id)
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n)
        if image.ndim == 2:
            plt.gray()

        plt.imshow(image)

        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def load_image(image_id: str) -> np.ndarray:
    """
    Load image as numpy array
    :param image_id:
    :return:
    """
    data_dir = config.get_data_dir()
    image_path = (data_dir / image_id).with_suffix(".jpg")
    return np.array(Image.open(image_path))
