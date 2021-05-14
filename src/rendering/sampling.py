"""
displaying sample images
"""
from typing import Optional

from config import config
from data_processing.metadata_loading import read_metadata
from rendering.images import show_images


def display_random_images(n: int, label: Optional[str] = None, cols: int = 1, title: Optional[str] = None) -> None:
    """
    display random images
    :param n:
    :param label:
    :param cols:
    :param title:
    :return:
    """
    metadata = read_metadata()

    if label is not None:
        image_ids = metadata.loc[metadata["dx"] == label].sample(n).index.to_list()
    else:
        image_ids = metadata.sample(n).index.to_list()

    title_list = None if title is None else metadata.loc[image_ids, title].to_list()

    show_images(image_ids, cols, title_list)


if __name__ == "__main__":
    config.set_data_dir("./data")
    display_random_images(n=3, label=None, title="dx")
