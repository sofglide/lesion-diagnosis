"""
Dataloader generation
"""
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from data_processing.data_splitting import split_data


def get_data_loaders(
    batch_size: int,
    val_fraction: float,
    image_size: Tuple[int, int],
    image_copies: int = 1,
    random_seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """

    :param batch_size:
    :param val_fraction:
    :param image_size:
    :param image_copies:
    :param random_seed:
    :return:
    """
    train_set, val_set = split_data(
        val_fraction=val_fraction, image_size=image_size, image_copies=image_copies, random_seed=random_seed
    )

    train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader: DataLoader = DataLoader(val_set, batch_size=batch_size, num_workers=2, drop_last=True)

    return train_loader, val_loader, train_set.num_classes
