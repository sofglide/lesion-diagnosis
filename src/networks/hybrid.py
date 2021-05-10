"""
Multiple pretrained models
"""
import itertools
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torchvision import models

INPUT_SIZE = 224
DEFAULT_EMBEDDDING_DIM = 512
DEFAULT_HIDDEN_LAYERS = 3


class Hybrid(nn.Module):
    """
    Resnet model, image size (3, 450, 600)
    """

    def __init__(self, num_classes: int = 1000, params: Optional[Dict[str, Any]] = None) -> None:
        """

        :param num_classes:
        """
        super().__init__()

        if params is None:
            params = dict()
        embedding_dim = params.get("embedding_dim", DEFAULT_EMBEDDDING_DIM)
        hidden_layers = params.get("hidden_layers", DEFAULT_HIDDEN_LAYERS)

        self.input_size = INPUT_SIZE
        self.embedders = _get_embedders(embedding_dim=embedding_dim)
        self.fusion_layers = _get_fusion_layers(len(self.embedders) * embedding_dim, num_classes, hidden_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation
        :param x:
        :return:
        """
        x = torch.cat([embedder(x) for embedder in self.embedders.values()], dim=1)
        x = self.fusion_layers(x)
        return x


def _get_embedders(embedding_dim: int) -> nn.ModuleDict:
    """
    Get pretrained embedders
    :param embedding_dim:
    :return:
    """
    resnet = models.resnet34(pretrained=True)
    densenet = models.densenet121(pretrained=True)

    for param in itertools.chain(resnet.parameters(), densenet.parameters()):
        param.requires_grad = False

    densenet_num_features = densenet.classifier.in_features
    densenet.classifier = nn.Sequential(
        OrderedDict(
            [("densenet_final_fc", nn.Linear(densenet_num_features, embedding_dim)), ("densenet_final_relu", nn.ReLU())]
        )
    )

    resnet_num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        OrderedDict(
            [("resnet_final_fc", nn.Linear(resnet_num_features, embedding_dim)), ("resnet_final_relu", nn.ReLU())]
        )
    )

    return nn.ModuleDict({"densenet": densenet, "resnet": resnet})


def _get_fusion_layers(input_dim: int, output_dim: int, hidden_layers: int) -> nn.Module:
    """
    final fully connected layers
    :param input_dim:
    :param output_dim:
    :param hidden_layers:
    :return:
    """
    layers_size = np.geomspace(input_dim, output_dim, hidden_layers + 2).astype(int)
    layers: List[Tuple[str, nn.Module]] = []
    for ind, current_input_dim in enumerate(layers_size[:-2]):
        layers.append((f"fusion_fc_{ind}", nn.Linear(current_input_dim, layers_size[ind + 1])))
        layers.append((f"fusion_relu_{ind}", nn.ReLU()))
    layers.append(("output_fc", nn.Linear(layers_size[-2], output_dim)))

    return nn.Sequential(OrderedDict(layers))
