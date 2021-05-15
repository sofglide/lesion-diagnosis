"""
transfer learning from densenet
"""
from typing import Any, Dict, Optional, Tuple

from torch import nn
from torchvision import models

from networks.base_net import BaseNet


class Densenet(BaseNet):
    """
    Densenet model, image size (3, 450, 600)
    """

    @property
    def input_size(self) -> Tuple[int, int]:
        return 224, 224

    @staticmethod
    def _get_model(num_classes: int, _: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, nn.Module]:
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model, model.classifier
