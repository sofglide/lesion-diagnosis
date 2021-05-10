"""
transfer learning from densenet
"""
from torch import Tensor, nn
from torchvision import models


class Densenet(nn.Module):
    """
    Densenet model, image size (3, 450, 600)
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """

        :param num_classes:
        """
        super().__init__()

        self.input_size = 224

        self.model = models.densenet121(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward propagation
        :param x:
        :return:
        """
        return self.model(x)
