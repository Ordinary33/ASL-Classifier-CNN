from torchvision import models
from torch import nn


class ASLCNNModel(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLCNNModel, self).__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
