import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from config import NUM_CLASSES


def build_model(pretrained: bool = True):
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None

    model = efficientnet_b2(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )

    return model
