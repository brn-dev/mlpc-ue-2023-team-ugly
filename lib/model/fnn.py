from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters


@dataclass
class FNNHyperParameters(HyperParameters):
    in_features: int
    layers_out_features: list[int]
    activation_provider: Callable[[], nn.Module]
    dropout: float
    end_with_activation: bool = False

class FNN(nn.Module):

    layers: nn.Sequential

    def __init__(self, hyper_parameters: FNNHyperParameters):
        super().__init__()

        layers: list[nn.Module] = []

        in_features = hyper_parameters.in_features
        for out_features in hyper_parameters.layers_out_features:
            layers.append(nn.Dropout(p=hyper_parameters.dropout))
            layers.append(nn.Linear(in_features, out_features))
            layers.append(hyper_parameters.activation_provider())

            in_features = out_features

        if not hyper_parameters.end_with_activation:
            self.layers = nn.Sequential(*layers[:-1])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
