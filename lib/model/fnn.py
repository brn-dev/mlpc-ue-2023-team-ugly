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

class FNN(nn.Module):

    layers: nn.Sequential

    def __init__(self, fnn_hyper_parameters: FNNHyperParameters):
        super().__init__()

        layers: list[nn.Module] = []

        in_features = fnn_hyper_parameters.in_features
        for out_features in fnn_hyper_parameters.layers_out_features:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(fnn_hyper_parameters.activation_provider())

            in_features = out_features

        self.layers = nn.Sequential(*layers[:-1])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
