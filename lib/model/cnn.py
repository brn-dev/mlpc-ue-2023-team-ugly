import abc
from dataclasses import dataclass
from typing import Union, Callable

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters


IntOr2iTuple = Union[int, tuple[int, int]]


@dataclass
class ConvHyperParameters(HyperParameters):
    out_channels: int
    kernel_size: IntOr2iTuple
    stride: IntOr2iTuple
    dilation: IntOr2iTuple
    groups: IntOr2iTuple
    bias: bool
    padding: Union[str, IntOr2iTuple]
    padding_mode: str = 'zeros'


@dataclass
class CNNHyperParameters(HyperParameters):

    in_channels: int

    layers_hyper_parameters: list[ConvHyperParameters]

    normalize_before: bool
    normalize_after: bool

    activation_provider: Callable[[], nn.Module]

    dropout: float


class CNN(nn.Module, abc.ABC):

    layers: nn.Sequential

    def __init__(self, hyper_parameters: CNNHyperParameters):
        super().__init__()
        layers: list[nn.Module] = []

        in_channels = hyper_parameters.in_channels

        for layer_hyper_parameters in hyper_parameters.layers_hyper_parameters:
            out_channels = layer_hyper_parameters.out_channels

            if hyper_parameters.normalize_before:
                layers.append(self._create_batch_norm(in_channels))

            if hyper_parameters.dropout > 0:
                layers.append(nn.Dropout(hyper_parameters.dropout))

            layers.append(self._create_conv(in_channels, layer_hyper_parameters))

            if hyper_parameters.normalize_after:
                layers.append(self._create_batch_norm(out_channels))

            # TODO: Dropout
            # TODO: Pool
            layers.append(hyper_parameters.activation_provider())

            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # n_sequences, sequence_length, in_features = x.shape

        return self.layers.forward(x)

    @abc.abstractmethod
    def _create_conv(self, in_channels: int, conv_hyper_parameters: ConvHyperParameters) -> nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_batch_norm(self, num_features: int) -> nn.Module:
        raise NotImplementedError

