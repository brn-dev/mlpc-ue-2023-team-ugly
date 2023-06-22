from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from lib.model.cnn import CNNHyperParameters
from lib.model.cnn1d import CNN1d
from lib.model.cnn2d import CNN2d
from lib.model.fnn import FNN, FNNHyperParameters
from lib.model.hyper_parameters import HyperParameters

@dataclass
class CNNClassifierHyperParameters(HyperParameters):

    in_fnn_hyper_parameters: Optional[FNNHyperParameters]

    cnn1d_embedding_hyper_parameters: Optional[FNNHyperParameters]

    cnn1d_hyper_parameters: CNNHyperParameters

    cnn2d_embedding_hyper_parameters: Optional[FNNHyperParameters]

    cnn2d_hyper_parameters: CNNHyperParameters

    out_fnn_hyper_parameters: FNNHyperParameters


class CNNClassifier(nn.Module):

    def __str__(self):
        return (
            f'CNNClassifier with {_count_parameters(self)} parameters: \n'
            f'in_fnn:            {_count_parameters(self.in_fnn):>9}, \n'
            f'conv1d_embedding:  {_count_parameters(self.cnn1d_embedding):>9}, \n'
            f'conv2d_embedding:  {_count_parameters(self.cnn2d_embedding):>9}, \n'
            f'conv1d_layers:     {_count_parameters(self.cnn1d):>9}, \n'
            f'conv2d_layers:     {_count_parameters(self.cnn2d):>9}, \n'
            f'out_fnn:           {_count_parameters(self.out_fnn):>9}, \n'
        )


    def __init__(self, hyper_parameters: CNNClassifierHyperParameters):
        super().__init__()

        if hyper_parameters.in_fnn_hyper_parameters is not None:
            self.in_fnn = FNN(hyper_parameters.in_fnn_hyper_parameters)
        else:
            self.in_fnn = nn.Identity()

        if hyper_parameters.cnn1d_embedding_hyper_parameters is not None:
            self.cnn1d_embedding = FNN(hyper_parameters.cnn1d_embedding_hyper_parameters)
        else:
            self.cnn1d_embedding = nn.Identity()

        if hyper_parameters.cnn2d_embedding_hyper_parameters is not None:
            self.cnn2d_embedding = FNN(hyper_parameters.cnn2d_embedding_hyper_parameters)
        else:
            self.cnn2d_embedding = nn.Identity()

        self.cnn1d = CNN1d(hyper_parameters.cnn1d_hyper_parameters)
        self.cnn2d = CNN2d(hyper_parameters.cnn2d_hyper_parameters)

        # TODO: hyper parameter
        self.combined_norm = nn.BatchNorm1d(hyper_parameters.out_fnn_hyper_parameters.in_features)

        self.out_fnn = FNN(hyper_parameters.out_fnn_hyper_parameters)


    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # n_sequences, sequence_length, in_features = sequences.shape

        sequences = self.in_fnn(sequences)

        embedded_for_1d = self.cnn1d_embedding.forward(sequences)
        embedded_for_2d = self.cnn2d_embedding.forward(sequences)

        convolved_through_1d = self.cnn1d.forward(
            embedded_for_1d.swapaxes(1, 2)
        ).swapaxes(1, 2)

        convolved_through_2d = to_1d(self.cnn2d.forward(
            to_2d(embedded_for_2d.swapaxes(1, 2))
        )).swapaxes(1, 2)

        combined = self.combined_norm(
            torch.concat((convolved_through_1d, convolved_through_2d), dim=-1).swapaxes(1, 2)
        ).swapaxes(1, 2)

        out = self.out_fnn.forward(combined)

        return out


def to_2d(one_d_sequences: torch.Tensor, sequence_length_3rd=False) -> torch.Tensor:
    # one_d_sequence.shape = n_sequences, features, sequence_length

    two_d = one_d_sequences[:, None, :, :]

    # two_d.shape = n_sequences, 1, features, sequence_length

    if sequence_length_3rd:
        two_d = two_d.swapaxes(2, 3)

    return two_d


def to_1d(two_d: torch.Tensor, sequence_length_3rd=False) -> torch.Tensor:
    if sequence_length_3rd:
        two_d = two_d.swapaxes(2, 3)

    # two_d.shape = n_sequences, 1, features, sequence_length

    one_d_sequences = two_d[:, 0, :, :]  # shape = n_sequences, features, sequence_length
    return one_d_sequences


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
