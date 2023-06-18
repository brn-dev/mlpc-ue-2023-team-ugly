from dataclasses import dataclass

import torch
from torch import nn

from lib.model.attention_classifier import AttentionClassifierHyperParameters, AttentionClassifier
from lib.model.cnn_classifier import CNNClassifierHyperParameters, CNNClassifier
from lib.model.hyper_parameters import HyperParameters


@dataclass
class CombinedClassifierHyperParameters(HyperParameters):
    cnn_classifier_hyper_parameters: CNNClassifierHyperParameters
    attention_classifier_hyper_parameters: AttentionClassifierHyperParameters


class CombinedClassifier(nn.Module):

    def __init__(self, hyper_parameters: CombinedClassifierHyperParameters):
        super().__init__()

        self.cnn = CNNClassifier(hyper_parameters.cnn_classifier_hyper_parameters)

        self.pre_attention_norm = nn.LayerNorm(hyper_parameters.attention_classifier_hyper_parameters.in_features)
        self.pre_attention_activation = nn.LeakyReLU()

        self.attention = AttentionClassifier(
            hyper_parameters.attention_classifier_hyper_parameters,
            batch_first=True
        )


    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        # n_sequences, sequence_length, in_features = input_sequences.shape

        cnn_out = self.cnn(input_sequences)
        cnn_out = self.pre_attention_activation(self.pre_attention_norm(cnn_out))

        attention_out = self.attention(cnn_out)

        return attention_out


    def __str__(self):
        return (
            f'CombinedClassifier with {_count_parameters(self)} parameters: \n\n'
            f'{self.cnn} \n\n'
            f'{self.attention} \n'
        )


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
