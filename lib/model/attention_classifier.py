from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from lib.model.fnn import FNN, FNNHyperParameters
from lib.model.multihead_self_attention import MultiheadSelfAttentionHyperParameters, MultiheadSelfAttention
from lib.model.positional_encoding import PositionalEncoding

ActivationProvider = Callable[[], nn.Module]


@dataclass
class AttentionClassifierHyperParameters(MultiheadSelfAttentionHyperParameters):
    in_features: int
    out_features: int

    attention_window_size: int
    d_model: int
    num_heads: int
    attention_stack_size: int
    attention_stack_activation_provider: ActivationProvider
    attention_dropout: float

    in_linear_hidden_out_features: list[int]
    out_linear_hidden_out_features: list[int]
    linear_activation_provider: ActivationProvider
    linear_dropout: float


class AttentionClassifier(nn.Module):
    def __init__(self, hyper_parameters: AttentionClassifierHyperParameters, batch_first: bool):
        super().__init__()

        self.batch_first = batch_first

        self.attention_window_size = hyper_parameters.attention_window_size
        self.stack_size = hyper_parameters.attention_stack_size
        self.out_features = hyper_parameters.out_features

        self.in_fnn = FNN(FNNHyperParameters(
            in_features=hyper_parameters.in_features,
            layers_out_features=hyper_parameters.in_linear_hidden_out_features + [hyper_parameters.d_model],
            activation_provider=hyper_parameters.linear_activation_provider,
            dropout=hyper_parameters.linear_dropout
        ))

        self.positional_encoder = PositionalEncoding(
            d_model=hyper_parameters.d_model,
            max_len=1024,
            batch_first=False,
        )

        attention_stack_modules: list[nn.Module] = []

        for _ in range(hyper_parameters.attention_stack_size - 1):
            attention_stack_modules.append(MultiheadSelfAttention(hyper_parameters, batch_first=False))
            attention_stack_modules.append(hyper_parameters.attention_stack_activation_provider())

        if hyper_parameters.attention_stack_size > 0:
            attention_stack_modules.append(MultiheadSelfAttention(hyper_parameters, batch_first=False))

        self.attention_stack = nn.Sequential(*attention_stack_modules)

        # TODO: norm?
        # self.norm = nn.LayerNorm(
        #     hyper_parameters.d_model
        # )

        self.out_fnn = FNN(FNNHyperParameters(
            in_features=hyper_parameters.d_model,
            layers_out_features=hyper_parameters.out_linear_hidden_out_features + [hyper_parameters.out_features],
            activation_provider=hyper_parameters.linear_activation_provider,
            dropout=hyper_parameters.linear_dropout
        ))

    def __str__(self):
        return (f'AttentionClassifier with {_count_parameters(self)} parameters, '
                f'in_fnn: {_count_parameters(self.in_fnn)}, '
                f'attention_stack: {_count_parameters(self.attention_stack)}, '
                f'out_fnn: {_count_parameters(self.out_fnn)}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using batch second internally
        if self.batch_first:
            x = torch.swapaxes(x, 0, 1)

        sequence_length, n_sequences, dimensions = x.shape

        x = torch.reshape(x, (self.attention_window_size, -1, dimensions))

        embedded = self.in_fnn.forward(x)

        embedded_with_pos = self.positional_encoder.forward(embedded)

        attention_out = self.attention_stack.forward(embedded_with_pos)

        out = self.out_fnn.forward(attention_out)

        out = torch.reshape(out, (sequence_length, n_sequences, self.out_features))

        if self.batch_first:
            out = torch.swapaxes(out, 0, 1)

        return out

def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
