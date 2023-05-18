from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from lib.model.fnn import FNN, FNNHyperParameters
from lib.model.hyper_parameters import HyperParameters
from lib.model.positional_encoding import PositionalEncoding


@dataclass
class AttentionClassifierHyperParameters(HyperParameters):
    in_features: int
    out_features: int

    self_attention: bool
    d_model: int
    num_heads: int
    stack_size: int
    attention_dropout: float

    in_linear_hidden_out_features: list[int]
    out_linear_hidden_out_features: list[int]
    linear_activation_provider: Callable[[], nn.Module]
    linear_dropout: float


class AttentionClassifier(nn.Module):
    def __init__(self, hyper_parameters: AttentionClassifierHyperParameters):
        super().__init__()
        # TODO: dropout

        self.self_attention = hyper_parameters.self_attention
        self.stack_size = hyper_parameters.stack_size

        self.in_fnn = FNN(FNNHyperParameters(
            in_features=hyper_parameters.in_features,
            layers_out_features=hyper_parameters.in_linear_hidden_out_features + [hyper_parameters.d_model],
            activation_provider=hyper_parameters.linear_activation_provider,
            dropout=hyper_parameters.linear_dropout
        ))
        self.positional_encoder = PositionalEncoding(
            d_model=hyper_parameters.d_model,
            max_len=1024
        )
        self.attention_stack = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hyper_parameters.d_model,
                num_heads=hyper_parameters.num_heads,
                dropout=hyper_parameters.attention_dropout
            )
            for _
            in range(hyper_parameters.stack_size)
        ])
        # self.norm = nn.LayerNorm(
        #     hyper_parameters.d_model
        # )
        self.out_fnn = FNN(FNNHyperParameters(
            in_features=hyper_parameters.d_model,
            layers_out_features=hyper_parameters.out_linear_hidden_out_features + [hyper_parameters.out_features],
            activation_provider=hyper_parameters.linear_activation_provider,
            dropout=hyper_parameters.linear_dropout
        ))

    # x: (N, B, D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.in_fnn.forward(x)

        embedded_with_pos = self.positional_encoder.forward(embedded)

        positional_query = self.positional_encoder.forward(torch.zeros_like(embedded_with_pos))

        attention_out = embedded_with_pos
        for i in range(self.stack_size):
            if self.self_attention:
                attention_out = self.attention_stack[i].forward(
                    query=attention_out,
                    key=attention_out,
                    value=attention_out
                )[0]
            else:
                attention_out = self.attention_stack[i].forward(
                    query=positional_query,
                    key=attention_out,
                    value=attention_out
                )[0]

        # attention_out = self.norm(embedded_with_pos + attention_out)

        out = self.out_fnn.forward(attention_out)

        return out
