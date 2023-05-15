from dataclasses import dataclass

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters
from lib.model.positional_encoding import PositionalEncoding


@dataclass
class AttentionClassifierHyperParameters(HyperParameters):
    d_model: int
    num_heads: int
    stack_size: int
    dropout: float


    out_size: int

    in_features: int


class AttentionClassifier(nn.Module):
    def __init__(self, hyper_parameters: AttentionClassifierHyperParameters):
        super().__init__()
        # TODO: dropout

        self.hyper_parameters = hyper_parameters
        self.stack_size = hyper_parameters.stack_size

        self.src_embedding = nn.Linear(
            in_features=hyper_parameters.in_features,
            out_features=hyper_parameters.d_model
        )
        self.positional_encoder = PositionalEncoding(
            d_model=hyper_parameters.d_model,
            max_len=1024
        )
        self.attention_stack = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hyper_parameters.d_model,
                num_heads=hyper_parameters.num_heads,
                dropout=hyper_parameters.dropout
            )
            for _
            in range(hyper_parameters.stack_size)
        ])
        self.out_fnn = nn.Linear(
            in_features=hyper_parameters.d_model,
            out_features=hyper_parameters.out_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.src_embedding.forward(x)

        embedded_with_pos = self.positional_encoder.forward(embedded)

        query = self.positional_encoder.forward(torch.zeros_like(embedded_with_pos))

        attention_out = embedded_with_pos

        for i in range(self.stack_size):
            attention_out = self.attention_stack[i].forward(
                query=query,
                key=attention_out,
                value=attention_out
            )[0]

        out = self.out_fnn.forward(attention_out)

        return out
