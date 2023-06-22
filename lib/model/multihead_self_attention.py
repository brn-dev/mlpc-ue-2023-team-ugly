from dataclasses import dataclass

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters


@dataclass
class MultiheadSelfAttentionHyperParameters(HyperParameters):
    d_model: int
    num_heads: int
    attention_dropout: float


class MultiheadSelfAttention(nn.Module):

    def __init__(self, hyper_parameters: MultiheadSelfAttentionHyperParameters, batch_first: bool):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hyper_parameters.d_model,
            num_heads=hyper_parameters.num_heads,
            dropout=hyper_parameters.attention_dropout,
            batch_first=batch_first
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (*, Sequence-length, Dimensions)
        """
        x = self.multihead_attention.forward(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )[0]

        return x
