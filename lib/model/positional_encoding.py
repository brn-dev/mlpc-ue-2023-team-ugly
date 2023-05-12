import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    pos_encoding: torch.Tensor


    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        # TODO: dropout

        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return token_embedding + self.pos_encoding[:token_embedding.shape[1]]
