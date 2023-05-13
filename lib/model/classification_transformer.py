import math
from dataclasses import dataclass

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters
from lib.model.positional_encoding import PositionalEncoding
from lib.torch_device import get_torch_device


@dataclass(init=True)
class TransformerHyperParameters(HyperParameters):
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float

    out_size: int

    in_features: int

    lr: float


class ClassificationTransformer(nn.Module):

    def __init__(self, hyper_parameters: TransformerHyperParameters):
        super().__init__()
        # TODO: dropout

        self.hyper_parameters = hyper_parameters

        self.src_embedding = nn.Linear(
            in_features=hyper_parameters.in_features,
            out_features=hyper_parameters.d_model
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=hyper_parameters.out_size + 1,
            embedding_dim=hyper_parameters.d_model
        )
        self.positional_encoder = PositionalEncoding(
            d_model=hyper_parameters.d_model,
            max_len=1024
        )
        self.transformer = nn.Transformer(
            d_model=hyper_parameters.d_model,
            nhead=hyper_parameters.nhead,
            num_encoder_layers=hyper_parameters.num_encoder_layers,
            num_decoder_layers=hyper_parameters.num_decoder_layers,
            dim_feedforward=hyper_parameters.dim_feedforward,
            dropout=hyper_parameters.dropout,
            # batch_first=True,
        )
        self.out_fnn = nn.Linear(
            in_features=hyper_parameters.d_model,
            out_features=hyper_parameters.out_size
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        # TODO: do we need this?
        src *= math.sqrt(512)
        tgt *= math.sqrt(512)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(get_torch_device())

        transformer_out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask
        )
        out = self.out_fnn(transformer_out)

        return out

    # TODO: allow for variable length sequences with a pad mask and src/tgt_key_padding_mask
