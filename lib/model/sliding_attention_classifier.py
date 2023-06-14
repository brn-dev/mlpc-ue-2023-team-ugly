from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from lib.model.attention_classifier import AttentionClassifierHyperParameters, AttentionClassifier

ActivationProvider = Callable[[], nn.Module]


@dataclass
class SlidingAttentionClassifierHyperParameters(AttentionClassifierHyperParameters):
    stride: int


class SlidingAttentionClassifier(AttentionClassifier):
    def __init__(self, hyper_parameters: SlidingAttentionClassifierHyperParameters, batch_first: bool):
        super().__init__(hyper_parameters, batch_first=batch_first)

        self.attention_window_size = hyper_parameters.attention_window_size
        self.stride = hyper_parameters.stride

    def __str__(self):
        return (f'SlidingAttentionClassifier with {_count_parameters(self)} parameters, '
                f'in_fnn: {_count_parameters(self.in_fnn)}, '
                f'attention_stack: {_count_parameters(self.attention_stack)}, '
                f'out_fnn: {_count_parameters(self.out_fnn)}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using batch second internally
        if self.batch_first:
            x = torch.swapaxes(x, 0, 1)

        sequence_length, n_sequences, dimensions = x.shape

        folded = nn.functional.pad(x.permute(1, 2, 0), (self.stride, self.stride)) \
            .unfold(2, self.attention_window_size, self.stride)
        folded = folded.reshape(-1, dimensions, self.attention_window_size).permute(2, 0, 1)

        embedded = self.in_fnn.forward(folded)

        embedded_with_pos = self.positional_encoder.forward(embedded)
        attention_out = self.attention_stack.forward(embedded_with_pos)

        out = self.out_fnn.forward(attention_out)

        out = (
            out[self.stride:-self.stride, :, :]
                .permute(1, 0, 2)
                .reshape(n_sequences, sequence_length, self.out_features)
                .permute(1, 0, 2)
        )

        if self.batch_first:
            out = torch.swapaxes(out, 0, 1)

        return out

def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
