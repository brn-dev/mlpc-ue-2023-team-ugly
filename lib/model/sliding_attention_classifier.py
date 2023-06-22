from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from lib.model.attention_classifier import AttentionClassifierHyperParameters, AttentionClassifier
from lib.model.multihead_self_attention import MultiheadSelfAttention
from lib.model.positional_encoding import PositionalEncoding

ActivationProvider = Callable[[], nn.Module]


@dataclass
class SlidingAttentionClassifierHyperParameters(AttentionClassifierHyperParameters):
    step: int


class SlidingAttentionClassifier(AttentionClassifier):
    def __init__(self, hyper_parameters: SlidingAttentionClassifierHyperParameters, batch_first: bool):
        super().__init__(hyper_parameters, batch_first=batch_first)

        self.d_model = hyper_parameters.d_model
        self.attention_window_size = hyper_parameters.attention_window_size
        self.step = hyper_parameters.step

        assert ((self.attention_window_size - self.step) / 2) % 1 == 0.0, '(window_size - step) must be even'

        self.one_sided_pad = (self.attention_window_size - self.step) // 2

        self.positional_encoder = PositionalEncoding(
            d_model=hyper_parameters.d_model,
            max_len=1024,
            batch_first=True,
        )

        attention_stack_modules: list[nn.Module] = []

        for _ in range(hyper_parameters.attention_stack_size - 1):
            attention_stack_modules.append(MultiheadSelfAttention(hyper_parameters, batch_first=True))
            attention_stack_modules.append(hyper_parameters.attention_stack_activation_provider())

        if hyper_parameters.attention_stack_size > 0:
            attention_stack_modules.append(MultiheadSelfAttention(hyper_parameters, batch_first=True))

        self.attention_stack = nn.Sequential(*attention_stack_modules)

    def __str__(self):
        return (f'SlidingAttentionClassifier with {_count_parameters(self)} parameters, '
                f'in_fnn: {_count_parameters(self.in_fnn)}, '
                f'attention_stack: {_count_parameters(self.attention_stack)}, '
                f'out_fnn: {_count_parameters(self.out_fnn)}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using batch first internally
        if not self.batch_first:
            x = torch.swapaxes(x, 0, 1)

        n_sequences, sequence_length, in_features = x.shape

        embedded = self.in_fnn.forward(x)

        if self.one_sided_pad > 0:
            assert sequence_length % self.step == 0, 'sequence_length must be divisible by step'

            windows = torch.nn.functional.pad(embedded, (0, 0, self.one_sided_pad, self.one_sided_pad))
            windows = windows.unfold(1, self.attention_window_size, self.step)
        else:
            windows = embedded

        windows = windows.reshape(-1, self.attention_window_size, self.d_model)

        embedded_with_pos = self.positional_encoder.forward(windows)
        attention_out = self.attention_stack.forward(embedded_with_pos)

        if self.one_sided_pad > 0:
            attention_out = attention_out[:, self.one_sided_pad:-self.one_sided_pad, :]

        # TODO: Fold
        attention_out = attention_out.reshape(n_sequences, sequence_length, self.d_model)

        # residual
        # attention_out += embedded

        out = self.out_fnn.forward(attention_out)

        if not self.batch_first:
            out = torch.swapaxes(out, 0, 1)

        return out

def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
