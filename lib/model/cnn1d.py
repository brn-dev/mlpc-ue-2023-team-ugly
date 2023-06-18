from torch import nn

from lib.model.cnn import CNN, ConvHyperParameters


class CNN1d(CNN):

    def _create_conv(self, in_channels: int, hyper_parameters: ConvHyperParameters) -> nn.Conv1d:
        return nn.Conv1d(
            in_channels=in_channels,
            out_channels=hyper_parameters.out_channels,
            kernel_size=hyper_parameters.kernel_size,
            stride=hyper_parameters.stride,
            padding=hyper_parameters.padding,
            groups=hyper_parameters.groups,
            bias=hyper_parameters.bias,
            padding_mode=hyper_parameters.padding_mode
        )

    def _create_batch_norm(self, num_features: int) -> nn.BatchNorm1d:
        return nn.BatchNorm1d(num_features)
