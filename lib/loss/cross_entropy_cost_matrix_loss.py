import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from lib.loss.cost_matrix_loss import CostMatrixLoss


class CrossEntropyCostMatrixLoss(nn.Module):

    def __init__(
            self,
            cross_entropy_loss: CrossEntropyLoss,
            cross_entropy_weight: float,
            cost_matrix_loss: CostMatrixLoss,
            cost_matrix_weight: float,
    ):
        super().__init__()

        self.cross_entropy_loss = cross_entropy_loss
        self.cross_entropy_weight = cross_entropy_weight

        self.cost_matrix_loss = cost_matrix_loss
        self.cost_matrix_weight = cost_matrix_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        return self.cross_entropy_weight * self.cross_entropy_loss.forward(predictions, targets) \
               + self.cost_matrix_weight * self.cost_matrix_loss.forward(predictions, targets)

