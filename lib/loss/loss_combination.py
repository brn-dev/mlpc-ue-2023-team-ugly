# import torch
# from torch import nn
#
#
# ModuleWithWeight = tuple[nn.Module, float]
#
# class LossCombination(nn.Module):
#
#     def __init__(self, loss_modules_with_weight: list[ModuleWithWeight]):
#         super().__init__()
#
#         self.loss_modules = nn.ModuleList([
#             loss_module_with_weight[0]
#             for loss_module_with_weight
#             in loss_modules_with_weight
#         ])
#
#         self.weights = torch.Tensor([
#             loss_module_with_weight[1]
#             for loss_module_with_weight
#             in loss_modules_with_weight
#         ])
#
#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#
#
# TODO