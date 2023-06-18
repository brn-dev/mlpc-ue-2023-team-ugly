from dataclasses import dataclass
from typing import Optional, Callable

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters

# (balancing_weights) -> loss_function
LossModuleProvider = Callable[[torch.Tensor], nn.Module]

# (model, lr) -> optimizer
OptimizerProvider = Callable[[nn.Module, float], torch.optim.Optimizer]

# (optimizer, epoch_milestones, lr_gamma) -> lr_scheduler
LrSchedulerProvider = Callable[[torch.optim.Optimizer, list[int], float], torch.optim.lr_scheduler.MultiStepLR]

@dataclass
class TrainingHyperParameters(HyperParameters):
    batch_size: int

    loss_module_provider: LossModuleProvider
    optimizer_provider: OptimizerProvider

    num_epochs: int

    lr: float
    lr_scheduler_milestones: list[int]
    lr_scheduler_gamma: float
    lr_scheduler_provider: Optional[LrSchedulerProvider]
