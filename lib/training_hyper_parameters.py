from dataclasses import dataclass
from typing import Optional, Callable

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters

LrSchedulerProvider = Callable[[torch.optim.Optimizer, list[int], float], torch.optim.lr_scheduler.MultiStepLR]
OptimizerProvider = Callable[[nn.Module, float], torch.optim.Optimizer]

@dataclass
class TrainingHyperParameters(HyperParameters):
    batch_size: int

    optimizer_provider: OptimizerProvider

    num_epochs: int

    lr: float
    lr_scheduler_milestones: list[int]
    lr_scheduler_gamma: float
    lr_scheduler_provider: Optional[LrSchedulerProvider]
