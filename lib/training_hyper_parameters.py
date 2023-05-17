from dataclasses import dataclass
from typing import Optional, Callable

import torch
from torch import nn

from lib.model.hyper_parameters import HyperParameters

LrSchedulerProvider = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.MultiStepLR]
OptimizerProvider = Callable[[nn.Module, float], torch.optim.Optimizer]

@dataclass
class TrainingHyperParameters(HyperParameters):
    batch_size: int
    num_epochs: int
    optimizer_provider: OptimizerProvider
    lr: float
    lr_scheduler_provider: Optional[LrSchedulerProvider]
