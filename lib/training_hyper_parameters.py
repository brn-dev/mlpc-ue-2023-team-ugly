from dataclasses import dataclass
from typing import Optional, Callable

import torch

from lib.model.hyper_parameters import HyperParameters

LrSchedulerProvider = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.MultiStepLR]

@dataclass
class TrainingHyperParameters(HyperParameters):
    batch_size: int
    num_epochs: int
    lr: float
    lr_scheduler_provider: Optional[LrSchedulerProvider]
