from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyDataset:
    data: np.ndarray
    labels: np.ndarray
