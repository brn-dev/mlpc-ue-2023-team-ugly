from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyDataset:
    data: np.ndarray
    labels: np.ndarray

    def __iter__(self):
        for value in [self.data, self.labels]:
            yield value

    def copy(self) -> 'NumpyDataset':
        return NumpyDataset(
            self.data.copy(),
            self.labels.copy()
        )
