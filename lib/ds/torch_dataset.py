import numpy as np
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader


def create_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    data_tensor = torch.Tensor(data)
    labels_tensor = torch.Tensor(labels)

    return TensorDataset(data_tensor, labels_tensor)


def create_data_loader(data: np.ndarray, labels: np.ndarray) -> DataLoader:
    return DataLoader(create_tensor_dataset(data, labels))
