import numpy as np
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader

SOS_token_value = 7

def create_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    data_tensor = torch.Tensor(data)
    labels_tensor = torch.Tensor(labels)

    return TensorDataset(data_tensor, labels_tensor)


def create_data_loader(data: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    return DataLoader(create_tensor_dataset(data, labels), batch_size)


def create_offset_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    data_tensor = torch.Tensor(data)
    labels_tensor = torch.Tensor(labels)

    labels_input_tensor = torch.cat(
        (
            torch.full(
                (labels_tensor.size(0), 1),
                SOS_token_value
            ),
            labels_tensor
        ),
        dim=-1
    )

    return TensorDataset(data_tensor, labels_input_tensor, labels_tensor)


def create_offset_data_loader(data: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    return DataLoader(create_offset_tensor_dataset(data, labels), batch_size)

