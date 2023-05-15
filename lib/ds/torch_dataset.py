import numpy as np
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader

SOS_TOKEN_VALUE = 7
EOS_TOKEN_VALUE = 8


def create_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    data_tensor = torch.Tensor(data)
    labels_tensor = torch.Tensor(labels)

    return TensorDataset(data_tensor, labels_tensor)


def create_data_loader(data: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    return DataLoader(create_tensor_dataset(data, labels), batch_size)


def create_offset_tensor_dataset(data: np.ndarray, labels: np.ndarray) -> TensorDataset:
    data_tensor = torch.Tensor(data)
    labels_tensor = torch.Tensor(labels)

    def create_label_tensor(value: int):
        return torch.full(
            (labels_tensor.size(0), 1),
            value
        )

    labels_input_tensor = torch.cat((create_label_tensor(SOS_TOKEN_VALUE), labels_tensor), dim=-1)
    labels_expected_tensor = torch.cat((labels_tensor, create_label_tensor(EOS_TOKEN_VALUE)), dim=-1)

    return TensorDataset(data_tensor, labels_input_tensor, labels_expected_tensor)


def create_offset_data_loader(data: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    return DataLoader(create_offset_tensor_dataset(data, labels), batch_size)



