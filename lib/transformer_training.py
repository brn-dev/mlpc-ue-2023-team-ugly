from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from lib.hyper_parameters import HyperParameters
from lib.training import train_with_cv
from lib.ds.torch_dataset import create_data_loader


@dataclass(frozen=True, init=True)
class TransformerHyperParameters(HyperParameters):
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1


def train_transformer_with_cv(
        data: np.ndarray,
        labels: np.ndarray,
        hyper_parameters: HyperParameters,
        device: torch.device = 'cpu'
):
    def create_and_train_func(d: np.ndarray, l: np.ndarray):
        transformer = nn.Transformer(**hyper_parameters)
        optimizer = optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
        transformer.to(device)

        data_loader = create_data_loader(d, l)

        train_network(transformer, data_loader, optimizer, device)

    def eval_func(network: nn.Module, d: np.ndarray, l: np.ndarray):
        data_loader = create_data_loader(d, l)
        loss, acc = test_network(network, data_loader, device)
        print(f'{loss = }, {acc = }')
        # TODO: collect acc, conf matrix


    train_with_cv(data, labels, create_and_train_func, eval_func)


def train_network(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device = 'cpu'
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    for data, target in data_loader:
        data, target = data.float().to(device), target.long().to(device)
        optimizer.zero_grad()
        output = model(data, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_network(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = 'cpu'
) -> Tuple[float, float]:
    model.eval()
    loss, num_correct, num_samples = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            loss += float(criterion(output, target).item())
            pred = output.argmax(dim=1).view(-1).long()
            num_correct += int((pred == target.view(-1)).sum().item())
            num_samples += pred.shape[0]
    return loss / num_samples, num_correct / num_samples