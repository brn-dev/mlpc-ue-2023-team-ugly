from typing import Optional, Any, Iterable

import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from lib.confusion_matrix import display_confmat
from lib.ds.torch_dataset import create_data_loader
from lib.model.attention_classifier import AttentionClassifier, AttentionClassifierHyperParameters
from lib.torch_device import get_torch_device
from lib.torch_utils import count_parameters
from lib.training import train_with_cv
from lib.training_hyper_parameters import TrainingHyperParameters


def train_attention_classifier_with_cv(
        data: np.ndarray,
        labels: np.ndarray,
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        device: torch.device
):
    def create_and_train_func(d: np.ndarray, l: np.ndarray):
        return train_attention_classifier(d, l, hyper_parameters, training_hyper_parameters, device)

    def eval_func(network: AttentionClassifier, d: np.ndarray, l: np.ndarray):
        data_loader = create_data_loader(d, l, batch_size=training_hyper_parameters.batch_size)
        test_attention_classifier(network, data_loader, device)


    train_with_cv(data, labels, create_and_train_func, eval_func)


def train_attention_classifier(
        data: np.ndarray,
        labels: np.ndarray,
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        device: torch.device
) -> AttentionClassifier:
    attention_classifier = AttentionClassifier(hyper_parameters)
    print(f'Training AttentionClassifier with {count_parameters(attention_classifier)} parameters')
    attention_classifier.to(device)

    optimizer = optim.Adam(attention_classifier.parameters(), lr=training_hyper_parameters.lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = create_lr_scheduler(optimizer, training_hyper_parameters)

    data_loader = create_data_loader(data, labels, training_hyper_parameters.batch_size)

    loss_weight = calculate_loss_weight(labels)
    print(f'{loss_weight = }')

    _train_attention_classifier(
        attention_classifier,
        data_loader,
        optimizer,
        loss_weight,
        training_hyper_parameters.num_epochs,
        lr_scheduler,
        device
    )

    return attention_classifier



def _train_attention_classifier(
        model: AttentionClassifier,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_weight: torch.Tensor,
        num_epochs: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR],
        device: torch.device
):
    model.train()

    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    all_pred_labels = torch.Tensor().long()
    all_target_labels = torch.Tensor().long()

    for epoch in range(num_epochs):
        epoch_loss, num_correct, num_samples = 0.0, 0, 0
        for data, labels in data_loader:
            data, labels = transpose_all(data, labels, dim0=0, dim1=1)
            data, labels = data.float().to(device), labels.long().to(device)

            optimizer.zero_grad()

            pred = model.forward(data)

            pred, labels = reshape_for_loss(pred, labels)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

            pred_labels = pred.argmax(dim=1).view(-1).long()
            num_correct += int((pred_labels == labels.view(-1)).sum().item())
            num_samples += pred_labels.shape[0]

            all_pred_labels = torch.cat((all_pred_labels, pred_labels.cpu()))
            all_target_labels = torch.cat((all_target_labels, labels.cpu()))

        if lr_scheduler is not None:
            lr_scheduler.step()

        acc = num_correct / num_samples
        bacc = sklearn.metrics.balanced_accuracy_score(all_target_labels, all_pred_labels)
        print(f'Training Epoch {epoch:<3}: '
              f'{epoch_loss = :.6f}, '
              f'{num_correct = :>5}, '
              f'{num_samples = :>5}, '
              f'{acc = :.6f}, '
              f'{bacc = :.6f}'
        )

    print('Training finished')


def test_attention_classifier(
        model: AttentionClassifier,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device
):
    loss, num_correct, num_samples = 0.0, 0, 0

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_pred_labels = torch.Tensor().long()
    all_target_labels = torch.Tensor().long()

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = transpose_all(data, labels, dim0=0, dim1=1)
            data, labels = data.float().to(device), labels.long().to(device)

            pred = model.forward(data)

            pred, labels = reshape_for_loss(pred, labels)
            loss += criterion(pred, labels).detach().item()

            pred_labels = pred.argmax(dim=1).view(-1).long()
            num_correct += int((pred_labels == labels.view(-1)).sum().item())
            num_samples += pred_labels.shape[0]

            all_pred_labels = torch.cat((all_pred_labels, pred_labels.cpu()))
            all_target_labels = torch.cat((all_target_labels, labels.cpu()))

    loss, acc = loss / num_samples, num_correct / num_samples
    bacc = sklearn.metrics.balanced_accuracy_score(all_target_labels, all_pred_labels)
    print(f'Evaluated with {loss = }, {acc = }, {bacc = }')

    display_confmat(all_pred_labels, all_target_labels)


def transpose_all(*tensors: torch.Tensor, dim0: int, dim1: int):
    return tuple((torch.transpose(x, dim0, dim1) for x in tensors))


def reshape_for_loss(
        pred: torch.Tensor,
        target_expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.flatten(pred[:-1, :, :], end_dim=1)
    target_expected = torch.flatten(target_expected[:-1, :])
    return pred, target_expected


def calculate_loss_weight(labels: np.ndarray) -> torch.Tensor:
    # print(f'{np.min(labels) = }')
    # print(f'{torch.flatten(torch.Tensor(labels).long()).min() = }')
    counts = torch.bincount(torch.flatten(torch.Tensor(labels).long()))
    counts_max = counts.max()
    return (counts_max / counts).to(get_torch_device())


# def calculate_loss_weight(labels: np.ndarray) -> torch.Tensor:
#     print(f'{np.min(labels) = }')
#     print(f'{torch.flatten(torch.Tensor(labels).long()).min() = }')
#     counts = torch.bincount(torch.flatten(torch.Tensor(labels).long()))
#     other_count = counts[0]
#     birds_count = counts[1:].sum()
#     return torch.Tensor([1] + [other_count / birds_count * 10] * (len(counts) - 1)).to(get_torch_device())

def create_lr_scheduler(optimizer: torch.optim.Optimizer, training_hyper_parameters: TrainingHyperParameters):
    if training_hyper_parameters.lr_scheduler_provider is None:
        return None
    return training_hyper_parameters.lr_scheduler_provider(optimizer)


