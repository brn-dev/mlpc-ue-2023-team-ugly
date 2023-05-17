from datetime import datetime
from typing import Optional

import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import StandardScaler

from lib.confusion_matrix import display_confmat
from lib.ds.numpy_dataset import NumpyDataset
from lib.ds.torch_dataset import create_data_loader
from lib.model.attention_classifier import AttentionClassifier, AttentionClassifierHyperParameters
from lib.model.model_persistence import save_model_with_scaler
from lib.torch_device import get_torch_device
from lib.training import train_with_cv, get_lr, create_lr_scheduler, count_parameters
from lib.training_hyper_parameters import TrainingHyperParameters


def train_attention_classifier_with_cv(
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        dataset: NumpyDataset,
        device: torch.device
):
    models: list[AttentionClassifier] = []

    def train_func(fold_nr: int, train_ds: NumpyDataset, eval_ds: NumpyDataset, normalization_scaler: StandardScaler):
        print(f'Training fold {fold_nr}')
        model = train_attention_classifier(
            hyper_parameters,
            training_hyper_parameters,
            train_ds,
            eval_ds,
            device
        )

        print(f'Evaluating fold {fold_nr}')
        eval_data_loader = create_data_loader(eval_ds.data, eval_ds.labels, training_hyper_parameters.batch_size)
        test_attention_classifier(model, eval_data_loader, device, show_confmat=True)

        save_model_with_scaler(
            model,
            normalization_scaler,
            f'attention_classifier cv{get_current_timestamp()} fold-{fold_nr}'
        )
        models.append(model)

    train_with_cv(dataset, train_func)
    return models


def train_attention_classifier(
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        train_ds: NumpyDataset,
        eval_ds: Optional[NumpyDataset],
        device: torch.device
) -> AttentionClassifier:
    attention_classifier = AttentionClassifier(hyper_parameters)
    attention_classifier.to(device)
    print(f'Training AttentionClassifier with {count_parameters(attention_classifier)} parameters')

    optimizer = training_hyper_parameters.optimizer_provider(attention_classifier, training_hyper_parameters.lr)
    lr_scheduler = create_lr_scheduler(optimizer, training_hyper_parameters)

    train_data_loader = create_data_loader(train_ds.data, train_ds.labels, training_hyper_parameters.batch_size)

    if eval_ds is not None:
        eval_data_loader = create_data_loader(eval_ds.data, eval_ds.labels, training_hyper_parameters.batch_size)
    else:
        eval_data_loader = None

    loss_weight = calculate_loss_weight(train_ds.labels)
    print(f'{loss_weight = }')

    _train_attention_classifier(
        attention_classifier,
        train_data_loader,
        eval_data_loader,
        optimizer,
        loss_weight,
        training_hyper_parameters.num_epochs,
        lr_scheduler,
        device
    )

    return attention_classifier


def _train_attention_classifier(
        model: AttentionClassifier,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: Optional[torch.utils.data.DataLoader],
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
        for data, labels in train_data_loader:
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

        acc = num_correct / num_samples
        bacc = sklearn.metrics.balanced_accuracy_score(all_target_labels, all_pred_labels)
        print(
            f'Training Epoch {epoch:<3}: '
            f'lr = {get_lr(optimizer):.6f}, '
            f'{epoch_loss = :.6f}, '
            f'{num_correct = :>5}, '
            f'{num_samples = :>5}, '
            f'{acc = :.6f}, '
            f'{bacc = :.6f}'
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        if eval_data_loader is not None:
            test_attention_classifier(model, eval_data_loader, device, show_confmat=False)

    print('Training finished')


def test_attention_classifier(
        model: AttentionClassifier,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        show_confmat: bool = True,
        confmat_title: str = None
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
    print(
        f'Evaluated with '
        f'{loss = :.6f}, '
        f'{acc = :.6f}, '
        f'{bacc = :.6f}'
    )

    if show_confmat:
        display_confmat(all_pred_labels, all_target_labels, confmat_title)


def transpose_all(*tensors: torch.Tensor, dim0: int, dim1: int):
    return tuple((torch.transpose(x, dim0, dim1) for x in tensors))


def reshape_for_loss(
        pred: torch.Tensor,
        target_expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.flatten(pred, end_dim=1)
    target_expected = torch.flatten(target_expected)
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

def get_current_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H.%M')

