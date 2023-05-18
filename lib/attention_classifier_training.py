from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import StandardScaler

from lib.confusion_matrix import display_confmat
from lib.ds.numpy_dataset import NumpyDataset
from lib.ds.torch_dataset import create_data_loader
from lib.metrics import CVFoldsMetrics, TrainingRunMetrics, MetricsCollector, TrainAndEvaluationMetrics, Metrics
from lib.model.attention_classifier import AttentionClassifier, AttentionClassifierHyperParameters
from lib.model.model_persistence import save_model_with_scaler
from lib.torch_device import get_torch_device
from lib.training import train_with_cv, get_lr, create_lr_scheduler, count_parameters
from lib.training_hyper_parameters import TrainingHyperParameters


def train_attention_classifier_with_cv(
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        dataset: NumpyDataset,
        device: torch.device,
        cv_folds_permute_seed: Optional[int]
) -> tuple[
    list[tuple[AttentionClassifier, StandardScaler]],
    CVFoldsMetrics
]:
    timestamp = get_current_timestamp()
    models_with_scalers: list[tuple[AttentionClassifier, StandardScaler]] = []
    cv_folds_metrics: CVFoldsMetrics = []

    def train_func(fold_nr: int, train_ds: NumpyDataset, eval_ds: NumpyDataset, normalization_scaler: StandardScaler):
        print(f'Training fold {fold_nr}')
        model, training_run_metrics = train_attention_classifier(
            hyper_parameters,
            training_hyper_parameters,
            train_ds,
            eval_ds,
            device
        )

        print(f'Evaluating fold {fold_nr}')
        eval_data_loader = create_data_loader(eval_ds.data, eval_ds.labels, training_hyper_parameters.batch_size)
        evaluate_attention_classifier(
            model,
            eval_data_loader,
            device,
            show_confmat=True,
            confmat_title=f'Validation Fold {fold_nr} Performance'
        )

        save_model_with_scaler(
            model,
            normalization_scaler,
            f'attention_classifier cv{timestamp} fold-{fold_nr}'
        )
        models_with_scalers.append((model, normalization_scaler))
        cv_folds_metrics.append(training_run_metrics)

    train_with_cv(dataset, train_func, cv_folds_permute_seed=cv_folds_permute_seed)
    return models_with_scalers, cv_folds_metrics


def train_attention_classifier(
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        train_ds: NumpyDataset,
        eval_ds: Optional[NumpyDataset],
        device: torch.device
) -> tuple[AttentionClassifier, TrainingRunMetrics]:
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

    training_run_metrics = _train_attention_classifier(
        attention_classifier,
        train_data_loader,
        eval_data_loader,
        optimizer,
        loss_weight,
        training_hyper_parameters.num_epochs,
        lr_scheduler,
        device
    )

    return attention_classifier, training_run_metrics


def _train_attention_classifier(
        model: AttentionClassifier,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_weight: torch.Tensor,
        num_epochs: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR],
        device: torch.device
) -> TrainingRunMetrics:
    model.train()
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    training_run_metrics: TrainingRunMetrics = []

    for epoch in range(1, num_epochs + 1):
        train_metrics, validation_metrics = train_epoch(
            model,
            criterion,
            train_data_loader,
            eval_data_loader,
            optimizer,
            device
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f'Training Epoch {epoch:>3}/{num_epochs:<3}: lr = {get_lr(optimizer)}, {train_metrics}')
        if validation_metrics is not None:
            print(f'Evaluation Epoch {epoch:>3}/{num_epochs:<3}: {validation_metrics}')

        training_run_metrics.append((train_metrics, validation_metrics))


    print('Training finished')
    return training_run_metrics


def train_epoch(
        model: AttentionClassifier,
        criterion: nn.CrossEntropyLoss,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        device: torch.device

) -> TrainAndEvaluationMetrics:
    metrics_collector = MetricsCollector()

    for data, labels in train_data_loader:
        data, labels = transpose_all(data, labels, dim0=0, dim1=1)
        data, labels = data.float().to(device), labels.long().to(device)

        optimizer.zero_grad()

        pred = model.forward(data)

        pred, labels = reshape_for_loss(pred, labels)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        pred_labels: torch.Tensor = pred.argmax(dim=1).view(-1).long()
        metrics_collector.update(
            loss.detach().item(),
            pred_labels.cpu().detach().numpy(),
            labels.cpu().detach().numpy()
        )

    train_metrics = metrics_collector.generate_metrics()

    eval_metrics: Optional[Metrics] = None
    if eval_data_loader is not None:
        eval_metrics = evaluate_attention_classifier(model, eval_data_loader, device, show_confmat=False)

    return train_metrics, eval_metrics


def evaluate_attention_classifier(
        model: AttentionClassifier,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        show_confmat: bool = True,
        confmat_title: str = None
) -> Metrics:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    metrics_collector = MetricsCollector()

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = transpose_all(data, labels, dim0=0, dim1=1)
            data, labels = data.float().to(device), labels.long().to(device)

            pred = model.forward(data)

            pred, labels = reshape_for_loss(pred, labels)

            pred_labels: torch.Tensor = pred.argmax(dim=1).view(-1).long()
            metrics_collector.update(
                criterion(pred, labels).detach().item(),
                pred_labels.cpu().detach().numpy(),
                labels.cpu().detach().numpy()
            )

    if show_confmat:
        display_confmat(
            metrics_collector.pred_labels,
            metrics_collector.target_labels,
            confmat_title
        )

    return metrics_collector.generate_metrics()


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
    counts = torch.bincount(torch.flatten(torch.Tensor(labels).long()))
    counts_max = counts.max()
    return (counts_max / counts).to(get_torch_device())


def get_current_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H.%M')
