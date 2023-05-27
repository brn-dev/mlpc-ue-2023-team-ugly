import copy
from datetime import datetime
from typing import Optional, Literal

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
from lib.training import get_lr, create_lr_scheduler, count_parameters
from lib.training_hyper_parameters import TrainingHyperParameters
from lib.cross_validation_training import train_with_cv


def train_attention_classifier_with_cv(
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        dataset: NumpyDataset,
        n_folds: int,
        device: torch.device,
        save_models: Literal[True, False, None, 'both', 'latest', 'best'] = None,
) -> tuple[
    list[tuple[AttentionClassifier, AttentionClassifier, StandardScaler]],
    CVFoldsMetrics
]:
    timestamp = get_current_timestamp()
    models_with_scalers: list[tuple[AttentionClassifier, AttentionClassifier, StandardScaler]] = []
    cv_folds_metrics: CVFoldsMetrics = []

    def train_func(fold_nr: int, train_ds: NumpyDataset, eval_ds: NumpyDataset, normalization_scaler: StandardScaler):
        print(f'Training fold {fold_nr}')
        latest_model, training_run_metrics, best_model, best_metrics = train_attention_classifier(
            hyper_parameters,
            training_hyper_parameters,
            train_ds,
            eval_ds,
            device
        )

        print(f'Evaluating fold {fold_nr}')
        eval_data_loader = create_data_loader(
            eval_ds.data,
            eval_ds.labels,
            training_hyper_parameters.batch_size,
            shuffle=False
        )
        evaluate_attention_classifier(
            latest_model,
            eval_data_loader,
            device,
            show_confmat=True,
            confmat_title=f'Validation Performance Fold {fold_nr} - Latest Model'
        )
        evaluate_attention_classifier(
            best_model,
            eval_data_loader,
            device,
            show_confmat=True,
            confmat_title=f'Validation Performance Fold {fold_nr} - Best Model'
        )

        if save_models is True or save_models == 'both' or save_models == 'latest':
            save_model_with_scaler(
                latest_model,
                normalization_scaler,
                f'attention_classifier '
                f'cv{timestamp} '
                f'fold-{fold_nr}-latest '
                f'train-bacc={training_run_metrics[-1][0].bacc:.4f} '
                f'eval-bacc={training_run_metrics[-1][1].bacc:.4f}'
            )
        if save_models is True or save_models == 'both' or save_models == 'best':
            save_model_with_scaler(
                best_model,
                normalization_scaler,
                f'attention_classifier '
                f'cv{timestamp} '
                f'fold-{fold_nr}-best '
                f'train-bacc={best_metrics[0].bacc:.4f} '
                f'eval-bacc={best_metrics[1].bacc:.4f}'
            )

        models_with_scalers.append((latest_model, best_model, normalization_scaler))
        cv_folds_metrics.append(training_run_metrics)

    train_with_cv(dataset, train_func, n_folds)
    return models_with_scalers, cv_folds_metrics


def train_attention_classifier(
        hyper_parameters: AttentionClassifierHyperParameters,
        training_hyper_parameters: TrainingHyperParameters,
        train_ds: NumpyDataset,
        eval_ds: Optional[NumpyDataset],
        device: torch.device
) -> tuple[AttentionClassifier, TrainingRunMetrics, AttentionClassifier, TrainAndEvaluationMetrics]:
    attention_classifier = AttentionClassifier(hyper_parameters)
    attention_classifier.to(device)
    print(f'Training AttentionClassifier with {count_parameters(attention_classifier)} parameters')

    optimizer = training_hyper_parameters.optimizer_provider(attention_classifier, training_hyper_parameters.lr)
    lr_scheduler = create_lr_scheduler(optimizer, training_hyper_parameters)

    train_data_loader = create_data_loader(
        train_ds.data,
        train_ds.labels,
        training_hyper_parameters.batch_size,
        shuffle=True
    )

    if eval_ds is not None:
        eval_data_loader = create_data_loader(
            eval_ds.data,
            eval_ds.labels,
            training_hyper_parameters.batch_size,
            shuffle=False
        )
    else:
        eval_data_loader = None

    train_label_counts = calculate_label_counts(train_ds.labels)
    print(f'train label counts = {train_label_counts.tolist()}')
    loss_weight = calculate_loss_weight(train_ds.labels, training_hyper_parameters.loss_weight_factors)
    print(f'loss weights = {[round(weight, 2) for weight in loss_weight.tolist()]}')

    if eval_ds is not None:
        eval_label_counts = calculate_label_counts(eval_ds.labels)
        print(f'eval label counts = {eval_label_counts.tolist()}')
        eval_loss_weight = calculate_loss_weight(eval_ds.labels, training_hyper_parameters.loss_weight_factors)
        print(f'eval loss weights = {[round(weight, 2) for weight in eval_loss_weight.tolist()]}')

    training_run_metrics, best_model, best_metrics = _train_attention_classifier(
        attention_classifier,
        train_data_loader,
        eval_data_loader,
        optimizer,
        loss_weight,
        training_hyper_parameters.num_epochs,
        lr_scheduler,
        device
    )

    return attention_classifier, training_run_metrics, best_model, best_metrics


def _train_attention_classifier(
        model: AttentionClassifier,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: Optional[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        loss_weight: torch.Tensor,
        num_epochs: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR],
        device: torch.device
) -> tuple[TrainingRunMetrics, AttentionClassifier, TrainAndEvaluationMetrics]:
    if num_epochs < 1:
        raise ValueError(f'{num_epochs = } has to be bigger than 0!')

    model.train()
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    best_score: float = -np.inf
    best_model: AttentionClassifier = copy.deepcopy(model)
    best_metrics: TrainAndEvaluationMetrics

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

        score = train_metrics.bacc
        print(f'Training Epoch {epoch:>3}/{num_epochs:<3}: lr = {get_lr(optimizer):.2E}, {train_metrics}')
        if validation_metrics is not None:
            score = validation_metrics.bacc
            print(f'Evaluation Epoch {epoch:>3}/{num_epochs:<3}: {validation_metrics}')

        if score >= best_score:
            best_score = score
            best_model = copy.deepcopy(model)
            best_metrics = (train_metrics, validation_metrics)

        training_run_metrics.append((train_metrics, validation_metrics))


    print('Training finished')
    return training_run_metrics, best_model, best_metrics


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

    metrics = metrics_collector.generate_metrics()

    if show_confmat:
        display_confmat(
            metrics_collector.pred_labels,
            metrics_collector.target_labels,
            confmat_title + f' - bacc={metrics.bacc:.4f}'
        )

    return metrics


def transpose_all(*tensors: torch.Tensor, dim0: int, dim1: int):
    return tuple((torch.transpose(x, dim0, dim1) for x in tensors))


def reshape_for_loss(
        pred: torch.Tensor,
        target_expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.flatten(pred, end_dim=1)
    target_expected = torch.flatten(target_expected)
    return pred, target_expected


def calculate_label_counts(labels: np.ndarray) -> torch.Tensor:
    return torch.bincount(torch.flatten(torch.Tensor(labels).long()))


def calculate_loss_weight(labels: np.ndarray, loss_weight_factors: Optional[torch.Tensor]) -> torch.Tensor:
    counts = calculate_label_counts(labels)
    counts_max = counts.max()
    weights = (counts_max / counts).to(get_torch_device())
    if loss_weight_factors is not None:
        weights *= loss_weight_factors
    return weights


def get_current_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H.%M')


def get_score(metrics: Metrics) -> float:
    return metrics.bacc


