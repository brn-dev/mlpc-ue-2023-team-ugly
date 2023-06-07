import copy
from datetime import datetime
from typing import Optional, Literal, TypeVar, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import StandardScaler

from lib.confusion_matrix import display_confmat
from lib.ds.numpy_dataset import NumpyDataset
from lib.ds.torch_dataset import create_data_loader
from lib.metrics import CVFoldsMetrics, TrainingRunMetrics, LabelCollector, TrainAndEvaluationMetrics, Metrics
from lib.model.model_persistence import save_model_with_scaler
from lib.torch_device import get_torch_device
from lib.training_hyper_parameters import TrainingHyperParameters
from lib.cross_validation_training import train_with_cv

M = TypeVar('M', bound=nn.Module)

ModelProvider = Callable[[], M]

def train_model_with_cv(
        model_provider: ModelProvider,
        training_hyper_parameters: TrainingHyperParameters,
        dataset: NumpyDataset,
        n_folds: int,
        device: torch.device,
        save_models: Literal[True, False, None, 'both', 'latest', 'best'],
        model_saving_name: str,
) -> tuple[
    list[tuple[M, M, StandardScaler]],
    CVFoldsMetrics,
    list[TrainAndEvaluationMetrics]
]:
    timestamp = _get_current_timestamp()
    models_with_scalers: list[tuple[M, M, StandardScaler]] = []
    cv_folds_metrics: CVFoldsMetrics = []
    best_models_metrics: list[TrainAndEvaluationMetrics] = []

    def train_func(fold_nr: int, train_ds: NumpyDataset, eval_ds: NumpyDataset, normalization_scaler: StandardScaler):
        print(f'Training fold {fold_nr}')
        model = model_provider()

        latest_model, training_run_metrics, best_model, best_metrics = train_model(
            model,
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
        evaluate_model(
            latest_model,
            eval_data_loader,
            device,
            show_confmat=True,
            confmat_title=f'Validation Performance Fold {fold_nr} - Latest Model',
        )
        evaluate_model(
            best_model,
            eval_data_loader,
            device,
            show_confmat=True,
            confmat_title=f'Validation Performance Fold {fold_nr} - Best Model ({best_metrics[0].epoch})',

        )

        if save_models is True or save_models == 'both' or save_models == 'latest':
            save_model_with_scaler(
                latest_model,
                normalization_scaler,
                f'{model_saving_name} '
                f'cv{timestamp} '
                f'fold-{fold_nr}-latest '
                f'score={training_run_metrics[-1][0].score:.4f} '
            )
        if save_models is True or save_models == 'both' or save_models == 'best':
            save_model_with_scaler(
                best_model,
                normalization_scaler,
                f'{model_saving_name} '
                f'cv{timestamp} '
                f'fold-{fold_nr}-best '
                f'score={best_metrics[0].score:.4f} '
            )

        models_with_scalers.append((latest_model, best_model, normalization_scaler))
        cv_folds_metrics.append(training_run_metrics)
        best_models_metrics.append(best_metrics)


    train_with_cv(dataset, train_func, n_folds)
    return models_with_scalers, cv_folds_metrics, best_models_metrics


def train_model(
        model: M,
        training_hyper_parameters: TrainingHyperParameters,
        train_ds: NumpyDataset,
        eval_ds: NumpyDataset,
        device: torch.device
) -> tuple[M, TrainingRunMetrics, M, TrainAndEvaluationMetrics]:
    model.to(device)
    print('\n\n#### Training ####')
    print('#' * len('#### Training ####'))
    print(model)
    print('#' * len('#### Training ####'))

    optimizer = training_hyper_parameters.optimizer_provider(model, training_hyper_parameters.lr)
    lr_scheduler = _create_lr_scheduler(optimizer, training_hyper_parameters)

    train_data_loader = create_data_loader(
        train_ds.data,
        train_ds.labels,
        training_hyper_parameters.batch_size,
        shuffle=True
    )

    eval_data_loader = create_data_loader(
        eval_ds.data,
        eval_ds.labels,
        training_hyper_parameters.batch_size,
        shuffle=False
    )

    train_label_counts = _calculate_label_counts(train_ds.labels)
    loss_weight = _calculate_loss_weight(train_ds.labels, training_hyper_parameters.loss_weight_factors)

    eval_label_counts = _calculate_label_counts(eval_ds.labels)
    eval_loss_weight = _calculate_loss_weight(eval_ds.labels, training_hyper_parameters.loss_weight_factors)

    print()
    print(f'train label counts = {train_label_counts.tolist()}')
    print(f'eval label counts  = {eval_label_counts.tolist()}\n')
    print(f'loss weights                    = {[round(weight, 2) for weight in loss_weight.tolist()]}')
    print(f'eval loss weights (theoretical) = {[round(weight, 2) for weight in eval_loss_weight.tolist()]}')
    print('\n')

    training_run_metrics, best_model, best_metrics = _train_model(
        model,
        train_data_loader,
        eval_data_loader,
        optimizer,
        loss_weight,
        training_hyper_parameters.num_epochs,
        lr_scheduler,
        device
    )

    return model, training_run_metrics, best_model, best_metrics


def _train_model(
        model: M,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_weight: torch.Tensor,
        num_epochs: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR],
        device: torch.device
) -> tuple[TrainingRunMetrics, M, TrainAndEvaluationMetrics]:
    if num_epochs < 1:
        raise ValueError(f'{num_epochs = } has to be bigger than 0!')

    model.train()
    criterion = nn.CrossEntropyLoss(weight=loss_weight)

    best_score: float = -np.inf
    best_model: nn.Module = copy.deepcopy(model)
    best_metrics: Optional[TrainAndEvaluationMetrics] = None

    training_run_metrics: TrainingRunMetrics = []

    for epoch_nr in range(1, num_epochs + 1):
        train_metrics, validation_metrics = train_epoch(
            epoch_nr,
            model,
            criterion,
            train_data_loader,
            eval_data_loader,
            optimizer,
            device
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f'Training Epoch {epoch_nr:>3}/{num_epochs:<3}: lr = {_get_lr(optimizer):.2E}, {train_metrics}')
        print(f'Evaluation Epoch {epoch_nr:>3}/{num_epochs:<3}: {validation_metrics}')

        score = validation_metrics.score
        if score >= best_score:
            best_score = score
            best_model = copy.deepcopy(model)
            best_metrics = (train_metrics, validation_metrics)

        training_run_metrics.append((train_metrics, validation_metrics))


    print('Training finished')
    return training_run_metrics, best_model, best_metrics


def train_epoch(
        epoch_nr: int,
        model: M,
        criterion: nn.CrossEntropyLoss,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> TrainAndEvaluationMetrics:
    metrics_collector = LabelCollector()

    for data, labels in train_data_loader:
        data, labels = _transpose_all(data, labels, dim0=0, dim1=1)
        data, labels = data.float().to(device), labels.long().to(device)

        optimizer.zero_grad()

        pred = model.forward(data)

        pred, labels = _reshape_for_loss(pred, labels)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        pred_labels: torch.Tensor = pred.argmax(dim=1).view(-1).long()
        metrics_collector.update(
            loss.detach().item(),
            pred_labels.cpu().detach().numpy(),
            labels.cpu().detach().numpy()
        )

    train_metrics = metrics_collector.generate_metrics(epoch_nr)
    eval_metrics = evaluate_model(
        model,
        eval_data_loader,
        device,
        show_confmat=False
    )
    eval_metrics.epoch = epoch_nr

    return train_metrics, eval_metrics


def evaluate_model(
        model: M,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        show_confmat: bool = True,
        confmat_title: str = None,
) -> Metrics:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    metrics_collector = LabelCollector()

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = _transpose_all(data, labels, dim0=0, dim1=1)
            data, labels = data.float().to(device), labels.long().to(device)

            pred = model.forward(data)

            pred, labels = _reshape_for_loss(pred, labels)

            pred_labels: torch.Tensor = pred.argmax(dim=1).view(-1).long()
            metrics_collector.update(
                criterion(pred, labels).detach().item(),
                pred_labels.cpu().detach().numpy(),
                labels.cpu().detach().numpy()
            )

    metrics = metrics_collector.generate_metrics()

    if show_confmat:
        confmat_title += f' - bacc={metrics.bacc:.4f} - score={metrics.score:.4f}'

        display_confmat(
            metrics_collector.pred_labels,
            metrics_collector.target_labels,
            confmat_title
        )

    return metrics


def _transpose_all(*tensors: torch.Tensor, dim0: int, dim1: int):
    return tuple((torch.transpose(x, dim0, dim1) for x in tensors))


def _reshape_for_loss(
        pred: torch.Tensor,
        target_expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.flatten(pred, end_dim=1)
    target_expected = torch.flatten(target_expected)
    return pred, target_expected


def _calculate_label_counts(labels: np.ndarray) -> torch.Tensor:
    return torch.bincount(torch.flatten(torch.Tensor(labels).long()))


def _calculate_loss_weight(labels: np.ndarray, loss_weight_factors: Optional[torch.Tensor]) -> torch.Tensor:
    counts = _calculate_label_counts(labels)
    counts_max = counts.max()
    weights = (counts_max / counts).to(get_torch_device())
    if loss_weight_factors is not None:
        weights *= loss_weight_factors
    return weights


def _get_current_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H.%M')


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _create_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        training_hyper_parameters: TrainingHyperParameters,
) -> Optional[torch.optim.lr_scheduler.MultiStepLR]:
    if training_hyper_parameters.lr_scheduler_provider is None:
        return None
    return training_hyper_parameters.lr_scheduler_provider(
        optimizer,
        training_hyper_parameters.lr_scheduler_milestones,
        training_hyper_parameters.lr_scheduler_gamma
    )
