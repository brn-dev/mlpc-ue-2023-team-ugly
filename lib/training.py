from collections import Callable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from lib.ds.dataset_splitting import create_folds
from lib.data_preprocessing import normalize_data
from lib.ds.numpy_dataset import NumpyDataset
from lib.training_hyper_parameters import TrainingHyperParameters

CVTrainFunc = Callable[[int, NumpyDataset, NumpyDataset, StandardScaler], None]


def train_with_cv(
        dataset: NumpyDataset,
        train_func: CVTrainFunc,
        n_folds=10
):
    data_folds, labels_folds = create_folds(dataset.data, dataset.labels, n_folds)

    for fold in range(n_folds):
        data_validation = data_folds[fold]
        labels_validation = labels_folds[fold]

        data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
        labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, labels_folds.shape[-1]))

        data_train_normalized, data_validation_normalized, normalization_scaler = normalize_data(
            data_train,
            data_validation
        )

        train_func(
            fold,
            NumpyDataset(data_train_normalized, labels_train),
            NumpyDataset(data_validation_normalized, labels_validation),
            normalization_scaler
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_lr_scheduler(optimizer: torch.optim.Optimizer, training_hyper_parameters: TrainingHyperParameters):
    if training_hyper_parameters.lr_scheduler_provider is None:
        return None
    return training_hyper_parameters.lr_scheduler_provider(
        optimizer,
        training_hyper_parameters.lr_scheduler_milestones,
        training_hyper_parameters.lr_scheduler_gamma
    )
