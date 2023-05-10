from collections import Callable
from typing import Any
from tqdm import tqdm

import numpy as np

from lib.ds.dataset_splitting import create_folds
from lib.data_preprocessing import normalize_data

# Parameters: train_data, train_labels; Returns: Model
CreateAndTrainFunc = Callable[[np.ndarray, np.ndarray], Any]

# Parameters: validation_data, validation_labels
EvalFunc = Callable[[Any, np.ndarray, np.ndarray], None]


def get_baseline(labels: np.ndarray) -> float:
    labels = labels.flatten()
    unique_labels, labels_count = np.unique(labels, return_counts=True)
    return max(labels_count) / len(labels)


def train_with_cv(
        data_folds: np.ndarray,
        labels_folds: np.ndarray,
        create_and_train_func: CreateAndTrainFunc,
        eval_func: EvalFunc,
        n_folds=10
):

    for fold in tqdm(range(n_folds)):
        data_validation = data_folds[fold]
        labels_validation = labels_folds[fold]
        base_acc = get_baseline(labels_validation)
        print(f'Baseline of fold {fold} = {base_acc}')

        data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
        labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, labels_folds.shape[-1]))

        data_train_normalized, data_validation_normalized = normalize_data(data_train, data_validation)

        model = create_and_train_func(data_train_normalized, labels_train)
        print(f'Validation accuracy of fold {fold}')
        eval_func(model, data_validation_normalized, labels_validation)
