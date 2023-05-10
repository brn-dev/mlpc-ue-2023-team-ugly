from collections import Callable
from typing import Any

import numpy as np

from lib.ds.dataset_splitting import create_folds
from lib.data_preprocessing import normalize_data
from lib.data_preprocessing import remove_correlated_columns

# Parameters: train_data, train_labels; Returns: Model
CreateAndTrainFunc = Callable[[np.ndarray, np.ndarray], Any]

# Parameters: validation_data, validation_labels
EvalFunc = Callable[[Any, np.ndarray, np.ndarray], None]


def train_with_cv(
        data: np.ndarray,
        labels: np.ndarray,
        create_and_train_func: CreateAndTrainFunc,
        eval_func: EvalFunc,
        n_folds=10
):
    data_folds, labels_folds = create_folds(data, labels, n_folds)

    for fold in range(n_folds):
        data_validation = data_folds[fold]
        labels_validation = labels_folds[fold]

        data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
        labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, labels_folds.shape[-1]))
        
        data_train, data_validation = remove_correlated_columns(data_train, data_validation)

        data_train_normalized, data_validation_normalized = normalize_data(data_train, data_validation)

        model = create_and_train_func(data_train_normalized, labels_train)
        eval_func(model, data_validation_normalized, labels_validation)




