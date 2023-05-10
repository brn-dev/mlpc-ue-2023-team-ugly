from collections import Callable
from typing import Any
from tqdm import tqdm
import dill as pkl

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
        n_folds=10):

    model_scores = dict()
    for k in range(5, 100, 5):
        scores = np.array([])
        for fold in tqdm(range(n_folds)):
            data_validation = data_folds[fold]
            labels_validation = labels_folds[fold]

            data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
                .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
            labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
                .reshape((-1, labels_folds.shape[-1]))

            data_train_normalized, data_validation_normalized = normalize_data(data_train, data_validation)

            model = create_and_train_func(data_train_normalized, labels_train, k=k)
            score = eval_func(model, data_validation_normalized, labels_validation)
            scores = np.append(scores, score)

        model_scores[f'knn_{k}'] = scores.mean()


def train_best(
        data_train: np.ndarray,
        labels_train: np.ndarray,
        data_test: np.ndarray,
        labels_test: np.ndarray,
        create_and_train_func: CreateAndTrainFunc,
        eval_func: EvalFunc,
        k: int
):

    data_train_normalized, data_test_normalized = normalize_data(data_train, data_test)
    model = create_and_train_func(data_train_normalized, labels_train, k=k)
    with open('knn.pkl', 'wb') as f:
        pkl.dump(model, f)
    print(eval_func(model, data_test_normalized, labels_test))
