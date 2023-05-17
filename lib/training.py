import os.path
from collections import Callable
from typing import Any
from tqdm import tqdm
import dill as pkl
import numpy as np
from lib.data_preprocessing import normalize_data

# Parameters: train_data, train_labels; Returns: Model
CreateAndTrainFunc = Callable[[np.ndarray, np.ndarray], Any]

# Parameters: validation_data, validation_labels
EvalFunc = Callable[[Any, np.ndarray, np.ndarray], None]

score_path = os.path.join('knn', 'scores_90corr.pkl')

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

    os.makedirs('scores', exist_ok=True)
    all_models = dict()

    for k in tqdm(range(1, 21)):
        valid_accuracies = np.array([])
        valid_b_accuracies = np.array([])
        train_accuracies = np.array([])
        train_b_accuracies = np.array([])
        accuracies = dict()
        for fold in range(n_folds):
            data_validation = data_folds[fold]
            labels_validation = labels_folds[fold]

            data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
                .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
            labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
                .reshape((-1, labels_folds.shape[-1]))

            data_train_normalized, data_validation_normalized = normalize_data(data_train, data_validation)

            model = create_and_train_func(data_train_normalized, labels_train, k=k)
            valid_acc, valid_b_acc = eval_func(model, data_validation_normalized, labels_validation)
            train_acc, train_b_acc = eval_func(model, data_train_normalized, labels_train)
            valid_accuracies = np.append(valid_accuracies, valid_acc)
            valid_b_accuracies = np.append(valid_b_accuracies, valid_b_acc)
            train_accuracies = np.append(train_accuracies, train_acc)
            train_b_accuracies = np.append(train_b_accuracies, train_b_acc)

        accuracies['valid'] = {'acc': valid_accuracies.mean(),
                               'b_acc': valid_accuracies.mean()}
        accuracies['train'] = {'acc': train_accuracies.mean(),
                               'b_acc': train_b_accuracies.mean()}

        all_models[f'knn{k}'] = accuracies

    with open(score_path, 'wb') as f:
        pkl.dump(all_models, f)


def train_best(
        data_train: np.ndarray,
        labels_train: np.ndarray,
        data_test: np.ndarray,
        labels_test: np.ndarray,
        create_and_train_func: CreateAndTrainFunc,
        eval_func: EvalFunc,
        k: int):

    data_train_normalized, data_test_normalized = normalize_data(data_train, data_test)
    model = create_and_train_func(data_train_normalized, labels_train, k=k)
    with open(os.path.join('scores_select150best', 'knn.pkl'), 'wb') as f:
        pkl.dump(model, f)
    print(eval_func(model, data_test_normalized, labels_test))
