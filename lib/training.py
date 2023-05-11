import os.path
from collections import Callable
from typing import Any
from tqdm import tqdm
import dill as pkl
import numpy as np
from lib.data_preprocessing import normalize_data
import torch
from lib.fnn import FNN

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
        train_func: CreateAndTrainFunc,
        n_folds=10):

    os.makedirs('scores', exist_ok=True)
    valid_accuracies = np.array([])
    valid_b_accuracies = np.array([])
    train_accuracies = np.array([])
    train_b_accuracies = np.array([])
    clf = None
    optim = None
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(n_folds):

        data_validation = data_folds[fold]
        labels_validation = labels_folds[fold]

        data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
        labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, labels_folds.shape[-1]))

        data_train_normalized, data_validation_normalized = normalize_data(data_train, data_validation)

        torch.manual_seed(69)
        optim = torch.optim.Adam(clf.parameters(), lr=1e-3)
        clf = FNN(data_train_normalized.shape[-1], 7).to(device=target_device)

        clf, optim, performances = train_func(clf, optim, data_train_normalized, labels_train, data_validation_normalized, labels_validation, 10, target_device)
        valid_accuracies = np.append(valid_accuracies, performances['valid']['acc'])
        valid_b_accuracies = np.append(valid_b_accuracies, performances['valid']['b_acc'])
        train_accuracies = np.append(train_accuracies, performances['train']['acc'])
        train_b_accuracies = np.append(train_b_accuracies, performances['train']['b_acc'])

    accuracies = dict({'valid_accuracy': valid_accuracies.mean(),
                       'valid_b_accuracy': valid_b_accuracies.mean(),
                       'train_accuracy': train_accuracies.mean(),
                       'train_b_accuracy': train_b_accuracies.mean()})

    model = dict({'accuracies': accuracies,
                  'model_params': clf.state_dict(),
                  'optim_params': optim.state_dict()})

    with open(os.path.join('scores', 'fnn_val.pkl'), 'wb') as f:
        pkl.dump(model, f)

    return model


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
