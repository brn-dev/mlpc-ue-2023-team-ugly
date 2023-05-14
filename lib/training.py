import os.path
from collections import Callable
from typing import Any
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import dill as pkl
import numpy as np
from lib.data_preprocessing import normalize_data
import torch
from lib.fnn import FNN1, FNN2, FNN3, FNN4, FNN5

models = [FNN1, FNN3, FNN4, FNN5]

# Parameters: train_data, train_labels; Returns: Model
CreateAndTrainFunc = Callable[[np.ndarray, np.ndarray], Any]

# Parameters: validation_data, validation_labels
EvalFunc = Callable[[Any, np.ndarray, np.ndarray], None]


def get_baseline(labels: np.ndarray) -> float:
    labels = labels.flatten()
    unique_labels, labels_count = np.unique(labels, return_counts=True)
    return max(labels_count) / len(labels)


def train_with_cv(
        data_train: np.ndarray,
        data_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
        train_func: CreateAndTrainFunc,
        n_folds=10):

    os.makedirs('scores', exist_ok=True)
    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_model_accuracies = []

    for j, clf_type in enumerate(models):
        valid_accuracies = np.array([])
        valid_b_accuracies = np.array([])
        train_accuracies = np.array([])
        train_b_accuracies = np.array([])
        skf = StratifiedKFold(n_splits=n_folds)
        for i, (train_idx, val_idx) in enumerate(skf.split(data_train, labels_train)):

            data_validation = data_train[val_idx]
            labels_validation = labels_train[val_idx]

            data_train_split = data_train[train_idx]
            labels_train_split = labels_train[train_idx]

            data_train_normalized, data_validation_normalized = normalize_data(data_train_split, data_validation)

            torch.manual_seed(69)
            clf = clf_type(data_train_normalized.shape[-1], 7).to(device=target_device)
            optim = torch.optim.Adamax(clf.parameters(), lr=1e-3, weight_decay=1e-2)

            clf, optim, performances = train_func(clf, optim, data_train_normalized, labels_train_split, data_validation_normalized, labels_validation, 30, target_device)
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

        with open(os.path.join('fnn', 'scores_new_data', f'fnn{j + 1}_train.pkl'), 'wb') as f:
            pkl.dump(model, f)

        all_model_accuracies.append(model)

    all_b_accuracies = [model['accuracies']['valid_b_accuracy'] for model in all_model_accuracies]

    data_train_normalized, data_test_normalized = normalize_data(data_train_split, data_test)

    torch.manual_seed(69)
    best_model_idx = np.argmax(all_b_accuracies)
    best_model = models[best_model_idx](data_train_normalized.shape[-1], 7).to(device=target_device)
    optim = torch.optim.Adamax(best_model.parameters(), lr=1e-3, weight_decay=1e-2)

    clf, optim, performances = train_func(best_model, optim, data_train_normalized, labels_train_split, data_test_normalized, labels_test, 30,
                                          target_device, confusion_bool=True)

    best_model = dict({'accuracies': performances,
                       'best_model_params': clf.state_dict(),
                       'optim_params': optim.state_dict()})

    with open(os.path.join('fnn', 'scores_new_data', f'fnn_test.pkl'), 'wb') as f:
        pkl.dump(best_model, f)

    return best_model
