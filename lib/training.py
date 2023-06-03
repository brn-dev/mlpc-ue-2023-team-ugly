import os.path
from collections import Callable
from typing import Any
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from lib.ds.dataset_loading import flatten
import dill as pkl
import numpy as np
from lib.data_preprocessing import normalize_data
import torch
from lib.fnn import FNN1, FNN2, FNN3, FNN4, FNN5, FNN6

models = [FNN5]
learning_rates = [8e-4]

# Parameters: train_data, train_labels; Returns: Model
CreateAndTrainFunc = Callable[[np.ndarray, np.ndarray], Any]

# Parameters: validation_data, validation_labels
EvalFunc = Callable[[Any, np.ndarray, np.ndarray], None]


def get_baseline(labels: np.ndarray) -> float:
    labels = labels.flatten()
    unique_labels, labels_count = np.unique(labels, return_counts=True)
    return max(labels_count) / len(labels)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
    model_names = [str(clf)[-6:-2].lower() for clf in models]

    # for j, clf_type in enumerate(models):
    #     valid_accuracies = np.array([])
    #     valid_b_accuracies = np.array([])
    #     train_accuracies = np.array([])
    #     train_b_accuracies = np.array([])
    #     for fold in range(n_folds):
    #         data_validation = data_train[fold]
    #         labels_validation = labels_train[fold]
    #
    #         data_train_split = data_train[np.setdiff1d(range(n_folds), fold)] \
    #             .reshape((-1, data_train.shape[-2], data_train.shape[-1]))
    #         labels_train_split = labels_train[np.setdiff1d(range(n_folds), fold)] \
    #             .reshape((-1, labels_train.shape[-1]))
    #
    #         data_train_split, labels_train_split = flatten(data_train_split, labels_train_split)
    #         data_validation, labels_validation = flatten(data_validation, labels_validation)
    #
    #         data_train_normalized, data_validation_normalized = normalize_data(data_train_split, data_validation)
    #
    #         torch.manual_seed(69)
    #         clf = clf_type(data_train_normalized.shape[-1], 7).to(device=target_device)
    #         optim = torch.optim.Adamax(clf.parameters(), lr=learning_rates[j])
    #
    #         clf, optim, performances = train_func(clf, optim, data_train_normalized, labels_train_split,
    #                                               data_validation_normalized, labels_validation, 200, target_device,
    #                                               confusion_bool=True)
    #         valid_accuracies = np.append(valid_accuracies, performances['valid']['acc'])
    #         valid_b_accuracies = np.append(valid_b_accuracies, performances['valid']['b_acc'])
    #         train_accuracies = np.append(train_accuracies, performances['train']['acc'])
    #         train_b_accuracies = np.append(train_b_accuracies, performances['train']['b_acc'])
    #
    #     accuracies = dict({'valid_accuracy': valid_accuracies.mean(),
    #                        'valid_b_accuracy': valid_b_accuracies.mean(),
    #                        'train_accuracy': train_accuracies.mean(),
    #                        'train_b_accuracy': train_b_accuracies.mean()})
    #
    #     model = dict({'accuracies': accuracies,
    #                   'model': model_names[j],
    #                   'lr': get_lr(optim)})
    #
    #     with open(os.path.join('fnn', 'scores_newest_data', f'{model_names[j]}_train.pkl'), 'wb') as f:
    #         pkl.dump(model, f)
    #
    #     all_model_accuracies.append(model)
    #
    # all_b_accuracies = [model['accuracies']['valid_b_accuracy'] for model in all_model_accuracies]

    data_train, labels_train = flatten(data_train, labels_train)
    data_test, labels_test = flatten(data_test, labels_test)
    data_train_normalized, data_test_normalized = normalize_data(data_train, data_test)

    torch.manual_seed(69)
    np.random.seed(69)
    best_model_idx = 0
    print(f'The best model is {model_names[best_model_idx]}!')
    best_model = models[best_model_idx](data_train_normalized.shape[-1], 7).to(device=target_device)
    print(count_parameters(best_model))
    optim = torch.optim.Adamax(best_model.parameters(), lr=learning_rates[best_model_idx])

    clf, optim, performances = train_func(best_model, optim, data_train_normalized, labels_train, data_test_normalized,
                                          labels_test, 1000,
                                          target_device, confusion_bool=True)

    best_model = dict({'accuracies': {'valid_accuracy': performances['valid']['acc'],
                                      'valid_b_accuracy': performances['valid']['b_acc'],
                                      'train_accuracy': performances['train']['acc'],
                                      'train_b_accuracy': performances['train']['b_acc']},
                       'model': model_names[best_model_idx],
                       'lr': get_lr(optim)})

    with open(os.path.join('fnn', 'scores_newest_data', f'{model_names[best_model_idx]}_test.pkl'), 'wb') as f:
        pkl.dump(best_model, f)

    with open(os.path.join('fnn', 'best_model.pkl'), 'wb') as f:
        pkl.dump(clf, f)

    return best_model
