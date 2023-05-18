from typing import Optional

import numpy as np

from lib.ds.dataset_loading import BIRD_NAMES

N_BIRDS = len(BIRD_NAMES)


def split(
        data: np.ndarray,
        labels: np.ndarray,
        test_size_pct=0.2,
        seed=6942066
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    files_per_bird = data.shape[0] // N_BIRDS
    test_files_per_bird = int(files_per_bird * test_size_pct)

    data_train = np.ndarray((0, data.shape[1], data.shape[2]))
    data_test = np.ndarray((0, data.shape[1], data.shape[2]))

    labels_train = np.ndarray((0, labels.shape[1]))
    labels_test = np.ndarray((0, labels.shape[1]))

    for i in range(N_BIRDS):
        test_indices = np.random.choice(range(files_per_bird), test_files_per_bird, replace=False)
        train_indices = np.setdiff1d(np.array(range(files_per_bird)), test_indices)
        test_indices += i * files_per_bird
        train_indices += i * files_per_bird

        data_test = np.append(data_test, data[test_indices], axis=0)
        labels_test = np.append(labels_test, labels[test_indices], axis=0)

        data_train = np.append(data_train, data[train_indices], axis=0)
        labels_train = np.append(labels_train, labels[train_indices], axis=0)

    return data_train, labels_train, data_test, labels_test


def create_folds(
        data: np.ndarray,
        labels: np.ndarray,
        n_folds: int,
        cv_folds_permute_seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    n_files = data.shape[0]
    files_per_fold = n_files // n_folds

    files_per_bird = n_files // N_BIRDS
    files_per_bird_per_fold = files_per_bird // n_folds

    data_folds = np.ndarray((n_folds, files_per_fold, data.shape[1], data.shape[2]))
    labels_folds = np.ndarray((n_folds, files_per_fold, labels.shape[1]))

    birds_folds_permutations = np.repeat(np.arange(N_BIRDS)[np.newaxis, :], N_BIRDS, axis=0)
    if cv_folds_permute_seed is not None:
        rng = np.random.default_rng(seed=cv_folds_permute_seed)
        for bird_nr in range(N_BIRDS):
            birds_folds_permutations[bird_nr] = rng.permutation(N_BIRDS)

    for fold in range(n_folds):
        for bird in range(N_BIRDS):
            data_folds[fold, files_per_bird_per_fold * bird:files_per_bird_per_fold * (bird + 1)] = \
                data[
                    files_per_bird * bird + files_per_bird_per_fold * fold
                    :
                    files_per_bird * bird + files_per_bird_per_fold * (fold + 1)
                ]

            labels_folds[fold, files_per_bird_per_fold * bird:files_per_bird_per_fold * (bird + 1)] = \
                labels[
                    files_per_bird * bird + files_per_bird_per_fold * fold
                    :
                    files_per_bird * bird + files_per_bird_per_fold * (fold + 1)
                ]

    return data_folds, labels_folds
