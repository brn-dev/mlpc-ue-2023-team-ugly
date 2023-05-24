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


# TODO: should be in training which should be called cv training
def create_folds(
        data: np.ndarray,
        labels: np.ndarray,
        n_folds: int,
        cv_folds_permute_seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:

    # -> (Birds, Folds, N (Samples per fold per bird), Sequence length, Dimensions)
    data_folds = data.copy().reshape((N_BIRDS, n_folds, -1, data.shape[-2], data.shape[-1]))
    labels_folds = labels.copy().reshape((N_BIRDS, n_folds, -1, labels.shape[-1]))

    if cv_folds_permute_seed is not None:
        rng = np.random.default_rng(seed=cv_folds_permute_seed)
        for bird_nr in range(N_BIRDS):
            bird_permutation = rng.permutation(n_folds)
            data_folds[bird_nr, :] = data_folds[bird_nr, bird_permutation]
            labels_folds[bird_nr, :] = labels_folds[bird_nr, bird_permutation]

    # -> (F, B, N, S, D)
    data_folds = np.swapaxes(data_folds, 0, 1)
    labels_folds = np.swapaxes(labels_folds, 0, 1)

    # -> (F, B * N, S, D)
    data_folds = data_folds.reshape((n_folds, -1, data.shape[-2], data.shape[-1]))
    labels_folds = labels_folds.reshape((n_folds, -1, labels.shape[-1]))

    return data_folds, labels_folds
