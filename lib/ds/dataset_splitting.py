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
