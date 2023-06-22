from typing import Optional

import numpy as np

from lib.ds.dataset_loading import BIRD_NAMES
from lib.ds.numpy_dataset import NumpyDataset

N_BIRDS = len(BIRD_NAMES)

def split_sequential(
        dataset: NumpyDataset,
        test_size_pct: float,
) -> tuple[NumpyDataset, Optional[NumpyDataset]]:

    data: np.ndarray
    labels: np.ndarray
    data, labels = dataset

    if test_size_pct == 0:
        return dataset.copy(), None

    n_sequences, sequence_length, n_features = data.shape

    data = data.reshape((N_BIRDS, n_sequences // N_BIRDS, sequence_length, n_features))
    labels = labels.reshape((N_BIRDS, n_sequences // N_BIRDS, sequence_length))

    n_sequences_per_bird = n_sequences // N_BIRDS
    n_test_sequences_per_bird = int(n_sequences_per_bird * test_size_pct)

    data_test = data[:, :n_test_sequences_per_bird, :, :].reshape(-1, sequence_length, n_features)
    labels_test = labels[:, :n_test_sequences_per_bird, :].reshape(-1, sequence_length)

    data_train = data[:, n_test_sequences_per_bird:, :, :].reshape(-1, sequence_length, n_features)
    labels_train = labels[:, n_test_sequences_per_bird:, :].reshape(-1, sequence_length)

    return NumpyDataset(data_train, labels_train), NumpyDataset(data_test, labels_test)


def split_random(
        dataset: NumpyDataset,
        test_size_pct=0.2,
        seed=6942066
) -> tuple[NumpyDataset, Optional[NumpyDataset]]:
    np.random.seed(seed)

    data, labels = dataset

    if test_size_pct == 0:
        return dataset.copy(), None

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

    return NumpyDataset(data_train, labels_train), NumpyDataset(data_test, labels_test)
