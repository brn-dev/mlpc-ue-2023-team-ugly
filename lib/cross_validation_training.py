from collections import Callable

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from lib.data_preprocessing import normalize_data
from lib.ds.numpy_dataset import NumpyDataset

CVTrainFunc = Callable[[int, NumpyDataset, NumpyDataset, StandardScaler], None]



def create_cross_validation_folds(
        dataset: NumpyDataset,
        n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    :param dataset: data shape (N sequences, Sequence length, Dimensions)
    :param n_folds:
    :return list of (train_ds, eval_ds) tuples, datasets have same shape as input dataset
    """
    data, labels = dataset
    n_sequences = data.shape[0]
    sequence_length = data.shape[1]
    dimensions = data.shape[2]

    assert data.shape[0] == labels.shape[0]
    assert data.shape[1] == labels.shape[1]
    assert n_sequences % n_folds == 0, f'{n_sequences = } must be divisible by {n_folds = }'

    # -> (F, N / F, S, D)
    data_folds = data.reshape((n_folds, n_sequences // n_folds, sequence_length, dimensions))
    labels_folds = labels.reshape((n_folds, n_sequences // n_folds, sequence_length))

    return data_folds, labels_folds


def train_with_cv(
        dataset: NumpyDataset,
        train_func: CVTrainFunc,
        n_folds,
):
    print(f'Creating {n_folds} folds')
    data_folds, labels_folds = create_cross_validation_folds(dataset, n_folds)

    n_sequences, sequence_length, dimensions = data_folds.shape[1:]

    for fold_nr in tqdm(range(n_folds), desc='CV Folds'):
        print(f'\n\n\nExecuting CV for fold {fold_nr}')

        eval_idx = fold_nr
        train_indices = np.setdiff1d(range(n_folds), eval_idx)

        data_train = data_folds[train_indices].reshape((-1, sequence_length, dimensions))
        labels_train = labels_folds[train_indices].reshape((-1, sequence_length))

        data_eval = data_folds[eval_idx]
        labels_eval = labels_folds[eval_idx]

        data_train_normalized, data_validation_normalized, normalization_scaler = normalize_data(
            data_train,
            data_eval
        )

        train_func(
            fold_nr,
            NumpyDataset(data_train_normalized, labels_train),
            NumpyDataset(data_validation_normalized, labels_eval),
            normalization_scaler
        )
