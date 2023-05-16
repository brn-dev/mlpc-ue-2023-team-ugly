from collections import Callable

import numpy as np

from lib.ds.dataset_splitting import create_folds
from lib.data_preprocessing import normalize_data
from lib.ds.numpy_dataset import NumpyDataset

CVTrainFunc = Callable[[int, NumpyDataset, NumpyDataset], None]


def train_with_cv(
        dataset: NumpyDataset,
        train_func: CVTrainFunc,
        n_folds=10
):
    data_folds, labels_folds = create_folds(dataset.data, dataset.labels, n_folds)

    for fold in range(n_folds):
        data_validation = data_folds[fold]
        labels_validation = labels_folds[fold]

        data_train = data_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, data_folds.shape[-2], data_folds.shape[-1]))
        labels_train = labels_folds[np.setdiff1d(range(n_folds), fold)] \
            .reshape((-1, labels_folds.shape[-1]))

        data_train_normalized, data_validation_normalized = normalize_data(data_train, data_validation)

        train_func(
            fold,
            NumpyDataset(data_train_normalized, labels_train),
            NumpyDataset(data_validation_normalized, labels_validation)
        )




