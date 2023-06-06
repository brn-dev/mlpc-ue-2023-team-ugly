import numpy as np


def to_binary(
        labels: np.ndarray,
        pos_labels: np.ndarray,
        neg_labels: np.ndarray,
) -> np.ndarray:
    # noinspection PyUnresolvedReferences
    assert np.intersect1d(pos_labels, neg_labels).size == 0

    pos_mask = np.isin(labels, pos_labels)
    neg_mask = np.isin(labels, neg_labels)

    binary_labels: np.ndarray = labels.copy().astype(int)

    binary_labels[pos_mask] = 1
    binary_labels[neg_mask] = 0

    return binary_labels

