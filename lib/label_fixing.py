import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

def _count_labels(
        labels: np.ndarray
):
    values, counts = np.unique(labels, return_counts=True)

    if values[0] == 0:
        values = values[1:]
        counts = counts[1:]

    return values, counts

def _calc_entropy(
        labels: np.ndarray
) -> float:
    values, counts = _count_labels(labels)
    return entropy(counts)


def _assign_majority_label(
        labels: np.ndarray
) -> np.ndarray:
    values, counts = _count_labels(labels)

    if len(values) == 0:
        return labels

    majority_label = values[np.argmax(counts)]
    labels[labels != 0] = majority_label

    return labels


def _calc_information_gain_for_splitting_point(
        labels: np.ndarray,
        splitting_point: int
) -> float:
    labels_left, labels_right = labels[:splitting_point], labels[splitting_point:]
    entropy_left, entropy_right = _calc_entropy(labels_left), _calc_entropy(labels_right)

    information_gain = _calc_entropy(labels)

    information_gain -= labels_left.shape[0] / labels.shape[0] * entropy_left
    information_gain -= labels_right.shape[0] / labels.shape[0] * entropy_right

    return information_gain


def _find_splitting_point_with_highest_information_gain(
        window: np.ndarray,
        splitting_point_window_shrink: int,
        split_at_0_only: bool
) -> tuple[int, float]:
    window_size = window.shape[0]

    best_split_index, best_split_information_gain = -1, -1.0

    for i in range(splitting_point_window_shrink, window_size - splitting_point_window_shrink):
        if split_at_0_only and window[i] != 0:
            continue

        information_gain = _calc_information_gain_for_splitting_point(window, i)

        if information_gain > best_split_information_gain:
            best_split_index, best_split_information_gain = i, information_gain

    return best_split_index, best_split_information_gain


def fix_labels_information_gain(
        labels: np.ndarray,
        window_size: int,
        window_overlap: int,
        splitting_point_window_shrink: int,
        split_at_0_only: bool,
        information_gain_threshold: float,
) -> np.ndarray:
    n_sequences, sequence_length = labels.shape

    fixed_labels = np.copy(labels).astype(int)

    num_skipped_windows = 0

    for sequence_nr in tqdm(range(n_sequences), desc='Fixing label sequences'):
        for window_start in range(0, sequence_length - window_size, window_size - window_overlap):
            window = labels[sequence_nr, window_start:window_start + window_size].copy()
            best_split_index, best_split_information_gain = \
                _find_splitting_point_with_highest_information_gain(
                    window,
                    splitting_point_window_shrink,
                    split_at_0_only
                )

            if best_split_index == -1 or best_split_information_gain < information_gain_threshold:
                num_skipped_windows += 1
                continue

            window_left, window_right = window[:best_split_index], window[best_split_index:]
            window_left = _assign_majority_label(window_left)
            window_right = _assign_majority_label(window_right)

            window = np.concatenate((window_left, window_right))
            fixed_labels[sequence_nr, window_start:window_start + window_size] = window

    print(f'Skipped {num_skipped_windows} out of '
          f'{n_sequences * (sequence_length - window_size) // (window_size - window_overlap)} windows')

    return fixed_labels