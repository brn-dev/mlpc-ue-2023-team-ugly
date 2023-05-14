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
        n_folds
) -> tuple[np.ndarray, np.ndarray]:
    n_files = data.shape[0]
    files_per_fold = n_files // n_folds

    files_per_bird = n_files // N_BIRDS
    files_per_bird_per_fold = files_per_bird // n_folds

    data_folds = np.ndarray((n_folds, files_per_fold, data.shape[1], data.shape[2]))
    labels_folds = np.ndarray((n_folds, files_per_fold, labels.shape[1]))

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


def split_by_1d(
        data: np.ndarray,
        labels: np.ndarray,
        test_size_pct=0.2,
        seed=6942066
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    # get indices for splitting into train and test set
    test_idxs = np.random.choice(range(len(labels)), int(len(labels) * test_size_pct), replace=False)
    train_idxs = np.delete(np.asarray(range(len(labels))), test_idxs)

    # splitting into train and test set
    data_train = data[train_idxs]
    labels_train = labels[train_idxs]
    data_test = data[test_idxs]
    labels_test = labels[test_idxs]

    return data_train, labels_train, data_test, labels_test


def redistribute_labels(data, labels, seed: int = 6942066):
    np.random.seed(seed)

    labels_flat = labels.flatten()  # 1d array
    data_flat = data.reshape((-1, data.shape[-1]))  # 2d array
    labels_unique = sorted(np.unique(labels_flat))

    classes_split = [np.argwhere(labels_flat == l) for l in labels_unique]
    smallest_class = np.argmin([len(c) for c in classes_split])

    smallest_class_k = len(classes_split[smallest_class])

    randomized_samples = [np.random.choice(c.flatten(), smallest_class_k, replace=False) \
                          for c in classes_split]

    randomized_samples = np.asarray(randomized_samples)

    randomized_labels = labels_flat[randomized_samples.flatten()]
    randomized_data = data_flat[randomized_samples.flatten()]

    randomized_data, randomized_labels = unison_shuffled_copies(randomized_data, randomized_labels)

    randomized_data = randomized_data.reshape((len(labels_unique), smallest_class_k, data.shape[-1]))

    # print(labels_flat[randomized_samples.flatten()][4600:4800])

    print(f"{randomized_data.shape=} {randomized_labels.shape=}")

    return randomized_data, randomized_labels.reshape((len(labels_unique), smallest_class_k))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
