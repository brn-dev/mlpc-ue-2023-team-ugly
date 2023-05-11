import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

CORRELATION_THRESHOLD = 0.8

def find_correlated_columns_to_drop(data: np.ndarray) -> np.ndarray:
    data_flattened = data.reshape((-1, data.shape[-1]))
    corr_matrix_abs = np.abs(np.corrcoef(data_flattened, rowvar=False))

    corr_matrix_upper_tri = np.triu(corr_matrix_abs, k=1)

    return np.asarray([
        column_idx
        for column_idx
        in range(corr_matrix_upper_tri.shape[0])
        if (corr_matrix_upper_tri[:, column_idx] > CORRELATION_THRESHOLD).any()
    ])

def remove_correlated_columns(data_train: np.ndarray, data_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correlated_columns_to_drop = find_correlated_columns_to_drop(data_train)

    data_train = data_train[:, :, np.setdiff1d(range(data_train.shape[-1]), correlated_columns_to_drop)]
    data_test = data_test[:, :, np.setdiff1d(range(data_test.shape[-1]), correlated_columns_to_drop)]

    return data_train, data_test

def normalize_data(data_train: np.ndarray, data_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(data_train.shape)
    data_train_flattened = data_train.reshape((-1, data_train.shape[-1]))
    data_test_flattened = data_test.reshape((-1, data_test.shape[-1]))

    scaler = StandardScaler()
    print(data_train_flattened.shape)
    scaler.fit(data_train_flattened)

    data_train_scaled = scaler.transform(data_train_flattened)
    data_test_scaled = scaler.transform(data_test_flattened)

    return data_train_scaled.reshape(data_train.shape), data_test_scaled.reshape(data_test.shape)


def labelsmoothing(enc: OneHotEncoder, labels: np.ndarray, alpha: float=0.3) -> np.ndarray:
    y_hot = enc.transform(labels).toarray()

    y_ls = (1-alpha) * y_hot + alpha / len(enc.categories_[0])

    print(f"{y_hot=}\n{y_ls=}")

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def redistribute_labels(data, labels, seed: int=6942066):
    np.random.seed(seed)

    labels_flat = labels.flatten() #1d array
    data_flat = data.reshape((-1, data.shape[-1])) #2d array
    labels_unique = sorted(np.unique(labels_flat))

    classes_split = [np.argwhere(labels_flat==l) for l in labels_unique]
    smallest_class = np.argmin([len(c) for c in classes_split])

    smallest_class_k = len(classes_split[smallest_class])

    randomized_samples = [np.random.choice(c.flatten(), smallest_class_k, replace=False)\
                          for c in classes_split]
    
    randomized_samples = np.asarray(randomized_samples)

    randomized_labels = labels_flat[randomized_samples.flatten()]
    randomized_data = data_flat[randomized_samples.flatten()]

    randomized_data, randomized_labels = unison_shuffled_copies(randomized_data, randomized_labels)

    randomized_data = randomized_data.reshape((len(labels_unique), smallest_class_k, data.shape[-1]))

    #print(labels_flat[randomized_samples.flatten()][4600:4800])

    print(f"{randomized_data.shape=} {randomized_labels.shape=}")

    return randomized_data, randomized_labels.reshape((len(labels_unique), smallest_class_k))