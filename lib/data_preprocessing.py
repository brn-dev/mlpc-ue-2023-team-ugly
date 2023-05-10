import numpy as np
from sklearn.preprocessing import StandardScaler

CORRELATION_THRESHOLD = 0.90

def find_correlated_columns_to_drop(data: np.ndarray) -> np.ndarray:
    data_flattened = data.reshape((-1, data.shape[-1]))
    corr_matrix_abs = np.abs(np.corrcoef(data_flattened, rowvar=False))

    corr_matrix_upper_tri = np.triu(corr_matrix_abs, k=1)

    return np.array([
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
    data_train_flattened = data_train.reshape((-1, data_train.shape[-1]))
    data_test_flattened = data_test.reshape((-1, data_test.shape[-1]))

    scaler = StandardScaler()

    scaler.fit(data_train_flattened)

    data_train_scaled = scaler.transform(data_train_flattened)
    data_test_scaled = scaler.transform(data_test_flattened)

    return data_train_scaled.reshape(data_train.shape), data_test_scaled.reshape(data_test.shape)


