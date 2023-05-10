import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from lib.training import train_with_cv, train_best

def main():

    data_train_folds_down = np.load(os.path.join('np_data', 'data_train_folds_down.npy'))
    labels_data_train_folds_down = np.load(os.path.join('np_data', 'labels_train_folds_down.npy'))

    data_train_down = np.load(os.path.join('np_data', 'data_train_down.npy'))
    labels_train = np.load(os.path.join('np_data', 'labels_train.npy'))
    data_test_down = np.load(os.path.join('np_data', 'data_test_down.npy'))
    labels_test = np.load(os.path.join('np_data', 'labels_test.npy'))

    def create_and_train_func(data: np.ndarray, labels: np.ndarray, k=5):
        clf = KNeighborsClassifier(n_neighbors=k)
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        return clf.score(data, labels)
        
    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)

    results = train_with_cv(data_train_folds_down, labels_data_train_folds_down, create_and_train_func, eval_func)
    best_knn = int(max(results, key=results.get)[5:])
    train_best(
        data_train_down,
        labels_train,
        data_test_down,
        labels_test,
        create_and_train_func,
        eval_func,
        best_knn
    )


if __name__ == '__main__':
    main()
