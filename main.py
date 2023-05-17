import numpy as np
import os
import dill as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from lib.training import train_with_cv, train_best


def main():
    data_train_folds_down = np.load(os.path.join('np_data', 'data_train_folds_down.npy'))
    labels_data_train_folds_down = np.load(os.path.join('np_data', 'labels_train_folds_down.npy'))

    data_train_down = np.load(os.path.join('np_data_select150best', 'data_train_down.npy'))
    labels_train = np.load(os.path.join('np_data_select150best', 'labels_train.npy'))
    data_test_down = np.load(os.path.join('np_data_select150best', 'data_test_down.npy'))
    labels_test = np.load(os.path.join('np_data_select150best', 'labels_test.npy'))

    # with open(os.path.join('scores', 'knn_accuracies.pkl'), 'rb') as f:
    # accuracies = pkl.load(f)
    with open(os.path.join('scores_select150best', 'knn_valid_balanced_accuracies.pkl'), 'rb') as f:
        valid_balanced_acc = pkl.load(f)

    def create_and_train_func(data: np.ndarray, labels: np.ndarray, k=5):
        clf = KNeighborsClassifier(n_neighbors=k)
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        acc = clf.score(data, labels)
        b_acc = balanced_accuracy_score(labels, clf.predict(data))
        return acc, b_acc

    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)

    train_with_cv(data_train_folds_down, labels_data_train_folds_down, create_and_train_func, eval_func)
    #best_knn_valid_acc = int(max(valid_accuracies, key=valid_accuracies.get)[4:])
    best_knn_valid_b_acc = int(max(valid_balanced_acc, key=valid_balanced_acc.get)[4:])
    #best_knn_train_acc = int(max(train_accuracies, key=train_accuracies.get)[4:])
    #best_knn_train_b_acc = int(max(train_balanced_acc, key=train_balanced_acc.get)[4:])
   # print(f'Best k: {best_knn_valid_acc} | accuracy: {valid_accuracies[f"knn_{best_knn_valid_acc}"]}')
   # print()
   # print(f'Best k: {best_knn_valid_b_acc} | balanced accuracy: {valid_balanced_acc[f"knn_{best_knn_valid_b_acc}"]}')
   # print()
   # print(f'Best k: {best_knn_train_acc} | accuracy: {train_accuracies[f"knn_{best_knn_train_acc}"]}')
   # print()
   # print(f'Best k: {best_knn_train_b_acc} | balanced accuracy: {train_balanced_acc[f"knn_{best_knn_train_b_acc}"]}')
    train_best(
        data_train_down,
        labels_train,
        data_test_down,
        labels_test,
        create_and_train_func,
        eval_func,
        best_knn_valid_b_acc)


if __name__ == '__main__':
    main()
