import numpy as np
from sklearn.tree import DecisionTreeClassifier

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split, create_folds
from lib.training import train_with_cv

def main():
    # x = load_all_data('dataset')[1].max(axis=1)
    # print(x.shape)
    # print(x)

    data_train, labels_train, data_test, labels_test = split(*load_all_data('dataset'), seed=7890)
    # print(data_train.shape)
    # print(data_train)
    # print(labels_train.shape)
    # print(labels_train)
    # print(data_test.shape)
    # print(data_test)
    # print(labels_test.shape)
    # print(labels_test)
    # data_train_folds, labels_train_folds = create_folds(data_train, labels_train)
    # print(data_train_folds.shape)
    # print(data_train_folds)
    # print(labels_train_folds.shape)
    # print(labels_train_folds)

    def create_and_train_func(data: np.ndarray, labels: np.ndarray):
        clf = DecisionTreeClassifier()
        data = data.reshape((-1, data.shape[-1]))[:, :20]
        labels = labels.flatten()
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        data = data.reshape((-1, data.shape[-1]))[:, :20]
        labels = labels.flatten()
        print(clf.score(data, labels))


    train_with_cv(data_train, labels_train, create_and_train_func, eval_func)


if __name__ == '__main__':
    main()

