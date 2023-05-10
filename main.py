import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split, create_folds
from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data

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

    def create_and_train_func(data: np.ndarray, labels: np.ndarray, seed=6942066):
        clf = SVC(verbose=True, random_state=seed)
        
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        print(f"{data.shape= } {labels.shape= }")
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        label_pred = clf.predict(data)
        print(f"\nMean Accuracy: {clf.score(data, labels) }")
        print(f"Balanced Accuracy: {balanced_accuracy_score(labels, label_pred)}")
        
    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)

    train_with_cv(data_train, labels_train, create_and_train_func, eval_func)


if __name__ == '__main__':
    main()

