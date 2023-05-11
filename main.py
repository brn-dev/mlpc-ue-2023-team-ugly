import numpy as np
import matplotlib.pyplot as plt
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lib.ds.dataset_loading import load_all_data
from lib.ds.dataset_splitting import split, create_folds
from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data, labelsmoothing, redistribute_labels

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
        clf = SVC(verbose=True, random_state=seed, kernel="rbf")
        
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        print(f"{data.shape= } {labels.shape= }")
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        result_dict = {}
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        
        label_pred = clf.predict(data)

        clf_score = clf.score(data, labels)
        clf_balanced_acc = balanced_accuracy_score(labels, label_pred)

        print(f"\nMean Accuracy: {clf_score}")
        print(f"Balanced Accuracy: {clf_balanced_acc}")

        result_dict["acc"] = clf_score
        result_dict["b_acc"] = clf_balanced_acc
        result_dict["conf_mat"] = confusion_matrix(labels, label_pred,
                                                   labels=clf.classes_, normalize="all")
        disp = ConfusionMatrixDisplay(confusion_matrix=result_dict["conf_mat"],
                                      display_labels=clf.classes_)
        disp.plot()
        plt.show()
        
    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)
    
    data, labels = load_all_data('dataset')

    print(f"{data.shape=} {labels.shape=}")

    data, labels = redistribute_labels(data, labels)

    data_train = data[:6]
    data_test = np.asarray([data[-1]])
    labels_train = labels[:6]
    labels_test = np.asarray([labels[-1]])

    scaler = StandardScaler()
    scaler.fit(data_train.reshape((-1, data_train.shape[-1])))

    data_train_scaled = scaler.transform(data_train.reshape((-1, data_train.shape[-1])))
    data_test_scaled = scaler.transform(data_test.reshape((-1, data_test.shape[-1])))

    data_train = data_train_scaled.reshape(data_train.shape)
    data_test = data_test_scaled.reshape(data_test.shape)

    print(labels_test)

    print(f"{data_train.shape=} {data_test.shape=} {labels_train.shape=} {labels_test.shape=}")

    clf = create_and_train_func(data_train, labels_train)

    eval_func(clf, data_test, labels_test)

    # train_with_cv(data_train, labels_train, create_and_train_func, eval_func)

    # enc = OneHotEncoder()
    # enc.fit(np.asarray([[l] for l in labels_train.flatten()]))

    # print(enc.categories_)

    # test = np.array([[0],[1],[2],[3],[4],[5],[6]])

    # labelsmoothing(enc, test)

if __name__ == '__main__':
    main()

