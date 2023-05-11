import numpy as np
import matplotlib.pyplot as plt
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

from lib.ds.dataset_loading import load_all_data, flatten
from lib.ds.dataset_splitting import split_by_1d, create_folds
from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data, labelsmoothing, redistribute_labels

def main():
    def create_and_train_func(data: np.ndarray, labels: np.ndarray, seed=6942066):
        clf = SVC(verbose=True, random_state=seed, kernel="rbf")
        
        #data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        print(f"{data.shape= } {labels.shape= }")
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        result_dict = {}
        #data = data.reshape((-1, data.shape[-1]))
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

        return clf_score, clf_balanced_acc
        
    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)
    

    # load dataset
    data_orig, labels_orig = load_all_data('dataset')

    data_train, labels_train, data_test, labels_test = split_by_1d(data_orig, labels_orig)

    data_train, data_test = remove_correlated_columns(data_train, data_test)

    print(f"{data_train.shape=} {labels_train.shape=} {data_test.shape=} {labels_test.shape=}")

    # redistribute labels that each class has the same rate of occurance
    data_train, labels_train = redistribute_labels(data_train, labels_train)

    skf = StratifiedKFold(n_splits=10)

    best_normalizer = None
    best_b_acc = -1

    # make 3d to 2d and 2d to 1d
    data_train, labels_train = flatten(data_train, labels_train)
    data_test, labels_test = flatten(data_test, labels_test)
    for i, (train_idx, val_idx) in enumerate(skf.split(data_train, labels_train)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_idx}")
        print(f"  Validation:  index={val_idx}")

        fold_data_train = data_train[train_idx]
        fold_labels_train = labels_train[train_idx]
        fold_data_val = data_train[val_idx]
        fold_labels_val = labels_train[val_idx]

        fold_data_train, fold_data_val, scaler = normalize_data(fold_data_train, fold_data_val)
        clf = create_and_train_func(fold_data_train, fold_labels_train)
        print("\n\nEvaluate Validation set:")
        _, b_acc = eval_func(clf, fold_data_val, fold_labels_val)

        if b_acc > best_b_acc:
            best_normalizer = scaler
            best_b_acc = b_acc


    # scaling training set and test set
    # scaler = StandardScaler()
    # scaler.fit(data_train.reshape((-1, data_train.shape[-1])))

    # data_train_scaled = scaler.transform(data_train.reshape((-1, data_train.shape[-1])))
    # data_test_scaled = scaler.transform(data_test.reshape((-1, data_test.shape[-1])))

    # data_train = data_train_scaled.reshape(data_train.shape)
    # data_test = data_test_scaled.reshape(data_test.shape)

    # data_train, data_test = normalize_data(data_train, data_test)

    #print(labels_test)

    #print(f"{data_train.shape=} {data_test.shape=} {labels_train.shape=} {labels_test.shape=}")

    # train model
    # clf = create_and_train_func(data_train, labels_train)


    data_test = best_normalizer.transform(data_test)

    print("\n\nEvaluate Test set:")

    # evaluate model
    eval_func(clf, data_test, labels_test)

    # train_with_cv(data_train, labels_train, create_and_train_func, eval_func)

    # enc = OneHotEncoder()
    # enc.fit(np.asarray([[l] for l in labels_train.flatten()]))

    # print(enc.categories_)

    # test = np.array([[0],[1],[2],[3],[4],[5],[6]])

    # labelsmoothing(enc, test)

if __name__ == '__main__':
    main()

