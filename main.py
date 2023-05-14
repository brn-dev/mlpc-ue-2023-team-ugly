import numpy as np
import matplotlib.pyplot as plt
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from lib.ds.dataset_loading import load_all_data, flatten
from lib.ds.dataset_splitting import split_by_1d, create_folds
from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data, labelsmoothing, redistribute_labels
from tqdm import tqdm
import contextlib
import sys
import pickle as pkl
from copy import deepcopy


class DummyFile(object):
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

def main():
    def create_and_train_func(data: np.ndarray, labels: np.ndarray, params: list, seed=6942066):
        #clf = SVR(verbose=False,  kernel="rbf", C=100, gamma=0.0005)
        clf = SVC(**params, random_state=seed)

        with nostdout():
            print(f"Training started...")
        
        clf.fit(data, labels)

        with nostdout():
            print(f"Training finished...")
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        result_dict = {}
        
        label_pred = clf.predict(data)

        clf_score = clf.score(data, labels)
        clf_balanced_acc = balanced_accuracy_score(labels, label_pred)

        with nostdout():
            print(f"\nMean Accuracy: {clf_score}")
            print(f"Balanced Accuracy: {clf_balanced_acc}")

        result_dict["acc"] = clf_score
        result_dict["b_acc"] = clf_balanced_acc
        result_dict["conf_mat"] = confusion_matrix(labels, label_pred,
                                                   labels=clf.classes_, normalize="true")

        return result_dict
        
    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)
    

    # load dataset
    data_orig, labels_orig = load_all_data('dataset')

    data_train, labels_train, data_test, labels_test = split_by_1d(data_orig, labels_orig)

    data_train, data_test = remove_correlated_columns(data_train, data_test)

    # redistribute labels that each class has the same rate of occurance
    data_train, labels_train = redistribute_labels(data_train, labels_train)

    skf = StratifiedKFold(n_splits=10)

    best_normalizer = None
    best_b_acc = -1
    best_params = None

    # make 3d to 2d and 2d to 1d
    data_train, labels_train = flatten(data_train, labels_train)
    data_test, labels_test = flatten(data_test, labels_test)

    param_grid = {'kernel': ['rbf', 'poly'], 'gamma': np.linspace(1e-4, 0.5, 25).tolist() + ['auto', 'scale'],
                  'C': np.linspace(1, 100, 10)}
    
    param_list = list(ParameterGrid(param_grid))

    # param_list = [{'C': 12.0, 'gamma': 'auto', 'kernel': 'rbf'}] # best parameters yielded by grid search

    for params in tqdm(param_list, position=0, file=sys.stdout):
        with nostdout():
            print(f"\nTraining with parameters: {params}")
        for i, (train_idx, val_idx) in tqdm(enumerate(skf.split(data_train, labels_train)), "Fold: ", total=10, position=1, file=sys.stdout):

            fold_data_train = data_train[train_idx]
            fold_labels_train = labels_train[train_idx]
            fold_data_val = data_train[val_idx]
            fold_labels_val = labels_train[val_idx]

            fold_data_train, fold_data_val, scaler = normalize_data(fold_data_train, fold_data_val)
            clf = create_and_train_func(fold_data_train, fold_labels_train, params)

            with nostdout():
                print("\nEvaluate Validation set:")
            eval_dict = eval_func(clf, fold_data_val, fold_labels_val)

            if eval_dict["b_acc"] > best_b_acc:
                best_normalizer = scaler
                best_clf = deepcopy(clf)
                best_b_acc = eval_dict["b_acc"]
                best_params = params
                eval_dict["params"] = params
                pkl.dump(eval_dict, open(f"best_params.pkl", "wb"))
            elif eval_dict["b_acc"] < best_b_acc * 0.9:
                break


    data_test = best_normalizer.transform(data_test)

    with nostdout():
        print("\n\nEvaluate Test set:")
        print(f"Best parameters: {best_params}")

    # evaluate model
    eval_dict = eval_func(best_clf, data_test, labels_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=eval_dict["conf_mat"],
                                    display_labels=clf.classes_)
    disp.plot()
    plt.show()

if __name__ == '__main__':
    main()

