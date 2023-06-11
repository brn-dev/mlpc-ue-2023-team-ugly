from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from copy import deepcopy
import numpy as np
import os
import pickle as pkl
import sys
from tqdm import tqdm
from typing import List

from lib.utils import nostdout
from lib.data_preprocessing import normalize_data, remove_correlated_columns, find_correlated_columns_to_drop, redistribute_labels
from lib.ds.dataset_loading import flatten
from lib.ds.dataset_splitting import split

BASE_PATH = os.path.join("out", "svm_training")

def create_and_train_func(data: np.ndarray, labels: np.ndarray, params: list, probability: bool=False, seed: int=6942066) -> SVC:
        clf = SVC(**params, random_state=seed, probability=probability)

        with nostdout():
            print(f"Training started...")
        
        clf.fit(data, labels)

        with nostdout():
            print(f"Training finished...")
        return clf

def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        result_dict = {}
        result_dict["validation"] = {}
        
        label_pred = clf.predict(data)

        clf_score = clf.score(data, labels)
        clf_balanced_acc = balanced_accuracy_score(labels, label_pred)

        with nostdout():
            print(f"\nMean Accuracy: {clf_score}")
            print(f"Balanced Accuracy: {clf_balanced_acc}")

        result_dict["validation"]["acc"] = clf_score
        result_dict["validation"]["b_acc"] = clf_balanced_acc
        result_dict["validation"]["conf_mat"] = confusion_matrix(labels, label_pred,
                                                        labels=clf.classes_)

        return result_dict

def run_kfold(skf: StratifiedKFold, data_train: np.ndarray, labels_train: np.ndarray, params: dict) -> dict:
    best_b_acc = 0
    best_fold_dict = None
    for i, (train_idx, val_idx) in tqdm(enumerate(skf.split(data_train, labels_train)), "Fold: ", total=10, position=1, file=sys.stdout):

        fold_data_train = data_train[train_idx]
        fold_labels_train = labels_train[train_idx]
        fold_data_val = data_train[val_idx]
        fold_labels_val = labels_train[val_idx]

        fold_data_train, fold_data_val, scaler = normalize_data(fold_data_train, fold_data_val)
        clf = create_and_train_func(fold_data_train, fold_labels_train, params, True)

        # with nostdout():
        #     print("\nEvaluate Training set:")
        # train_dict = eval_func(clf, fold_data_train, fold_labels_train)

        with nostdout():
            print("\nEvaluate Validation set:")
        eval_dict = eval_func(clf, fold_data_val, fold_labels_val)

        if eval_dict["validation"]["b_acc"] > best_b_acc:
            best_b_acc = eval_dict["validation"]["b_acc"]
            eval_dict["validation"]["params"] = params
            eval_dict["validation"]["clf"] = deepcopy(clf)
            eval_dict["validation"]["sclaer"] = scaler

            # pkl.dump(eval_dict, open(f"best_params.pkl", "wb"))
            best_fold_dict = eval_dict
        elif eval_dict["validation"]["b_acc"] < best_b_acc * 0.9:
            break
    return best_fold_dict
            
    #pkl.dump(fold_dict, open(f"{PATH}/params_c{params['C']}.pkl", "wb"))

def run_training_with_params(skf: StratifiedKFold, data_train: np.ndarray, labels_train: np.ndarray, param_list: List[dict]) -> dict:
    best_b_acc = 0
    best_params = None
    for params in tqdm(param_list, position=0, file=sys.stdout):
        with nostdout():
            print(f"\nTraining with parameters: {params}")
        best_fold = run_kfold(skf, data_train, labels_train, params)

        if best_fold["validation"]["b_acc"] > best_b_acc:
            best_params = best_fold
            best_b_acc = best_fold["validation"]["b_acc"]
    return best_params

def train_binary_classifiers(num_classes: int, data: np.ndarray, labels: np.ndarray, param_list: List[dict], kfold_splits: int=10) -> List[dict]:
    best_classifiers = []
    for bird in range(num_classes):
        # -1 because this should not be in the dataset
        modified_labels = np.where(labels == bird, labels, -1)

        data_train, labels_train, data_test, labels_test = split(data, modified_labels)

        cols_to_drop = find_correlated_columns_to_drop(data_train)

        pkl.dump(cols_to_drop, open(os.path.join(BASE_PATH, f"class{bird}_to_drop.pkl"), "wb"))

        data_train, data_test = remove_correlated_columns(data_train, data_test, cols_to_drop)

        # redistribute labels that each class has the same rate of occurance
        data_train, labels_train = redistribute_labels(data_train, labels_train)

        skf = StratifiedKFold(n_splits=kfold_splits)

        # make 3d to 2d and 2d to 1d
        data_train, labels_train = flatten(data_train, labels_train)
        data_test, labels_test = flatten(data_test, labels_test)

        param_dict = run_training_with_params(skf, data_train, labels_train, param_list)
        best_classifiers.append(param_dict)

        pkl.dump(param_dict, open(os.path.join(BASE_PATH, f"class{bird}_dict.pkl"), "wb"))

    return best_classifiers