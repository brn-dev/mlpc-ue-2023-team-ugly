import numpy as np
import matplotlib.pyplot as plt
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from lib.ds.dataset_loading import load_all_data, flatten
from lib.ds.dataset_splitting import split_by_1d, create_folds, split
from lib.training import train_with_cv
from lib.data_preprocessing import remove_correlated_columns, normalize_data, labelsmoothing, redistribute_labels, find_correlated_columns_to_drop
from lib.utils import get_baseline

from lib.svm_training import train_binary_classifiers, run_training_with_params
from lib.svm_testing import test_binary_classifiers, test_models
from collections import Counter

import pickle as pkl
from copy import deepcopy
import datetime
import os

BASE_PATH = os.path.join("out", "svm_training")

def train():
    # load dataset
    a = datetime.datetime.now()
    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    
    print("New load_annotations_for_bird()")
    print(f"{(b-a).total_seconds()}s")

    param_list = [{'C': 0.5, 'gamma': 'auto', 'kernel': 'rbf', 'cache_size': 2000}] # best parameters yielded by grid search
    best_dicts = train_binary_classifiers(7, data_orig, labels_orig, param_list)

def train2():
    # load dataset
    a = datetime.datetime.now()
    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    
    print("New load_annotations_for_bird()")
    print(f"{(b-a).total_seconds()}s")

    data_train, labels_train, data_test, labels_test = split(data_orig, labels_orig)

    param_list = [{'C': 0.5, 'gamma': 'auto', 'kernel': 'rbf', 'cache_size': 2000}] # best parameters yielded by grid search
    
    cols_to_drop = find_correlated_columns_to_drop(data_train)

    pkl.dump(cols_to_drop, open(os.path.join(BASE_PATH, f"new_to_drop.pkl"), "wb"))

    data_train, data_test = remove_correlated_columns(data_train, data_test, cols_to_drop)

    # redistribute labels that each class has the same rate of occurance
    data_train, labels_train = redistribute_labels(data_train, labels_train)

    skf = StratifiedKFold(n_splits=10)

    # make 3d to 2d and 2d to 1d
    data_train, labels_train = flatten(data_train, labels_train)

    param_dict = run_training_with_params(skf, data_train, labels_train, param_list)
    pkl.dump(param_dict, open(os.path.join(BASE_PATH, f"new_model.pkl"), "wb"))

def test():
    # load dataset
    a = datetime.datetime.now()
    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    
    print("New load_annotations_for_bird()")
    print(f"{(b-a).total_seconds()}s")

    test_binary_classifiers(7, data_orig, labels_orig)

def test_vote():
    a = datetime.datetime.now()
    class0 = pkl.load((open(os.path.join(BASE_PATH, f"class0_pred_proba.pkl"), "rb")))
    class1 = pkl.load((open(os.path.join(BASE_PATH, f"class1_pred_proba.pkl"), "rb")))
    class2 = pkl.load((open(os.path.join(BASE_PATH, f"class2_pred_proba.pkl"), "rb")))
    class3 = pkl.load((open(os.path.join(BASE_PATH, f"class3_pred_proba.pkl"), "rb")))
    class4 = pkl.load((open(os.path.join(BASE_PATH, f"class4_pred_proba.pkl"), "rb")))
    class5 = pkl.load((open(os.path.join(BASE_PATH, f"class5_pred_proba.pkl"), "rb")))
    class6 = pkl.load((open(os.path.join(BASE_PATH, f"class6_pred_proba.pkl"), "rb")))

    birds = [class1, class2, class3, class4, class5, class6]

    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    print(f"Data loaded: {(b-a).total_seconds()}s")

    data_train, labels_train, data_test, labels_test = split(data_orig, labels_orig)
    _, labels_test = flatten(data_test, labels_test)

    for i in range(len(class0)):
        if class0[i] == -1:
            bird_set = False
            for bird_nr, bird in enumerate(birds):
                if bird[i] != -1:
                    class0[i] = bird_nr+1
                    bird_set = True
                    break
            if not bird_set:
                class0[i] = 0
    
    print(class0)
    print(class0[class0 < 0])
    print(class0[class0 > 6])
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels_test, class0)}")

def test_vote2():
    a = datetime.datetime.now()
    class0 = pkl.load((open(os.path.join(BASE_PATH, f"class0_pred_proba.pkl"), "rb")))
    class1 = pkl.load((open(os.path.join(BASE_PATH, f"class1_pred_proba.pkl"), "rb")))
    class2 = pkl.load((open(os.path.join(BASE_PATH, f"class2_pred_proba.pkl"), "rb")))
    class3 = pkl.load((open(os.path.join(BASE_PATH, f"class3_pred_proba.pkl"), "rb")))
    class4 = pkl.load((open(os.path.join(BASE_PATH, f"class4_pred_proba.pkl"), "rb")))
    class5 = pkl.load((open(os.path.join(BASE_PATH, f"class5_pred_proba.pkl"), "rb")))
    class6 = pkl.load((open(os.path.join(BASE_PATH, f"class6_pred_proba.pkl"), "rb")))

    birds = np.array([class0, class1, class2, class3, class4, class5, class6])
    print(f"{birds.shape=}")

    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    print(f"Data loaded: {(b-a).total_seconds()}s")

    data_train, labels_train, data_test, labels_test = split(data_orig, labels_orig)
    _, labels_test = flatten(data_test, labels_test)

    result = np.argmax(birds[:, :, 1], axis=0)
    print(f"{result.shape=}")

    N = 1

    weight = np.linspace(0.7, 0.9, N)
    weight = np.concatenate((weight, np.array([1]), np.flip(weight)))

    x = result.copy()
    for i in range(N,len(x)-N):
        window = x[i-N:i+N+1]
        eval_array_weight = {}
        for el_idx, el in enumerate(window):
            if el in list(eval_array_weight.keys()):
                eval_array_weight[el] += weight[el_idx]
            else:
                eval_array_weight[el] = weight[el_idx]
        max_occurance = sorted(eval_array_weight.items(), key=lambda x:x[1])[-1][0]
        x[i] = max_occurance



    np.savetxt("orig.csv", labels_test, delimiter=',')
    np.savetxt("pred.csv", result, delimiter=',')
    np.savetxt("pred_sm.csv", x, delimiter=',')
    
    print(result)
    print(result[result < 0])
    print(result[result > 6])
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels_test, result)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels_test, x)}")

def test2():
    a = datetime.datetime.now()
    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    print(f"Data loaded: {(b-a).total_seconds()}s")

    test_models(data_orig, labels_orig)

def test_vote3():
    a = datetime.datetime.now()
    class0 = pkl.load((open(os.path.join(BASE_PATH, f"class0_pred_proba.pkl"), "rb")))
    class1 = pkl.load((open(os.path.join(BASE_PATH, f"new_pred_proba.pkl"), "rb")))

    print(f"{class0.shape=}")
    print(f"{class1.shape=}")

    data_orig, labels_orig = load_all_data('dataset')
    b = datetime.datetime.now()
    print(f"Data loaded: {(b-a).total_seconds()}s")

    data_train, labels_train, data_test, labels_test = split(data_orig, labels_orig)
    _, labels_test = flatten(data_test, labels_test)

    class0 = np.argmax(class0, axis=1)
    class1 = np.argmax(class1, axis=1)

    print(f"{class0.shape=}")
    print(f"{class1.shape=}")

    not_bird = np.argwhere(class0 == 0)
    class0[class0 == 1] = 0
    class0[not_bird] = class1[not_bird]
    result = class0

    N = 1

    weight = np.linspace(0.7, 0.9, N)
    weight = np.concatenate((weight, np.array([1]), np.flip(weight)))

    x = result.copy()
    for i in range(N,len(x)-N):
        window = x[i-N:i+N+1]
        eval_array_weight = {}
        for el_idx, el in enumerate(window):
            if el in list(eval_array_weight.keys()):
                eval_array_weight[el] += weight[el_idx]
            else:
                eval_array_weight[el] = weight[el_idx]
        max_occurance = sorted(eval_array_weight.items(), key=lambda x:x[1])[-1][0]
        x[i] = max_occurance

    print(f"Balanced Accuracy: {balanced_accuracy_score(labels_test, result)}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels_test, x)}")

    np.savetxt("new_pred.csv", result, delimiter=',')
    np.savetxt("new_pred_sm.csv", x, delimiter=',')

if __name__ == '__main__':
    train2()
    test2()
    test_vote3()
