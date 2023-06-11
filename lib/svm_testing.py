import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from lib.data_preprocessing import remove_correlated_columns
from lib.svm_training import eval_func
from lib.ds.dataset_loading import flatten
from lib.ds.dataset_splitting import split
import pickle as pkl
import os
from lib.utils import nostdout
import sys

BASE_PATH = os.path.join("out", "svm_training")

def test_binary_classifiers(num_classes: int, data_orig: np.ndarray, labels: np.ndarray):
    proba_predictions = []
    for bird in tqdm(range(num_classes), "Evaluate Testset:", position=0, file=sys.stdout):
        to_drop = pkl.load(open(os.path.join(BASE_PATH, f"class{bird}_to_drop.pkl"), "rb"))
        class_dict = pkl.load(open(os.path.join(BASE_PATH, f"class{bird}_dict.pkl"), "rb"))

        data_train, labels_train, data_test, labels_test = split(data_orig, labels)

        _, data_test = remove_correlated_columns(data_train, data_test, to_drop)

        clf = class_dict["validation"]["clf"]
        scaler = class_dict["validation"]["sclaer"]

        

        data_test, labels_test = flatten(data_test, labels_test)

        data_test = scaler.transform(data_test)

        label_pred = clf.predict(data_test)

        modified_labels = np.where(labels_test == bird, labels_test, -1)

        clf_score = clf.score(data_test, modified_labels)
        clf_balanced_acc = balanced_accuracy_score(modified_labels, label_pred)
        with nostdout():
            print("\nEvaluate Test set(normal):")
            print(f"\nMean Accuracy: {clf_score}")
            print(f"Balanced Accuracy: {clf_balanced_acc}")

        pkl.dump(label_pred, (open(os.path.join(BASE_PATH, f"class{bird}_pred.pkl"), "wb")))

        label_pred_proba = clf.predict_proba(data_test)

        pkl.dump(label_pred_proba, (open(os.path.join(BASE_PATH, f"class{bird}_pred_proba.pkl"), "wb")))

        proba_predictions.append(label_pred_proba)
        label_pred_proba = np.argmax(label_pred_proba, axis=1)
        label_pred_proba = np.where(label_pred_proba == 0, -1, bird)

        clf_score = clf.score(data_test, modified_labels)
        clf_balanced_acc = balanced_accuracy_score(modified_labels, label_pred_proba)

        with nostdout():
            print("\nEvaluate Test set(probability):")
            print(f"\nMean Accuracy: {clf_score}")
            print(f"Balanced Accuracy: {clf_balanced_acc}")

        errors = []
        for i in range(len(label_pred)):
            if label_pred[i] != label_pred_proba[i]:
                errors.append(i)
        
        with nostdout():
            print(f"{len(errors)} errors found")
            print(f"Errors at: {errors}")

def test_models(data_orig, labels):
    to_drop = pkl.load(open(os.path.join(BASE_PATH, f"new_to_drop.pkl"), "rb"))
    class_dict = pkl.load(open(os.path.join(BASE_PATH, f"new_model.pkl"), "rb"))

    data_train, labels_train, data_test, labels_test = split(data_orig, labels)

    _, data_test = remove_correlated_columns(data_train, data_test, to_drop)

    clf = class_dict["validation"]["clf"]
    scaler = class_dict["validation"]["sclaer"]

    

    data_test, labels_test = flatten(data_test, labels_test)

    data_test = scaler.transform(data_test)

    label_pred = clf.predict(data_test)

    clf_score = clf.score(data_test, labels_test)
    clf_balanced_acc = balanced_accuracy_score(labels_test, label_pred)
    with nostdout():
        print("\nEvaluate Test set(normal):")
        print(f"\nMean Accuracy: {clf_score}")
        print(f"Balanced Accuracy: {clf_balanced_acc}")

    pkl.dump(label_pred, (open(os.path.join(BASE_PATH, f"new_pred.pkl"), "wb")))

    label_pred_proba = clf.predict_proba(data_test)

    pkl.dump(label_pred_proba, (open(os.path.join(BASE_PATH, f"new_pred_proba.pkl"), "wb")))

    label_pred_proba = np.argmax(label_pred_proba, axis=1)

    clf_score = clf.score(data_test, labels_test)
    clf_balanced_acc = balanced_accuracy_score(labels_test, label_pred_proba)

    with nostdout():
        print("\nEvaluate Test set(probability):")
        print(f"\nMean Accuracy: {clf_score}")
        print(f"Balanced Accuracy: {clf_balanced_acc}")

    errors = []
    for i in range(len(label_pred)):
        if label_pred[i] != label_pred_proba[i]:
            errors.append(i)
    
    with nostdout():
        print(f"{len(errors)} errors found")
        print(f"Errors at: {errors}")