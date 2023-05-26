import numpy as np
import os
import dill as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from lib.training import train_with_cv, train_best

dataset_path = os.path.join('np_data_new')


def main():

    data_train_down = np.load(os.path.join(dataset_path, 'balanced_data_train_down.npy'))
    labels_train_down = np.load(os.path.join(dataset_path, 'balanced_labels_train.npy'))
    data_test = np.load(os.path.join(dataset_path, 'flatten_data_test.npy'))
    labels_test = np.load(os.path.join(dataset_path, 'flatten_labels_test.npy'))

    def create_and_train_func(data: np.ndarray, labels: np.ndarray, k=5):
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=20)
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        clf.fit(data, labels)
        return clf

    def eval_func(clf, data: np.ndarray, labels: np.ndarray):
        data = data.reshape((-1, data.shape[-1]))
        labels = labels.flatten()
        acc = clf.score(data, labels)
        b_acc = balanced_accuracy_score(labels, clf.predict(data))

        confusion_m = confusion_matrix(labels, clf.predict(data), normalize='true')
        _, ax = plt.subplots(1, figsize=(8, 8))
        confusion_display = ConfusionMatrixDisplay(confusion_m,
                                                   display_labels=['other', 'comcuc', 'cowpig1', 'eucdov',
                                                                   'eueowl1', 'grswoo', 'tawowl1'])
        confusion_display.plot(ax=ax, cmap='magma')
        ax.set_title('Confusion matrix computed on the test set')
        plt.show()

        return acc, b_acc

    def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)

    # train_with_cv(data_train_down, labels_train_down, create_and_train_func, eval_func)

    all_models = pkl.load(open(os.path.join('knn', 'scores_90corr', 'all_model_scores.pkl'), 'rb'))
    all_train_accuracies = [all_models[knn]['train']['acc'] for knn in all_models]
    all_train_b_accuracies = [all_models[knn]['train']['b_acc'] for knn in all_models]
    all_validation_accuracies = [all_models[knn]['valid']['acc'] for knn in all_models]
    all_validation_b_accuracies = [all_models[knn]['valid']['b_acc'] for knn in all_models]

    plt.plot(range(1, len(all_train_b_accuracies) + 1), all_train_b_accuracies, label='Balanced training accuracy')
    plt.plot(range(1, len(all_validation_b_accuracies) + 1), all_validation_b_accuracies, label='Balanced validation '
                                                                                                'accuracy')
    plt.xticks(np.arange(1, 21))
    plt.title('Accuracies per K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    best_k = np.argmax(all_validation_b_accuracies) + 1
    print(best_k)

    train_best(
        data_train_down,
        labels_train_down,
        data_test,
        labels_test,
        create_and_train_func,
        eval_func,
        best_k)


if __name__ == '__main__':
    main()
