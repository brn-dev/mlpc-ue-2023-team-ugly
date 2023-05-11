import numpy as np
import os
import dill as pkl
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from lib.training import train_with_cv, train_best
from lib.data_preprocessing import normalize_data
from lib.ds.dataset_loading import flatten
from torch.utils.data import DataLoader, TensorDataset
from lib.fnn import FNN
import torch


def main():
    data_train_folds_down = np.load(os.path.join('np_data_90corr_select150best', 'data_train_folds_down.npy'))
    labels_data_train_folds_down = np.load(os.path.join('np_data_90corr_select150best', 'labels_train_folds_down.npy'))

    # data_train_down = np.load(os.path.join('np_data_select150best', 'data_train_down.npy'))
    # labels_train = np.load(os.path.join('np_data_select150best', 'labels_train.npy'))
    # data_test_down = np.load(os.path.join('np_data_select150best', 'data_test_down.npy'))
    # labels_test = np.load(os.path.join('np_data_select150best', 'labels_test.npy'))

    # with open(os.path.join('scores', 'knn_accuracies.pkl'), 'rb') as f:
    # accuracies = pkl.load(f)
    # with open(os.path.join('scores_select150best', 'knn_valid_balanced_accuracies.pkl'), 'rb') as f:
    #     valid_balanced_acc = pkl.load(f)

    def train_func(model: torch.nn.Module, optim: torch.optim.Optimizer, data_train: np.ndarray,
                   labels_train: np.ndarray, data_test: np.ndarray,
                   labels_test: np.ndarray, nr_epochs: int, device: torch.device = 'cpu', confusion_bool: bool = False):

        target_device = device
        clf = model

        data_train, labels_train = flatten(data_train, labels_train)
        data_train = torch.from_numpy(data_train)
        labels_train = torch.from_numpy(labels_train)
        data_train = TensorDataset(data_train, labels_train)

        data_test, labels_test = flatten(data_test, labels_test)
        data_test = torch.from_numpy(data_test)
        labels_test = torch.from_numpy(labels_test)
        data_test = TensorDataset(data_test, labels_test)

        loader_train = DataLoader(data_train, batch_size=32, shuffle=True)
        loader_test = DataLoader(data_test, batch_size=32, shuffle=False)
        optim = optim
        print(clf, end='\n\n')

        performances = dict()
        accuracies = np.array([])
        losses = np.array([])
        b_accuracies = np.array([])
        acc_loss = dict()
        for epoch in range(nr_epochs):
            train_network(clf, loader_train, optim, target_device)
            performance = test_network(clf, loader_train, target_device)

            print(f'Epoch: {str(epoch + 1).zfill(len(str(nr_epochs)))} ' +
                  f'/ Loss: {performance[0]:.4f} / Accuracy: {performance[1]:.4f} / Balanced accuracy: {performance[2]:.4f}')
            accuracies = np.append(accuracies, performance[1])
            losses = np.append(losses, performance[0])
            b_accuracies = np.append(b_accuracies, performance[2])

        acc_loss["acc"] = accuracies.mean()
        acc_loss['loss'] = losses.mean()
        acc_loss['b_acc'] = b_accuracies.mean()
        print(
            f'\nFinal train loss: {acc_loss["loss"]:.4f} / Final train accuracy: {acc_loss["acc"]:.4f} / Final train balanced accuracy: {acc_loss["b_acc"]:.4f}')
        performances['train'] = acc_loss

        performance = test_network(clf, loader_test, target_device, confusion_bool=confusion_bool)
        acc_loss['loss'] = performance[0]
        acc_loss['acc'] = performance[1]
        acc_loss['b_acc'] = performance[2]
        print(
            f'\nFinal validation loss: {performance[0]:.4f} / Final validation accuracy: {performance[1]:.4f} / Final validation balanced accuracy: {performance[2]:.4f}')
        performances['valid'] = acc_loss

        return clf, optim, performances

    def train_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                      optimizer: torch.optim.Optimizer, device: torch.device = 'cpu') -> None:
        """
        Train specified network for one epoch on specified data loader.

        :param model: network to train
        :param data_loader: data loader to be trained on
        :param optimizer: optimizer used to train network
        :param device: device on which to train network
        """
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for data, target in data_loader:
            data, target = data.float().to(device), target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    def test_network(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                     device: torch.device = 'cpu', confusion_bool: bool = False) -> Tuple[float, float, float]:
        """
        Test specified network on specified data loader.

        :param model: network to test on
        :param data_loader: data loader to be tested on
        :param device: device on which to test network
        :return: cross-entropy loss as well as accuracy
        """
        model.eval()
        loss, num_correct, num_samples = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        targets = np.array([])
        predictions = np.array([])
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.float().to(device), target.long().to(device)
                output = model(data)
                loss += float(criterion(output, target).item())
                pred = output.argmax(dim=1).view(-1).long()
                targets = np.append(targets, target.cpu().detach().numpy())
                predictions = np.append(predictions, pred.cpu().detach().numpy())
                num_correct += int((pred == target.view(-1)).sum().item())
                num_samples += pred.shape[0]

        targets = targets.flatten()
        predictions = predictions.flatten()

        if confusion_bool:
            print('Listen')
            confusion_m = confusion_matrix(targets, predictions, normalize='true')
            confusion_display = ConfusionMatrixDisplay(confusion_m)
            confusion_display.plot()
            plt.show()

        balanced_acc = balanced_accuracy_score(targets, predictions)
        return loss / num_samples, num_correct / num_samples, balanced_acc

    # def get_baseline(labels: np.ndarray) -> float:
    #     labels = labels.flatten()
    #     unique_labels, labels_count = np.unique(labels, return_counts=True)
    #     return max(labels_count) / len(labels)

    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_and_data = train_with_cv(data_train_folds_down,
    #                                labels_data_train_folds_down,
    #                                train_func
    #                                )

    all_data_train = np.load(os.path.join('np_data_90corr_select150best', 'data_train_down.npy'))
    all_data_test = np.load(os.path.join('np_data_90corr_select150best', 'data_test_down.npy'))
    all_label_train = np.load(os.path.join('np_data_90corr_select150best', 'labels_train.npy'))
    all_label_test = np.load(os.path.join('np_data_90corr_select150best', 'labels_test.npy'))
    all_data_train, all_data_test = normalize_data(all_data_train, all_data_test)

    torch.manual_seed(69)
    model = FNN(all_data_train.shape[-1], 7).to(device=target_device)
    optim = torch.optim.Adamax(model.parameters(), lr=1e-3)

    train_func(model, optim, all_data_train, all_label_train, all_data_test, all_label_test, 30, target_device, confusion_bool=True)


if __name__ == '__main__':
    main()
