import numpy as np
import os
from typing import Tuple

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from lib.training import train_with_cv
from lib.data_preprocessing import normalize_data
from lib.ds.dataset_loading import flatten
from torch.utils.data import DataLoader, TensorDataset
import torch


dataset_path = 'np_data_new'

def main():
    # data_train_folds_down = np.load(os.path.join('np_data_90corr_select150best', 'data_train_folds_down.npy'))
    # labels_data_train_folds_down = np.load(os.path.join('np_data_90corr_select150best', 'labels_train_folds_down.npy'))

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

        data_train = torch.from_numpy(data_train)
        labels_train = torch.from_numpy(labels_train)
        data_train = TensorDataset(data_train, labels_train)

        data_test = torch.from_numpy(data_test)
        labels_test = torch.from_numpy(labels_test)
        data_test = TensorDataset(data_test, labels_test)

        loader_train = DataLoader(data_train, batch_size=32, shuffle=True)
        loader_test = DataLoader(data_test, batch_size=32, shuffle=False)
        optim = optim
        print(clf, end='\n\n')

        performances = dict()
        train_accuracies = np.array([])
        train_losses = np.array([])
        train_b_accuracies = np.array([])

        valid_accuracies = np.array([])
        valid_losses = np.array([])
        valid_b_accuracies = np.array([])

        train_acc_loss = dict()
        valid_acc_loss = dict()
        for epoch in range(nr_epochs):
            train_network(clf, loader_train, optim, target_device)
            performance_train = test_network(clf, loader_train, target_device)
            performance_valid = test_network(clf, loader_test, target_device)

            print(f'Epoch: {str(epoch + 1).zfill(len(str(nr_epochs)))} ' +
                  f'/ Train | Validation Loss: {performance_train[0]:.4f} | {performance_valid[0]:.4f} / Train | '
                  f'Validation Accuracy: {performance_train[1]:.4f} | {performance_valid[1]:.4f} / Train | Validation '
                  f'Balanced accuracy: {performance_train[2]:.4f} | {performance_valid[2]:.4f}')
            train_accuracies = np.append(train_accuracies, performance_train[1])
            train_losses = np.append(train_losses, performance_train[0])
            train_b_accuracies = np.append(train_b_accuracies, performance_train[2])

            valid_accuracies = np.append(valid_accuracies, performance_valid[1])
            valid_losses = np.append(valid_losses, performance_valid[0])
            valid_b_accuracies = np.append(valid_b_accuracies, performance_valid[2])

        train_acc_loss["acc"] = train_accuracies[-1]
        train_acc_loss['loss'] = train_losses[-1]
        train_acc_loss['b_acc'] = train_b_accuracies[-1]

        valid_acc_loss["acc"] = valid_accuracies[-1]
        valid_acc_loss['loss'] = valid_losses[-1]
        valid_acc_loss['b_acc'] = valid_b_accuracies[-1]
        print(
            f'\nFinal train | valid loss: {train_acc_loss["loss"]:.4f} | {valid_acc_loss["loss"]:.4f} / Final train | '
            f'valid accuracy: {train_acc_loss["acc"]:.4f} | {valid_acc_loss["acc"]:.4f} / Final train | valid '
            f'balanced accuracy: {train_acc_loss["b_acc"]:.4f} | {valid_acc_loss["b_acc"]:.4f}\n')

        performances['train'] = train_acc_loss
        performances['valid'] = valid_acc_loss

        test_network(clf, loader_test, target_device, confusion_bool=confusion_bool)

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

    all_data_train = np.load(os.path.join(dataset_path, 'balanced_data_train_down.npy'))
    all_data_test = np.load(os.path.join(dataset_path, 'flatten_data_test.npy'))
    all_labels_train = np.load(os.path.join(dataset_path, 'balanced_labels_train.npy'))
    all_labels_test = np.load(os.path.join(dataset_path, 'flatten_labels_test.npy'))
    all_data_train, all_data_test = normalize_data(all_data_train, all_data_test)

    model_and_data = train_with_cv(all_data_train,
                                   all_data_test,
                                   all_labels_train,
                                   all_labels_test,
                                   train_func
                                   )


if __name__ == '__main__':
    main()
