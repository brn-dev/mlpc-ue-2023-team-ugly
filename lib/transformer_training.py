import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from lib.confusion_matrix import display_confmat
from lib.ds.torch_dataset import create_offset_data_loader
from lib.model.classification_transformer import TransformerHyperParameters, ClassificationTransformer
from lib.torch_device import get_torch_device
from lib.training import train_with_cv


def train_transformer_with_cv(
        data: np.ndarray,
        labels: np.ndarray,
        hyper_parameters: TransformerHyperParameters,
        device: torch.device = 'cpu'
):
    def create_and_train_func(d: np.ndarray, l: np.ndarray):
        transformer = ClassificationTransformer(hyper_parameters)
        optimizer = optim.Adam(transformer.parameters(), lr=hyper_parameters.lr, betas=(0.9, 0.98), eps=1e-9)
        transformer.to(device)

        data_loader = create_offset_data_loader(d, l)

        loss_weight = calculate_loss_weight(torch.Tensor(l))

        train_network(transformer, data_loader, optimizer, loss_weight, device)

        return transformer

    def eval_func(network: nn.Module, d: np.ndarray, l: np.ndarray):
        data_loader = create_offset_data_loader(d, l)
        test_network(network, data_loader, device)


    train_with_cv(data, labels, create_and_train_func, eval_func)


def train_transformer(
        data: np.ndarray,
        labels: np.ndarray,
        hyper_parameters: TransformerHyperParameters,
        device: torch.device = 'cpu'
) -> ClassificationTransformer:
    transformer = ClassificationTransformer(hyper_parameters)
    optimizer = optim.Adam(transformer.parameters(), lr=hyper_parameters.lr, betas=(0.9, 0.98), eps=1e-9)
    transformer.to(device)

    data_loader = create_offset_data_loader(data, labels)

    loss_weight = calculate_loss_weight(torch.Tensor(labels))

    train_network(transformer, data_loader, optimizer, loss_weight, device)

    return transformer



def train_network(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_weight: torch.Tensor,
        device: torch.device = 'cpu'
):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    for epoch in range(8):
        epoch_loss, num_correct, num_samples = 0.0, 0, 0
        for data, target_input, target_expected in data_loader:
            data, target_input, target_expected = reshape_for_transformer(data, target_input, target_expected)
            data, target_input, target_expected = \
                data.float().to(device), target_input.long().to(device), target_expected.long().to(device)

            optimizer.zero_grad()

            pred = model(data, target_input)

            pred, target_expected = reshape_for_loss(pred, target_expected)
            loss = criterion(pred, target_expected)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

            pred_labels = pred.argmax(dim=1).view(-1).long()
            num_correct += int((pred_labels == target_expected.view(-1)).sum().item())
            num_samples += pred_labels.shape[0]
        acc = num_correct / num_samples
        print(f'Epoch {epoch:>3}: {epoch_loss = :.6f}, {num_correct = :>5}, {num_samples = :>5}, {acc = :.6f}')


def test_network(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = 'cpu'
):
    loss, num_correct, num_samples = 0.0, 0, 0

    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_pred_labels = torch.Tensor().long()
    all_target_labels = torch.Tensor().long()

    with torch.no_grad():
        for data, target_input, target_expected in data_loader:
            data, target_input, target_expected = reshape_for_transformer(data, target_input, target_expected)
            data, target_input, target_expected = \
                data.float().to(device), target_input.long().to(device), target_expected.long().to(device)

            pred = model(data, target_input)

            pred, target_expected = reshape_for_loss(pred, target_expected)
            loss += float(criterion(pred, target_expected).item())

            pred_labels = pred.argmax(dim=1).view(-1).long()
            print((pred_labels == target_expected).all())
            num_correct += int((pred_labels == target_expected.view(-1)).sum().item())
            num_samples += pred_labels.shape[0]

            all_pred_labels = torch.cat((all_pred_labels, pred_labels.cpu()))
            all_target_labels = torch.cat((all_target_labels, target_expected.cpu()))

    loss, acc = loss / num_samples, num_correct / num_samples
    print(f'Evaluated with {loss = }, {acc = }')

    display_confmat(all_pred_labels, all_target_labels)


def reshape_for_transformer(
        data: torch.Tensor,
        target_input: torch.Tensor,
        target_expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data, target_input, target_expected = (torch.transpose(x, 0, 1) for x in [data, target_input, target_expected])
    return data, target_input, target_expected


def reshape_for_loss(
        pred: torch.Tensor,
        target_expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = torch.flatten(pred[1:], end_dim=1)
    target_expected = torch.flatten(target_expected)
    return pred, target_expected


def calculate_loss_weight(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(torch.flatten(labels).long())
    counts_max = counts.max()
    return (counts_max / counts).to(get_torch_device())


