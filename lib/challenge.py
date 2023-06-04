import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


def predict_for_challenge(
        challenge_data: np.ndarray,
        model: nn.Module,
        normalization_scaler: StandardScaler,
        device: torch.device
):
    """
    :param challenge_data: shape (N=16, S=3000, D=548)
    :param model:
    :param normalization_scaler:
    :param device:
    """

    n_sequences, sequence_length, dimensions = challenge_data.shape

    challenge_data = challenge_data.reshape((-1, dimensions))
    challenge_data = normalization_scaler.transform(challenge_data)
    challenge_data = challenge_data.reshape((n_sequences, sequence_length, dimensions))

    with torch.no_grad():

        challenge_data_tensor = torch.Tensor(challenge_data).to(device)

        model = model.to(device)
        model.eval()

        result = model(challenge_data_tensor)

        return result


def save_results_to_csv(results: np.ndarray, path: str):
    with open(path, mode='wt') as f:
        for i in range(results.shape[0]):
            f.write(f'test{i:02d},{",".join([str(x) for x in results[i]])}\n')


def load_results_from_csv(path: str):
    return np.genfromtxt(path, delimiter=',')[:, 1:]
