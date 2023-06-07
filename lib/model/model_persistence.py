import glob
import re
from os import PathLike

import torch
import os.path
import joblib

from sklearn.preprocessing import StandardScaler
from torch import nn

PT_EXT = '.pt'
SCALER_EXT = '.joblib'

MODEL_FOLDER_PATH = os.path.join('saved_models')


def save_model_with_scaler(model: nn.Module, normalization_scaler: StandardScaler, file_name: str):
    save_model(model, file_name)
    save_scaler(normalization_scaler, file_name)
    print(f'Saved model with scaler as "{file_name}"')


def load_model_with_scaler(file_name: str) -> tuple[nn.Module, StandardScaler]:
    model = load_model(file_name)
    normalization_scaler = load_scaler(file_name)
    return model, normalization_scaler


def load_models_with_scalers_with_prefix(
        folder_path: str,
        prefix: str
) -> list[tuple[nn.Module, StandardScaler, float]]:
    """
    :return: list of (Model, Normalization-Scaler, BACC) tuples
    """
    models: list[tuple[nn.Module, StandardScaler, float]] = []

    model_paths = set([
        os.path.splitext(filename)[0]
        for filename in os.listdir(folder_path)
        if filename.startswith(prefix)
    ])

    for model_path in model_paths:
        model, scaler = load_model_with_scaler(model_path)
        bacc = float(re.findall(r'score=(\d+\.\d+)', model_path)[0])
        models.append((model, scaler, bacc))

    return models

def save_model(model: torch.nn.Module, file_name: str) -> str:
    save_path = os.path.join(MODEL_FOLDER_PATH, append_ext(file_name, PT_EXT))
    with open(save_path, 'wb') as f:
        torch.save(model, f)
        return save_path


def load_model(file_name='model.pt'):
    with open(os.path.join(MODEL_FOLDER_PATH, append_ext(file_name, PT_EXT)), 'rb') as f:
        return torch.load(f)


def save_scaler(scaler: StandardScaler, file_name: str):
    save_path = os.path.join(MODEL_FOLDER_PATH, append_ext(file_name, SCALER_EXT))
    joblib.dump(scaler, save_path)


def load_scaler(file_name: str) -> StandardScaler:
    load_path = os.path.join(MODEL_FOLDER_PATH, append_ext(file_name, SCALER_EXT))
    return joblib.load(load_path)


def append_ext(file_name: str, ext: str) -> str:
    if file_name.endswith(ext):
        return file_name
    return file_name + ext
