import torch
import os.path

PT_EXT = '.pt'

MODEL_FOLDER_PATH = os.path.join('saved_models')

def save_model(model: torch.nn.Module, file_name='model.pt') -> str:
    save_path = os.path.join(MODEL_FOLDER_PATH, prepare_file_name(file_name))
    with open(save_path, 'wb') as f:
        torch.save(model, f)
        return save_path


def load_model(file_name='model.pt'):
    with open(os.path.join(MODEL_FOLDER_PATH, prepare_file_name(file_name)), 'rb') as f:
        return torch.load(f)


def prepare_file_name(file_name: str) -> str:
    if not file_name.endswith(PT_EXT):
        return file_name + PT_EXT
    return file_name
