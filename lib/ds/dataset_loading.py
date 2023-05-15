import os

import numpy as np

BIRD_NAMES = ['comcuc', 'cowpig1', 'eucdov', 'eueowl1', 'grswoo', 'tawowl1']

def load_annotations_for_file(bird: str, file_nr, ds_base_path: str) -> tuple[np.array, np.array]:
    data = np.load(os.path.join(ds_base_path, bird, f'{file_nr}.npy'))
    labels = np.load(os.path.join(ds_base_path, bird, f'{file_nr}.labels.npy')).astype(int)
    return data, labels


def get_file_nrs_of_bird(bird: str, ds_base_path: str):
    return [f.split('.')[0] for f in os.listdir(os.path.join(ds_base_path, bird))[::2]]


def load_annotations_for_bird(bird: str, ds_base_path: str):
    file_nrs = get_file_nrs_of_bird(bird, ds_base_path)
    bird_data = np.zeros(shape=(0, 100, 548))
    bird_labels = np.zeros(shape=(0, 100))
    for file_nr in file_nrs:
        data, labels = load_annotations_for_file(bird, file_nr, ds_base_path)
        data = data[None, :, :]
        labels = np.apply_along_axis(lambda x: x[0], arr=labels, axis=1)[None, :]
        bird_data = np.append(bird_data, data, axis=0)
        bird_labels = np.append(bird_labels, labels, axis=0)
    return bird_data, bird_labels

def load_all_data(ds_base_path: str):
    data = np.zeros(shape=(0, 100, 548))
    labels = np.zeros(shape=(0, 100))
    for bird_name in BIRD_NAMES:
        bird_data, bird_labels = load_annotations_for_bird(bird_name, ds_base_path)
        data = np.append(data, bird_data, axis=0)
        labels = np.append(labels, bird_labels, axis=0)
    return data, labels

def flatten(data: np.array, labels: np.array) -> tuple[np.array, np.array]:
    return data.reshape(-1, data.shape[-1]), labels.flatten()

