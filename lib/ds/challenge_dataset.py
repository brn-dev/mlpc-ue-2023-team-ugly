import os.path

import numpy as np


def load_challenge_data(dir_path='challenge_dataset'):
    sequences: list[np.ndarray] = []
    for i in range(16):
        sequences.append(np.load(os.path.join(dir_path, f'test{i:02d}.npy')))
    return np.array(sequences)
