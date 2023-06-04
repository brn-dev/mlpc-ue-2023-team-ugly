import numpy as np
from tqdm import tqdm

from lib.ds.dataset_splitting import N_BIRDS
from lib.ds.numpy_dataset import NumpyDataset


def _find_random_bird_subsequence(
        labels: np.ndarray,
        bird_fragment_indices: dict[int, int],
        rng: np.random.Generator
):
    fragments_per_bird = labels.shape[1]
    bird_nr = rng.choice([bnr for bnr, bfi in bird_fragment_indices.items() if bfi < fragments_per_bird])
    bird_subsequence_start_idx = bird_fragment_indices[bird_nr]
    fragments_left = fragments_per_bird - bird_subsequence_start_idx

    subsequence_length = rng.integers(30, rng.integers(31, 101))

    if subsequence_length > fragments_left:
        subsequence_length = fragments_left
    else:
        while (subsequence_length < fragments_left
               and labels[bird_nr, bird_subsequence_start_idx + subsequence_length] != 0):
            subsequence_length += 1

    return bird_nr, bird_subsequence_start_idx, subsequence_length


def combine_birds_labels_only(
        labels: np.ndarray,
        sequence_length: int,
        num_duplicates=1,
        random_seed=42,
) -> np.ndarray:
    labels = labels.reshape((N_BIRDS, -1))

    if num_duplicates > 1:
        labels = np.reshape(
            np.repeat(np.expand_dims(labels, axis=1), num_duplicates, axis=1),
            newshape=(N_BIRDS, -1)
        )

    labels_random_sequence: np.ndarray = np.ndarray(shape=(0,))

    bird_fragment_indices = {
        bird_nr: 0
        for bird_nr
        in range(N_BIRDS)
    }
    fragments_per_bird = labels.shape[1]

    rng = np.random.default_rng(seed=random_seed)

    for bird_nr in range(N_BIRDS):
        bird_permutation = rng.permutation(labels.shape[1])
        labels[bird_nr, :] = labels[bird_nr, bird_permutation]

    tqdm_desc = f'Creating random label sequence from {num_duplicates} duplicates'
    with tqdm(total=N_BIRDS * fragments_per_bird, desc=tqdm_desc) as progress_bar:
        while not all(bird_fragment_idx == fragments_per_bird for bird_fragment_idx in bird_fragment_indices.values()):
            bird_nr, bird_subsequence_start_idx, subsequence_length = \
                _find_random_bird_subsequence(labels, bird_fragment_indices, rng)

            labels_random_sequence = np.append(
                labels_random_sequence,
                labels[bird_nr, bird_subsequence_start_idx:bird_subsequence_start_idx + subsequence_length],
                axis=0
            )

            bird_fragment_indices[bird_nr] += subsequence_length
            progress_bar.update(subsequence_length)

    return labels_random_sequence.reshape((-1, sequence_length))



def combine_birds(
        numpy_ds: NumpyDataset,
        sequence_length: int,
        num_duplicates=1,
        random_seed=42
) -> NumpyDataset:

    data, labels = numpy_ds.copy()
    data = data.reshape((N_BIRDS, -1, data.shape[-1]))
    labels = labels.reshape((N_BIRDS, -1))

    if num_duplicates > 1:
        data = np.reshape(
            np.repeat(np.expand_dims(data, axis=1), num_duplicates, axis=1),
            newshape=(N_BIRDS, -1, data.shape[-1])
        )
        labels = np.reshape(
            np.repeat(np.expand_dims(labels, axis=1), num_duplicates, axis=1),
            newshape=(N_BIRDS, -1)
        )

    data_random_sequence: np.ndarray = np.ndarray(shape=(0, data.shape[-1]))
    labels_random_sequence: np.ndarray = np.ndarray(shape=(0,))

    bird_fragment_indices = {
        bird_nr: 0
        for bird_nr
        in range(N_BIRDS)
    }
    fragments_per_bird = data.shape[1]

    rng = np.random.default_rng(seed=random_seed)

    for bird_nr in range(N_BIRDS):
        bird_permutation = rng.permutation(data.shape[1])
        data[bird_nr, :] = data[bird_nr, bird_permutation]
        labels[bird_nr, :] = labels[bird_nr, bird_permutation]

    tqdm_desc = f'Creating random sequence from {num_duplicates} duplicates'
    with tqdm(total=N_BIRDS * fragments_per_bird, desc=tqdm_desc) as progress_bar:
        while not all(bird_fragment_idx == fragments_per_bird for bird_fragment_idx in bird_fragment_indices.values()):
            bird_nr, bird_subsequence_start_idx, subsequence_length = \
                _find_random_bird_subsequence(labels, bird_fragment_indices, rng)

            data_random_sequence = np.append(
                data_random_sequence,
                data[bird_nr, bird_subsequence_start_idx:bird_subsequence_start_idx + subsequence_length],
                axis=0
            )
            labels_random_sequence = np.append(
                labels_random_sequence,
                labels[bird_nr, bird_subsequence_start_idx:bird_subsequence_start_idx + subsequence_length],
                axis=0
            )

            bird_fragment_indices[bird_nr] += subsequence_length
            progress_bar.update(subsequence_length)

    return NumpyDataset(
        data_random_sequence.reshape((-1, sequence_length, data.shape[-1])),
        labels_random_sequence.reshape((-1, sequence_length)),
    )
