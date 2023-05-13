import random

from torch.utils.data import Dataset

from lib.ds.bird_classes import NUM_CLASSES


class BalancingDataset(Dataset):
    _base_set: Dataset

    _length_per_class: int

    _samples_by_class: dict[int, list[tuple[list[int], int]]]
    _index_by_class: dict[int, int]

    def __init__(self, base_set: Dataset, length_per_class=2000):
        self._base_set = base_set
        self._length_per_class = length_per_class

        self._samples_by_class = {cls: [] for cls in range(0, NUM_CLASSES)}
        self._index_by_class = {cls: 0 for cls in range(0, NUM_CLASSES)}

        for sample in base_set:
            self._samples_by_class[sample[-1]].append(sample)

        print(
            f'Created balancing dataset. Original number of samples per class: '
            f'{", ".join([f"{cls}: {len(samples)}" for cls, samples in self._samples_by_class.items()])}, '
            f'new number of samples per class: {self._length_per_class}'
        )

    def __len__(self):
        return self._length_per_class * NUM_CLASSES

    def __getitem__(self, index) -> tuple[list[int], int]:
        if index >= len(self):
            raise IndexError
        cls = random.randrange(0, NUM_CLASSES)
        sample = self._samples_by_class[cls][self._index_by_class[cls]]
        self._index_by_class[cls] = (self._index_by_class[cls] + 1) % len(self._samples_by_class[cls])
        return sample
