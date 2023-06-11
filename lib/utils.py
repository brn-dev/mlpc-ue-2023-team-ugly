import contextlib
from tqdm import tqdm
import sys
import numpy as np

class DummyFile(object):
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout

def get_baseline(labels: np.ndarray) -> float:
        labels = labels.flatten()
        unique_labels, labels_count = np.unique(labels, return_counts=True)
        return max(labels_count) / len(labels)