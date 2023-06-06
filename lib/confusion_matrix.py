from typing import Iterable, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lib.ds.bird_classes import CLASS_NAMES


def display_confmat(preds: Iterable, targets: Iterable, title: str = None):
    confusion_m: np.ndarray = confusion_matrix(targets, preds, normalize='true')

    display_labels: Optional[list[str]] = None
    if confusion_m.shape[0] == 7:
        display_labels = CLASS_NAMES

    confusion_display = ConfusionMatrixDisplay(confusion_m, display_labels=display_labels)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=300)
    confusion_display.plot(ax=ax, cmap='magma')
    ax.set_title(title)
    plt.show()
