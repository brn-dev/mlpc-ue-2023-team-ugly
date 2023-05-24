from typing import Iterable

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lib.ds.bird_classes import CLASS_NAMES


def display_confmat(preds: Iterable, targets: Iterable, title: str = None):
    confusion_m = confusion_matrix(targets, preds, normalize='true')
    confusion_display = ConfusionMatrixDisplay(confusion_m, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    confusion_display.plot(ax=ax, cmap='magma')
    ax.set_title(title)
    plt.show()
