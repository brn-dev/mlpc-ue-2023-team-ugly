import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lib.ds.bird_classes import CLASS_NAMES


def display_confmat(preds: torch.Tensor, targets: torch.Tensor, title: str = None):
    confusion_m = confusion_matrix(targets, preds, normalize='true')
    confusion_display = ConfusionMatrixDisplay(confusion_m, display_labels=CLASS_NAMES)
    plt.figure(figsize=(20, 20))
    confusion_display.plot()
    plt.title(title)
    plt.show()
