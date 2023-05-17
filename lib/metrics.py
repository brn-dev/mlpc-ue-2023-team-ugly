from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import sklearn

from lib.torch_device import get_torch_device


@dataclass
class Metrics(dict):
    avg_loss: Optional[float]
    num_samples: Optional[int]
    num_correct: Optional[int]
    acc: Optional[float]
    bacc: Optional[float]

    def __str__(self):
        metrics = vars(self)

        metrics_stringified: list[str] = []
        for key, value in metrics.items():
            if value is not None:
                value_str: str
                if value % 1.0 == 0:
                    value_str = f'{int(value):5d}'
                else:
                    value_str = f'{value:.6f}'
                metrics_stringified.append(f'{key} = {value_str}')

        return ', '.join(metrics_stringified)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, attribute_key: str):
        return getattr(self, attribute_key)

    def __setitem__(self, attribute_key: str, attribute_value: Any):
        setattr(self, attribute_key, attribute_value)


TrainAndEvaluationMetrics = tuple[Metrics, Optional[Metrics]]
TrainingRunMetrics = list[TrainAndEvaluationMetrics]
CVFoldsMetrics = list[TrainingRunMetrics]


class MetricsCollector:

    def __init__(self):
        self.total_loss = 0.0
        self.pred_labels: np.ndarray = np.zeros((0,)).astype(int)
        self.target_labels: np.ndarray = np.zeros((0,)).astype(int)

    def update(self, loss: float, pred_labels: np.ndarray, target_labels: np.ndarray):
        self.total_loss += loss
        self.pred_labels = np.concatenate((self.pred_labels, pred_labels))
        self.target_labels = np.concatenate((self.target_labels, target_labels))

    def generate_metrics(self) -> Metrics:
        num_samples = len(self.target_labels)
        num_correct = int((self.pred_labels == self.target_labels).sum().item())

        avg_loss = self.total_loss / num_samples
        acc = num_correct / num_samples
        bacc = sklearn.metrics.balanced_accuracy_score(self.target_labels, self.pred_labels)

        return Metrics(
            avg_loss=avg_loss,
            num_samples=num_samples,
            num_correct=num_correct,
            acc=acc,
            bacc=bacc
        )


def calculate_average_metrics_for_final_epoch_of_folds(cv_folds_metrics: CVFoldsMetrics) -> TrainAndEvaluationMetrics:
    avg_train_metrics = calculate_average_metrics([
        fold_metrics[-1][0]
        for fold_metrics
        in cv_folds_metrics
    ])

    avg_evaluation_metrics: Optional[Metrics] = None
    if cv_folds_metrics[0][-1][1] is not None:
        avg_evaluation_metrics = calculate_average_metrics([
            fold_metrics[-1][1]
            for fold_metrics
            in cv_folds_metrics
        ])

    return avg_train_metrics, avg_evaluation_metrics


def calculate_average_metrics_per_epoch(cv_folds_metrics: CVFoldsMetrics) -> TrainingRunMetrics:
    avg_epoch_metrics: TrainingRunMetrics = []
    for epoch in range(len(cv_folds_metrics[0])):
        train_avg_metrics = calculate_average_metrics([
            fold_metrics[epoch][0]
            for fold_metrics
            in cv_folds_metrics
        ])
        evaluation_avg_metrics = calculate_average_metrics([
            fold_metrics[epoch][1]
            for fold_metrics
            in cv_folds_metrics
            if fold_metrics[epoch][1] is not None
        ])
        avg_epoch_metrics.append((train_avg_metrics, evaluation_avg_metrics))
    return avg_epoch_metrics


def calculate_average_metrics(metrics_list: list[Metrics]):
    avg_metrics = Metrics(
        avg_loss=0.0,
        num_samples=0,
        num_correct=0,
        acc=0.0,
        bacc=0.0,
    )

    for metrics in metrics_list:
        for attribute_key, attribute_value in vars(metrics).items():
            avg_metrics[attribute_key] += attribute_value

    num_metrics = len(metrics_list)
    for attribute_key, attribute_value in vars(avg_metrics).items():
        avg_metrics[attribute_key] /= num_metrics

    return avg_metrics


