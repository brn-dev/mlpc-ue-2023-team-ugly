from collections import Counter
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import sklearn

COST_MATRICES = dict({
    2: np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ]),
    7: np.array([
        [0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.25, 1., -0.3, -0.1, -0.1, -0.1, -0.1],
        [-0.02, -0.1, 1., -0.1, -0.1, -0.1, -0.1],
        [-0.25, -0.1, -0.3, 1., -0.1, -0.1, -0.1],
        [-0.25, -0.1, -0.3, -0.1, 1., -0.1, -0.1],
        [-0.25, -0.1, -0.3, -0.1, -0.1, 1., -0.1],
        [-0.25, -0.1, -0.3, -0.1, -0.1, -0.1, 1.]],
    ),
})


@dataclass
class Metrics(dict):
    epoch: Optional[int]
    avg_loss: Optional[float]
    num_samples: Optional[int]
    num_correct: Optional[int]
    acc: Optional[float]
    bacc: Optional[float]
    score: Optional[float]

    def __str__(self):
        metrics = vars(self)

        metrics_stringified: list[str] = []
        for key, value in metrics.items():
            if value is not None:
                value_str: str
                if isinstance(value, int):
                    value_str = f'{value:5d}'
                elif isinstance(value, float):
                    value_str = f'{value:.6f}'
                else:
                    value_str = str(value)
                metrics_stringified.append(f'{key} = {value_str}')

        return ', '.join(metrics_stringified)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, attribute_key: str):
        return getattr(self, attribute_key)

    def __setitem__(self, attribute_key: str, attribute_value: Any):
        setattr(self, attribute_key, attribute_value)


# noinspection PyMethodMayBeStatic
class LabelCollector:

    def __init__(self):
        self.total_loss = 0.0
        self.pred_labels: np.ndarray = np.zeros((0,)).astype(int)
        self.target_labels: np.ndarray = np.zeros((0,)).astype(int)

    def update(self, loss: float, pred_labels: np.ndarray, target_labels: np.ndarray):
        self.total_loss += loss
        self.pred_labels = np.concatenate((self.pred_labels, pred_labels))
        self.target_labels = np.concatenate((self.target_labels, target_labels))

    def generate_metrics(self, epoch: int = -1) -> Metrics:
        num_samples = len(self.target_labels)
        num_correct = int((self.pred_labels == self.target_labels).sum().item())

        if num_samples == 0:
            return Metrics(
                    epoch=epoch,
                    avg_loss=-1,
                    num_samples=0,
                    num_correct=0,
                    acc=-1,
                    bacc=-1,
                    score=-1,
                )

        avg_loss = self.total_loss / num_samples
        acc = num_correct / num_samples
        bacc = sklearn.metrics.balanced_accuracy_score(self.target_labels, self.pred_labels)
        score = self.__calc_score()

        return Metrics(
            epoch=epoch,
            avg_loss=avg_loss,
            num_samples=num_samples,
            num_correct=num_correct,
            acc=acc,
            bacc=bacc,
            score=score,
        )

    def count_labels(self) -> tuple[dict[int, int], dict[int, int]]:
        target_label_counts = self.__count_elements(self.target_labels)
        pred_label_counts = self.__count_elements(self.pred_labels)
        return target_label_counts, pred_label_counts

    def __count_elements(self, arr: np.ndarray) -> dict[int, int]:
        counter = Counter()
        counter.update(arr)
        return counter

    def __calc_score(self) -> float:
        confusion_matrix: np.ndarray = sklearn.metrics.confusion_matrix(self.target_labels, self.pred_labels)
        optimal_matrix = np.diag(confusion_matrix.sum(axis=1))

        cost_matrix = COST_MATRICES[confusion_matrix.shape[0]]

        return (cost_matrix * confusion_matrix).sum() / (cost_matrix * optimal_matrix).sum()


TrainAndEvaluationMetrics = tuple[Metrics, Metrics]
TrainingEvaluationAndTestMetrics = tuple[Metrics, Metrics, Optional[Metrics]]
TrainingRunMetrics = list[TrainAndEvaluationMetrics]
CVFoldsMetrics = list[TrainingRunMetrics]


@dataclass
class MetricsCollection:
    cv_metrics: CVFoldsMetrics
    test_metrics: Optional[Metrics]

    def __init__(self):
        self.cv_metrics = []
        self.test_metrics = None

    def append_cv_fold(self, fold_metrics: TrainingRunMetrics):
        self.cv_metrics.append(fold_metrics)

    def get_score(self):
        return self.test_metrics.score


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
        epoch=0,
        avg_loss=0.0,
        num_samples=0,
        num_correct=0,
        acc=0.0,
        bacc=0.0,
        score=0.0
    )

    for metrics in metrics_list:
        for attribute_key, attribute_value in vars(metrics).items():
            avg_metrics[attribute_key] += attribute_value

    num_metrics = len(metrics_list)
    for attribute_key, attribute_value in vars(avg_metrics).items():
        avg_metrics[attribute_key] /= num_metrics

    return avg_metrics
