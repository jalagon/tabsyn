import enum
from typing import Any, Optional, Tuple, Dict, Union, cast
from functools import partial

import numpy as np
import scipy.special
import sklearn.metrics as skm

from . import util
from .util import TaskType


class PredictionType(enum.Enum):
    """Type of predictions produced by a model."""

    LOGITS = 'logits'
    PROBS = 'probs'

class MetricsReport:
    """Wrapper around metric dictionaries for easier access."""

    def __init__(self, report: dict, task_type: TaskType):
        """Create the report from a raw metrics dictionary."""
        self._res = {k: {} for k in report.keys()}
        if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
            self._metrics_names = ["acc", "f1"]
            for k in report.keys():
                self._res[k]["acc"] = report[k]["accuracy"]
                self._res[k]["f1"] = report[k]["macro avg"]["f1-score"]
                if task_type == TaskType.BINCLASS:
                    self._res[k]["roc_auc"] = report[k]["roc_auc"]
                    self._metrics_names.append("roc_auc")

        elif task_type == TaskType.REGRESSION:
            self._metrics_names = ["r2", "rmse"]
            for k in report.keys():
                self._res[k]["r2"] = report[k]["r2"]
                self._res[k]["rmse"] = report[k]["rmse"]
        else:
            raise "Unknown TaskType!"

    def get_splits_names(self) -> list[str]:
        """Return the available split names (train/val/test)."""
        return list(self._res.keys())

    def get_metrics_names(self) -> list[str]:
        """Return the names of stored metrics."""
        return self._metrics_names

    def get_metric(self, split: str, metric: str) -> float:
        """Fetch a metric for a given split."""
        return self._res[split][metric]

    def get_val_score(self) -> float:
        """Return the primary validation score."""
        return self._res["val"].get("r2", self._res["val"]["f1"])
    
    def get_test_score(self) -> float:
        """Return the primary test score."""
        return self._res["test"].get("r2", self._res["test"]["f1"])
    
    def print_metrics(self) -> dict:
        """Pretty-print the metrics and return a rounded copy."""
        res = {
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]}
        }

        print("*" * 100)
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])

        return res

class SeedsMetricsReport:
    """Aggregate metrics across multiple random seeds."""

    def __init__(self) -> None:
        self._reports: list[MetricsReport] = []

    def add_report(self, report: MetricsReport) -> None:
        """Add a single seed's metrics."""
        self._reports.append(report)

    def get_mean_std(self) -> dict:
        """Compute mean and standard deviation of metrics across seeds."""
        res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def print_result(self) -> dict:
        """Print aggregated validation and test results."""
        res = {
            split: {k: float(np.around(self._agg_res[split][k], 4)) for k in self._agg_res[split]}
            for split in ["val", "test"]
        }
        print("=" * 100)
        print("EVAL RESULTS:")
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])
        print("=" * 100)
        return res

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray, std: float | None = None) -> float:
    """Root mean squared error with optional rescaling."""
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray,
    task_type: TaskType,
    prediction_type: Optional[PredictionType],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert model outputs to class labels and probabilities."""
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        util.raise_unknown('prediction_type', prediction_type)

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute evaluation metrics for regression and classification tasks."""
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert 'std' in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info['std'])
        r2 = skm.r2_score(y_true, y_pred)
        result = {'rmse': rmse, 'r2': r2}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == TaskType.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result
