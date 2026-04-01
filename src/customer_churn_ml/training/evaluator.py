"""Evaluation utilities for binary classification models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from ..utils import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
    y_true,
    y_pred,
    y_prob=None,
    *,
    model_name: str = "model",
) -> dict[str, Any]:
    """Compute the standard churn metrics used in the notebook."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1": float(f1_score(y_true_arr, y_pred_arr)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ),
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
        metrics["fpr"] = fpr.tolist()
        metrics["tpr"] = tpr.tolist()
        metrics["thresholds"] = thresholds.tolist()
        # keep the original probabilities for plotting convenience
        metrics["y_prob"] = y_prob_arr.tolist()
    else:
        metrics["roc_auc"] = None

    logger.info(
        "Evaluated %s: accuracy=%.4f, roc_auc=%s",
        model_name,
        metrics["accuracy"],
        "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}",
    )
    return metrics


def build_comparison_table(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create a compact comparison table from per-model metrics."""

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "ROC_AUC": metrics.get("roc_auc"),
            }
        )
    rows.sort(key=lambda row: row["ROC_AUC"] if row["ROC_AUC"] is not None else -1.0, reverse=True)
    return rows


def plot_roc_curves(results: Mapping[str, Mapping[str, Any]], y_true, ax=None):
    """Plot ROC curves for a set of model results.

    Expects each model's metrics dict to include either:
      - 'y_prob' (iterable of predicted probabilities for the positive class), or
      - precomputed 'fpr' and 'tpr' lists.

    Returns the matplotlib Axes with the plotted curves.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    y_true_arr = np.asarray(y_true)

    plotted_any = False
    for model_name, metrics in results.items():
        # prefer precomputed fpr/tpr if available
        fpr = metrics.get("fpr")
        tpr = metrics.get("tpr")

        if fpr is None or tpr is None:
            y_prob = metrics.get("y_prob")
            if y_prob is None:
                # try also the raw probabilities under another common key
                y_prob = metrics.get("probabilities") or metrics.get("probs")
            if y_prob is None:
                # no ROC info for this model
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true_arr, np.asarray(y_prob))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not compute ROC curve for %s: %s", model_name, exc)
                continue

        roc_value = metrics.get("roc_auc")
        label = model_name if roc_value is None else f"{model_name} (AUC = {roc_value:.3f})"
        ax.plot(fpr, tpr, label=label)
        plotted_any = True

    if not plotted_any:
        logger.info(
            "No ROC curves plotted: no model provided predicted probabilities "
            "or precomputed fpr/tpr."
        )
    else:
        ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves for Churn Prediction Models")
        ax.legend(loc="lower right")

    return ax
