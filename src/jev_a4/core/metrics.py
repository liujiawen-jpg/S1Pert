"""Three-class metrics, independent of Torch and of any prior predictor.

Class order is [up, same, down]. AP uses grouped ties (sklearn's non-interpolated
average precision convention). Missing-class AP is null, not an invented zero.
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ("up", "same", "down")


def average_precision(truth, score):
    truth = np.asarray(truth, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if truth.shape != score.shape or truth.ndim != 1:
        raise ValueError("AP expects aligned one-dimensional arrays")
    if not np.isfinite(score).all():
        raise ValueError("AP scores must be finite")
    positives = int(truth.sum())
    if not positives:
        return None
    order = np.argsort(-score, kind="stable")
    ranked, values = truth[order], score[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    tp = np.cumsum(ranked)[ends].astype(np.float64)
    precision = tp / (ends + 1)
    increments = np.diff(np.r_[0.0, tp]) / positives
    return float(np.sum(increments * precision))


def metrics_from_confusion(confusion):
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.shape != (3, 3) or not np.isfinite(cm).all() or np.any(cm < 0):
        raise ValueError("confusion matrix must be finite, nonnegative and [3,3]")
    true_support, predicted_support = cm.sum(1), cm.sum(0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted_support, out=np.zeros(3), where=predicted_support > 0)
    recall = np.divide(tp, true_support, out=np.zeros(3), where=true_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=(precision + recall) > 0)
    present = true_support > 0
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else None,
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else None,
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": float(true_support[i]),
                              "predicted_support": float(predicted_support[i])}
                      for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix_true_rows_predicted_columns": cm.tolist(),
    }


def classification_metrics(labels, probabilities, targets=None):
    raw_labels = np.asarray(labels)
    if raw_labels.dtype.kind not in "iuf" or not np.isfinite(raw_labels).all():
        raise ValueError("strict labels must be finite integers in 0..2")
    if np.any((raw_labels < 0) | (raw_labels > 2)) or np.any(raw_labels != np.floor(raw_labels)):
        raise ValueError("strict labels must be integers in 0..2")
    labels = raw_labels.astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != (len(labels), 3) or not len(labels):
        raise ValueError("non-empty labels and [N,3] probabilities are required")
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("strict labels must be in 0..2")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("invalid probabilities")
    if not np.allclose(probabilities.sum(1), 1, atol=2e-5):
        raise ValueError("probabilities do not sum to one")
    predicted = probabilities.argmax(1)
    confusion = np.bincount(labels * 3 + predicted, minlength=9).reshape(3, 3)
    result = metrics_from_confusion(confusion)
    for i, name in enumerate(CLASS_NAMES):
        result["per_class"][name]["support"] = int(confusion[i].sum())
        result["per_class"][name]["predicted_support"] = int(confusion[:, i].sum())
        result["per_class"][name]["average_precision"] = average_precision(labels == i, probabilities[:, i])
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1)
    one_hot = np.eye(3)[labels]
    changed_true, changed_pred = labels != 1, predicted != 1
    tp = int(np.sum(changed_true & changed_pred))
    cp = tp / int(changed_pred.sum()) if changed_pred.any() else 0.0
    cr = tp / int(changed_true.sum()) if changed_true.any() else 0.0
    result.update({
        "records": int(len(labels)), "class_order": list(CLASS_NAMES),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "nll": float(-np.log(chosen).mean()),
        "brier_multiclass_sum": float(np.square(probabilities - one_hot).sum(1).mean()),
        "changed": {"precision": float(cp), "recall": float(cr),
                    "f1": float(2 * cp * cr / (cp + cr)) if cp + cr else 0.0,
                    "support": int(changed_true.sum()),
                    "average_precision": average_precision(changed_true, probabilities[:, 0] + probabilities[:, 2])},
        "macro_f1_missing_class_convention": "fixed three classes; absent true and predicted class has F1=0",
        "balanced_accuracy_missing_class_convention": "mean recall over reference-present classes",
    })
    if targets is not None:
        targets = np.asarray(targets)
        if targets.shape != labels.shape:
            raise ValueError("targets are not record aligned")
        unique, inverse, counts = np.unique(targets, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
        weighted = np.bincount(labels * 3 + predicted, weights=weights, minlength=9).reshape(3, 3)
        result["targets"] = int(len(unique))
        result["equal_target_weighted_confusion_metrics"] = metrics_from_confusion(weighted)
        result["equal_target_weighting"] = "each target total weight 1; derive metrics from weighted confusion (not average per-target F1)"
    return result
