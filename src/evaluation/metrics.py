"""
Evaluation metrics for the PEFT Regime Benchmark.

Includes standard NLP metrics (accuracy, F1) plus the custom
Regime Confidence Score that penalizes high-confidence wrong predictions
with severity weighting based on financial risk.

Label mapping: Bull=0, Bear=1, Volatile=2
Severity logic:
  - Bull↔Bear (opposite direction) = 2.0x severity
  - Bull↔Volatile or Bear↔Volatile (adjacent) = 1.0x severity
  - Calling a Bear market Bull is catastrophically worse than calling it Volatile.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

# Severity matrix: SEVERITY_MATRIX[pred, true] = penalty multiplier
# Rows = predicted label, Cols = true label
# Bull=0, Bear=1, Volatile=2
SEVERITY_MATRIX = np.array(
    [
        [0.0, 2.0, 1.0],  # predicted Bull: vs true Bear=2x, vs true Volatile=1x
        [2.0, 0.0, 1.0],  # predicted Bear: vs true Bull=2x, vs true Volatile=1x
        [1.0, 1.0, 0.0],  # predicted Volatile: vs true Bull=1x, vs true Bear=1x
    ]
)


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return float(accuracy_score(labels, preds))


def f1_macro(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged F1 score across all classes."""
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def per_class_f1(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    Per-class F1 scores.

    Returns:
        dict with keys 'bull', 'bear', 'volatile'.
    """
    scores = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "bull": float(scores[0]),
        "bear": float(scores[1]),
        "volatile": float(scores[2]),
    }


def confusion_matrix_dict(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    Confusion matrix as a nested dict.

    Returns:
        dict of shape {true_label: {pred_label: count}}.
    """
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    regime_names = {0: "bull", 1: "bear", 2: "volatile"}
    return {
        regime_names[i]: {regime_names[j]: int(cm[i, j]) for j in range(3)}
        for i in range(3)
    }


def regime_confidence_score(
    preds: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Custom metric that penalizes high-confidence wrong predictions.

    For each wrong prediction i:
        penalty_i = probabilities[i, preds[i]] * SEVERITY_MATRIX[preds[i], labels[i]]

    Final score = 1.0 - sum(penalties) / (N * 2.0)

    Normalization: max possible penalty per sample = 1.0 (max_prob=1.0) * 2.0 (max severity).
    Dividing by (N * 2.0) maps the score to [0, 1].

    - Perfect predictions → 1.0
    - All wrong with max confidence (prob=1.0) and max severity (Bull↔Bear) → 0.0

    Args:
        preds: Predicted class indices, shape (N,).
        labels: True class indices, shape (N,).
        probabilities: Predicted probabilities, shape (N, num_classes).

    Returns:
        float in [0, 1]. Higher is better.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    n = len(preds)
    if n == 0:
        return 1.0

    wrong_mask = preds != labels
    if not np.any(wrong_mask):
        return 1.0

    wrong_preds = preds[wrong_mask]
    wrong_labels = labels[wrong_mask]
    wrong_probs = probabilities[wrong_mask]

    # Confidence = probability of the (wrong) predicted class
    confidence = wrong_probs[np.arange(len(wrong_preds)), wrong_preds]

    # Severity from the matrix
    severity = SEVERITY_MATRIX[wrong_preds, wrong_labels]

    penalties = confidence * severity
    total_penalty = float(np.sum(penalties))

    score = 1.0 - total_penalty / (n * 2.0)
    # Clip to [0, 1] for safety
    return float(np.clip(score, 0.0, 1.0))


def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Compute all benchmark metrics in one call.

    Returns:
        dict with keys: accuracy, f1_macro, per_class_f1,
                        confusion_matrix, regime_confidence_score.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    return {
        "accuracy": accuracy(preds, labels),
        "f1_macro": f1_macro(preds, labels),
        "per_class_f1": per_class_f1(preds, labels),
        "confusion_matrix": confusion_matrix_dict(preds, labels),
        "regime_confidence_score": regime_confidence_score(preds, labels, probabilities),
    }
