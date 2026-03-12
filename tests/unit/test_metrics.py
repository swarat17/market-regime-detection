"""
Phase 4 unit tests — Custom Evaluation Metrics.

No GPU or model downloads needed. Pure numpy/sklearn.
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    per_class_f1,
    regime_confidence_score,
)

# Label constants: Bull=0, Bear=1, Volatile=2


def test_perfect_predictions_score_one():
    """All correct predictions → regime_confidence_score == 1.0."""
    preds = np.array([0, 1, 2, 0, 1, 2])
    labels = np.array([0, 1, 2, 0, 1, 2])
    # Probabilities don't matter for correct predictions, but must be valid
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])
    score = regime_confidence_score(preds, labels, probs)
    assert score == 1.0, f"Expected 1.0 for perfect predictions, got {score}"


def test_opposite_regime_error_penalized_more():
    """
    Bull→Bear error (severity=2.0) reduces score more than Bull→Volatile (severity=1.0)
    at the same confidence level.
    """
    confidence = 0.9

    # Case 1: predict Bull (0), true is Bear (1) — opposite, severity=2.0
    preds_opposite = np.array([0])
    labels_bear = np.array([1])
    probs_opposite = np.array([[confidence, 1 - confidence, 0.0]])
    score_opposite = regime_confidence_score(preds_opposite, labels_bear, probs_opposite)

    # Case 2: predict Bull (0), true is Volatile (2) — adjacent, severity=1.0
    preds_adjacent = np.array([0])
    labels_volatile = np.array([2])
    probs_adjacent = np.array([[confidence, 0.0, 1 - confidence]])
    score_adjacent = regime_confidence_score(preds_adjacent, labels_volatile, probs_adjacent)

    assert score_opposite < score_adjacent, (
        f"Bull→Bear score ({score_opposite:.4f}) should be lower than "
        f"Bull→Volatile score ({score_adjacent:.4f}) at same confidence"
    )


def test_high_confidence_wrong_penalized_more_than_low():
    """Same wrong label, higher confidence → lower regime_confidence_score."""
    labels = np.array([1])  # true = Bear

    # High confidence wrong: predict Bull with prob=0.9
    preds = np.array([0])
    probs_high = np.array([[0.9, 0.1, 0.0]])
    score_high = regime_confidence_score(preds, labels, probs_high)

    # Low confidence wrong: predict Bull with prob=0.3
    probs_low = np.array([[0.3, 0.7, 0.0]])
    score_low = regime_confidence_score(preds, labels, probs_low)

    assert score_high < score_low, (
        f"High confidence wrong ({score_high:.4f}) should score lower than "
        f"low confidence wrong ({score_low:.4f})"
    )


def test_compute_all_metrics_has_all_keys():
    """compute_all_metrics returns dict with all 5 expected keys."""
    preds = np.array([0, 1, 2, 0])
    labels = np.array([0, 1, 1, 2])
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
    ])
    result = compute_all_metrics(preds, labels, probs)

    expected_keys = {"accuracy", "f1_macro", "per_class_f1", "confusion_matrix", "regime_confidence_score"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


def test_per_class_f1_has_three_regimes():
    """per_class_f1 dict has exactly the keys 'bull', 'bear', 'volatile'."""
    preds = np.array([0, 1, 2, 0, 1, 2])
    labels = np.array([0, 1, 2, 1, 0, 2])
    result = per_class_f1(preds, labels)

    assert set(result.keys()) == {"bull", "bear", "volatile"}, (
        f"Expected keys {{'bull', 'bear', 'volatile'}}, got {set(result.keys())}"
    )
    for regime, score in result.items():
        assert 0.0 <= score <= 1.0, f"F1 for {regime} out of range: {score}"
