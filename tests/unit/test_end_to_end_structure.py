"""
End-to-end structure tests for Phase 7.

Verifies that configs are loadable and that the metrics output format
is compatible with what the evaluator expects.
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def test_all_configs_loadable():
    """All four YAML config files must load without error and have required keys."""
    required_keys = [
        "method",
        "base_model",
        "learning_rate",
        "num_epochs",
        "output_dir",
    ]
    config_files = list(CONFIG_DIR.glob("*.yaml"))

    assert (
        len(config_files) == 4
    ), f"Expected 4 config files in configs/, found {len(config_files)}: {config_files}"

    for cfg_path in config_files:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict), f"{cfg_path.name}: expected dict, got {type(cfg)}"
        for key in required_keys:
            assert key in cfg, f"{cfg_path.name}: missing required key '{key}'"


def test_metrics_and_evaluator_compatible():
    """
    compute_all_metrics output keys must match the columns the evaluator writes to CSV.

    The evaluator reads: accuracy, f1_macro, bull_f1, bear_f1, volatile_f1,
    regime_confidence_score from compute_all_metrics output (via per_class_f1 sub-dict).
    """
    import numpy as np

    from src.evaluation.metrics import compute_all_metrics

    # Three perfect predictions (one per class)
    preds = np.array([0, 1, 2])
    labels = np.array([0, 1, 2])
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])

    metrics = compute_all_metrics(preds, labels, probs)

    # Top-level keys the evaluator accesses directly
    assert "accuracy" in metrics, "Missing 'accuracy'"
    assert "f1_macro" in metrics, "Missing 'f1_macro'"
    assert "regime_confidence_score" in metrics, "Missing 'regime_confidence_score'"

    # Per-class F1 sub-dict
    assert "per_class_f1" in metrics, "Missing 'per_class_f1'"
    pcf = metrics["per_class_f1"]
    for regime in ("bull", "bear", "volatile"):
        assert regime in pcf, f"Missing per-class F1 for '{regime}'"
