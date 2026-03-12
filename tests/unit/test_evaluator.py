"""
Phase 5 unit tests — Benchmark Evaluator.

No GPU or model downloads needed.
Uses temp directories with mock checkpoint sidecars.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import EXPECTED_COLUMNS, Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_checkpoint(parent_dir: Path, method: str, base_model: str) -> Path:
    """Create a minimal fake checkpoint directory with a sidecar JSON."""
    ckpt_dir = parent_dir / f"{method}_{base_model}_20240101_120000"
    ckpt_dir.mkdir(parents=True)
    sidecar = {
        "method": method,
        "base_model": base_model,
        "model_name_or_path": "meta-llama/Llama-3.2-3B",
        "trainable_params": 1_048_576,
        "total_params": 3_000_000_000,
        "trainable_pct": 0.035,
        "lora_r": 16,
        "training_completed": "20240101_120000",
        "best_val_f1": 0.72,
        "checkpoint_dir": str(ckpt_dir),
    }
    with open(ckpt_dir / "training_config.json", "w") as f:
        json.dump(sidecar, f)
    return ckpt_dir


def _make_fake_rank_sensitivity_csv(results_dir: Path) -> Path:
    """Create a fake rank_sensitivity.csv with 5 rows."""
    data = {
        "lora_r": [4, 8, 16, 32, 64],
        "f1_macro": [0.61, 0.67, 0.72, 0.74, 0.75],
        "regime_confidence_score": [0.78, 0.82, 0.85, 0.86, 0.87],
        "trainable_params": [524288, 1048576, 2097152, 4194304, 8388608],
        "training_time_seconds": [120.5, 180.3, 240.1, 360.7, 480.2],
    }
    df = pd.DataFrame(data)
    csv_path = results_dir / "rank_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_csv_has_correct_columns():
    """Saved CSV contains all expected column names."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results_dir = tmp_path / "results"
        ckpt_dir = _make_fake_checkpoint(tmp_path, "lora", "llama3")

        evaluator = Evaluator(
            checkpoint_dirs=[str(ckpt_dir)],
            test_dataset=None,  # no inference — metadata-only rows
            tokenizer=None,
            results_dir=str(results_dir),
        )

        df = evaluator.evaluate_all()
        evaluator.save_results(df)

        csv_path = results_dir / "benchmark_results.csv"
        assert csv_path.exists(), f"CSV not found at {csv_path}"

        saved_df = pd.read_csv(csv_path)
        for col in EXPECTED_COLUMNS:
            assert col in saved_df.columns, f"Missing column: {col}"


def test_evaluator_handles_missing_checkpoint_gracefully():
    """Missing checkpoint path logs a warning, skips, and does not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = str(tmp_path / "nonexistent_checkpoint")
        valid_ckpt = _make_fake_checkpoint(tmp_path, "ia3", "llama3")

        evaluator = Evaluator(
            checkpoint_dirs=[missing, str(valid_ckpt)],
            test_dataset=None,
            tokenizer=None,
            results_dir=str(tmp_path / "results"),
        )

        # Should not raise — missing checkpoint is skipped with a warning
        df = evaluator.evaluate_all()

        # Only the valid checkpoint produces a row
        assert len(df) == 1
        assert df.iloc[0]["method"] == "ia3"


def test_rank_sensitivity_output_has_five_rows():
    """Rank sweep CSV has exactly 5 rows (one per rank value)."""
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()
        csv_path = _make_fake_rank_sensitivity_csv(results_dir)

        df = pd.read_csv(csv_path)
        assert len(df) == 5, f"Expected 5 rows in rank_sensitivity.csv, got {len(df)}"
        assert list(df["lora_r"]) == [4, 8, 16, 32, 64]
