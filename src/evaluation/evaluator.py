"""
Benchmark evaluator for the PEFT Regime Benchmark.

Loads each trained checkpoint, runs inference on the test set,
computes all metrics, and saves results to CSV + plots.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.utils.logger import logger

EXPECTED_COLUMNS = [
    "method",
    "base_model",
    "accuracy",
    "f1_macro",
    "bull_f1",
    "bear_f1",
    "volatile_f1",
    "regime_confidence_score",
    "trainable_params",
    "inference_time_ms",
]


class Evaluator:
    """
    Evaluates all trained PEFT checkpoints and produces benchmark results.

    Usage:
        evaluator = Evaluator(checkpoint_dirs, test_dataset, tokenizer)
        df = evaluator.evaluate_all()
        evaluator.save_results(df)
        evaluator.generate_plots(df)
    """

    def __init__(
        self,
        checkpoint_dirs: list,
        test_dataset=None,
        tokenizer=None,
        results_dir: str = "results",
    ):
        self.checkpoint_dirs = checkpoint_dirs
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all checkpoints and return a DataFrame of results.

        Missing or invalid checkpoints are skipped with a warning.
        """
        rows = []
        for ckpt_dir in self.checkpoint_dirs:
            result = self.evaluate_checkpoint(ckpt_dir)
            if result is not None:
                rows.append(result)

        if not rows:
            logger.warning("No valid checkpoints evaluated — returning empty DataFrame.")
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
        return df

    def evaluate_checkpoint(self, checkpoint_dir: str) -> dict | None:
        """
        Load a checkpoint, run inference on the test set, and compute metrics.

        Reads training_config.json sidecar for method/base_model metadata.
        Returns None and logs a warning if the checkpoint directory is missing
        or the sidecar is unreadable.

        Args:
            checkpoint_dir: Path to a checkpoint directory produced by RegimeTrainer.

        Returns:
            dict with EXPECTED_COLUMNS keys, or None on failure.
        """
        ckpt_path = Path(checkpoint_dir)

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found, skipping: {checkpoint_dir}")
            return None

        sidecar_path = ckpt_path / "training_config.json"
        if not sidecar_path.exists():
            logger.warning(f"No training_config.json in {checkpoint_dir}, skipping.")
            return None

        try:
            with open(sidecar_path) as f:
                sidecar = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read sidecar JSON at {sidecar_path}: {e}")
            return None

        method = sidecar.get("method", "unknown")
        base_model = sidecar.get("base_model", "unknown")
        trainable_params = sidecar.get("trainable_params", 0)

        if self.test_dataset is None or self.tokenizer is None:
            logger.warning("No test dataset or tokenizer provided — returning metadata-only row.")
            return self._empty_row(method, base_model, trainable_params)

        try:
            preds, labels, probs, inference_ms = self._run_inference(ckpt_path)
        except Exception as e:
            logger.warning(f"Inference failed for {checkpoint_dir}: {e}")
            return None

        metrics = compute_all_metrics(preds, labels, probs)
        pcf = metrics["per_class_f1"]

        return {
            "method": method,
            "base_model": base_model,
            "accuracy": round(metrics["accuracy"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "bull_f1": round(pcf["bull"], 4),
            "bear_f1": round(pcf["bear"], 4),
            "volatile_f1": round(pcf["volatile"], 4),
            "regime_confidence_score": round(metrics["regime_confidence_score"], 4),
            "trainable_params": trainable_params,
            "inference_time_ms": round(inference_ms, 2),
        }

    def _run_inference(self, ckpt_path: Path) -> tuple:
        """
        Load model from checkpoint and run inference on self.test_dataset.

        Returns:
            (preds, labels, probs, inference_time_ms)
        """
        import torch
        from transformers import AutoModelForSequenceClassification

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()

        all_preds, all_labels, all_probs = [], [], []
        start = time.time()

        from torch.utils.data import DataLoader

        loader = DataLoader(self.test_dataset, batch_size=8, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["label"].numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float().cpu().numpy()
                probs = self._softmax(logits)
                preds = np.argmax(logits, axis=-1)

                all_preds.extend(preds.tolist())
                all_labels.extend(batch_labels.tolist())
                all_probs.extend(probs.tolist())

        elapsed_ms = (time.time() - start) * 1000.0 / len(all_labels)  # per-sample

        return (
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs),
            elapsed_ms,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def _empty_row(method: str, base_model: str, trainable_params: int) -> dict:
        return {
            "method": method,
            "base_model": base_model,
            "accuracy": None,
            "f1_macro": None,
            "bull_f1": None,
            "bear_f1": None,
            "volatile_f1": None,
            "regime_confidence_score": None,
            "trainable_params": trainable_params,
            "inference_time_ms": None,
        }

    def save_results(self, df: pd.DataFrame, filename: str = "benchmark_results.csv"):
        """Save benchmark results DataFrame to CSV."""
        out_path = self.results_dir / filename
        df.to_csv(out_path, index=False)
        logger.info(f"Results saved to {out_path}")

    def generate_plots(self, df: pd.DataFrame):
        """Generate all benchmark plots to results/plots/."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        if df.empty or df["f1_macro"].isna().all():
            logger.warning("No numeric results to plot.")
            return

        df_plot = df.dropna(subset=["f1_macro"])

        # --- Plot 1: F1 grouped bar chart ---
        fig, ax = plt.subplots(figsize=(10, 5))
        methods = df_plot["method"].unique()
        x = np.arange(len(methods))
        width = 0.35
        base_models = df_plot["base_model"].unique()
        for i, bm in enumerate(base_models):
            subset = df_plot[df_plot["base_model"] == bm].set_index("method")
            vals = [subset.loc[m, "f1_macro"] if m in subset.index else 0 for m in methods]
            ax.bar(x + i * width, vals, width, label=bm)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(methods)
        ax.set_ylabel("F1 Macro")
        ax.set_title("F1 Macro by PEFT Method and Base Model")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(plots_dir / "f1_comparison.png", dpi=150)
        plt.close(fig)

        # --- Plot 2: Regime Confidence Score ---
        fig, ax = plt.subplots(figsize=(10, 5))
        df_rcs = df_plot.dropna(subset=["regime_confidence_score"])
        for i, bm in enumerate(base_models):
            subset = df_rcs[df_rcs["base_model"] == bm].set_index("method")
            vals = [subset.loc[m, "regime_confidence_score"] if m in subset.index else 0 for m in methods]
            ax.bar(x + i * width, vals, width, label=bm)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Regime Confidence Score")
        ax.set_title("Regime Confidence Score by PEFT Method and Base Model")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(plots_dir / "confidence_score_comparison.png", dpi=150)
        plt.close(fig)

        # --- Plot 3: Per-class F1 heatmap ---
        heatmap_data = df_plot[["method", "base_model", "bull_f1", "bear_f1", "volatile_f1"]].copy()
        heatmap_data["label"] = heatmap_data["method"] + "\n" + heatmap_data["base_model"]
        heatmap_data = heatmap_data.set_index("label")[["bull_f1", "bear_f1", "volatile_f1"]]
        heatmap_data.columns = ["Bull", "Bear", "Volatile"]
        fig, ax = plt.subplots(figsize=(8, max(4, len(heatmap_data))))
        sns.heatmap(
            heatmap_data.astype(float),
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Per-Class F1 Heatmap")
        fig.tight_layout()
        fig.savefig(plots_dir / "per_class_f1_heatmap.png", dpi=150)
        plt.close(fig)

        logger.info(f"Plots saved to {plots_dir}")
