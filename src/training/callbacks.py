"""
Custom training callbacks for the PEFT Regime Benchmark.

Logs per-epoch metrics to W&B: train_loss, eval_loss, eval_accuracy, eval_f1_macro,
trainable_params, gpu_memory_mb, epoch_time_seconds.
"""

import time

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.utils.logger import log_metrics, logger


class RegimeBenchmarkCallback(TrainerCallback):
    """
    Logs per-epoch training metrics to W&B and console.

    Tracks: train_loss, eval_loss, eval_accuracy, eval_f1_macro,
    trainable_params, gpu_memory_mb, epoch_time_seconds.
    """

    def __init__(self, trainable_params: int = 0):
        self.trainable_params = trainable_params
        self._epoch_start_time = None

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Reset peak memory stats and record epoch start time."""
        self._epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log epoch metrics to W&B."""
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0.0

        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Extract latest metrics from trainer log history
        metrics = {"trainable_params": self.trainable_params, "epoch_time_seconds": epoch_time}

        if torch.cuda.is_available():
            metrics["gpu_memory_mb"] = gpu_memory_mb

        # Pull the most recent logged values
        if state.log_history:
            latest = state.log_history[-1]
            for key in ["loss", "eval_loss", "eval_accuracy", "eval_f1_macro"]:
                if key in latest:
                    metrics[key] = latest[key]

        log_metrics(metrics, step=state.global_step)
        logger.info(
            f"Epoch {state.epoch:.0f} | "
            f"time={epoch_time:.1f}s | "
            f"gpu_mem={gpu_memory_mb:.0f}MB | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        """Forward per-step logs to W&B."""
        if logs:
            log_metrics(logs, step=state.global_step)
