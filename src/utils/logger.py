"""
Logging and W&B utilities for the PEFT Regime Benchmark.

IMPORTANT: Must be imported before any tokenizer import to set
TOKENIZERS_PARALLELISM=false and avoid deadlocks on Windows.
"""

import logging
import os

# Must be set before any HuggingFace tokenizer import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger("peft_benchmark")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_wandb(project: str, config: dict, run_name: str = None):
    """
    Initialize a W&B run.

    Returns wandb.Run or a mock object if WANDB_MODE=disabled.
    """
    import wandb

    run = wandb.init(
        project=project,
        config=config,
        name=run_name,
        reinit=True,
    )
    return run


def log_metrics(metrics: dict, step: int = None):
    """Log metrics to W&B. No-op if W&B is disabled."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def finish_run():
    """Finish the current W&B run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
