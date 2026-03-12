"""
LoRA rank sensitivity analysis.

Trains LoRA + Llama-3.2-3B five times with r = 4, 8, 16, 32, 64
(all other hyperparameters fixed). Records F1, regime_confidence_score,
trainable_params, and training_time_seconds for each rank.

Saves:
    results/rank_sensitivity.csv    (5 rows)
    results/plots/rank_sensitivity.png  (dual-axis: F1 + param count vs rank)

Usage:
    python scripts/rank_sensitivity.py
    python scripts/rank_sensitivity.py --max-steps 100  # quick test
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils.logger import logger

RANK_VALUES = [4, 8, 16, 32, 64]

BASE_CONFIG = {
    "method": "lora",
    "base_model": "llama3",
    "model_name_or_path": "meta-llama/Llama-3.2-3B",
    "dataset": "financial_phrasebank",
    "max_length": 128,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 3e-4,
    "num_epochs": 2,
    "fp16": True,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 0,
    "optim": "paged_adamw_8bit",
    "seed": 42,
    "quantize": True,
    "bnb_4bit_quant_type": "fp4",
    "bnb_4bit_use_double_quant": False,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    # For r >= 32 add k_proj and o_proj for better expressivity
    "target_modules_small": ["q_proj", "v_proj"],
    "target_modules_large": ["q_proj", "v_proj", "k_proj", "o_proj"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA rank sensitivity sweep.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps per rank (-1 = use num_epochs). Use small value for testing.",
    )
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def run_rank_sweep(max_steps: int = -1, results_dir: str = "results") -> pd.DataFrame:
    from src.data.loader import get_tokenized_dataset, load_dataset
    from src.evaluation.metrics import compute_all_metrics
    from src.models.base_loader import load_base_model
    from src.models.peft_factory import create_peft_model
    from src.training.trainer import RegimeTrainer

    import numpy as np
    import torch

    logger.info("Loading dataset for rank sensitivity sweep...")
    dataset_dict = load_dataset(seed=42)

    rows = []

    for rank in RANK_VALUES:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training LoRA with r={rank}")
        logger.info(f"{'='*50}")

        target_modules = (
            BASE_CONFIG["target_modules_large"]
            if rank >= 32
            else BASE_CONFIG["target_modules_small"]
        )

        config = {
            **{k: v for k, v in BASE_CONFIG.items() if not k.startswith("target_modules_")},
            "lora_r": rank,
            "target_modules": target_modules,
            "output_dir": f"models/rank_sweep/r{rank}",
            "max_steps": max_steps,
        }

        t_start = time.time()
        trainer = RegimeTrainer(config)
        trainer.setup()

        tokenized = get_tokenized_dataset(dataset_dict, trainer.tokenizer, max_length=128)

        checkpoint_dir = trainer.train(
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
        )
        training_time = time.time() - t_start

        # Evaluate on test set
        trainable_params = trainer.peft_info["trainable_params"]

        preds_list, labels_list, probs_list = [], [], []
        trainer.model.eval()

        from torch.utils.data import DataLoader

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = DataLoader(tokenized["test"], batch_size=4, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["label"].numpy()

                outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float().cpu().numpy()

                exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp / exp.sum(axis=-1, keepdims=True)

                preds_list.extend(np.argmax(logits, axis=-1).tolist())
                labels_list.extend(batch_labels.tolist())
                probs_list.extend(probs.tolist())

        metrics = compute_all_metrics(
            np.array(preds_list),
            np.array(labels_list),
            np.array(probs_list),
        )

        row = {
            "lora_r": rank,
            "f1_macro": round(metrics["f1_macro"], 4),
            "regime_confidence_score": round(metrics["regime_confidence_score"], 4),
            "trainable_params": trainable_params,
            "training_time_seconds": round(training_time, 1),
        }
        rows.append(row)
        logger.info(f"r={rank} | {row}")

    df = pd.DataFrame(rows)

    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    csv_path = results_path / "rank_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Rank sensitivity results saved to {csv_path}")

    _plot_rank_sensitivity(df, results_path / "plots")

    return df


def _plot_rank_sensitivity(df: pd.DataFrame, plots_dir: Path):
    """Dual-axis line chart: F1 (left) and trainable params (right) vs LoRA rank."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_f1 = "#2563eb"
    color_params = "#dc2626"

    ax1.set_xlabel("LoRA Rank (r)")
    ax1.set_ylabel("F1 Macro", color=color_f1)
    ax1.plot(df["lora_r"], df["f1_macro"], "o-", color=color_f1, linewidth=2, label="F1 Macro")
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Trainable Parameters", color=color_params)
    ax2.plot(
        df["lora_r"],
        df["trainable_params"],
        "s--",
        color=color_params,
        linewidth=2,
        label="Trainable Params",
    )
    ax2.tick_params(axis="y", labelcolor=color_params)

    ax1.set_xticks(df["lora_r"].tolist())
    fig.suptitle("LoRA Rank Sensitivity: F1 vs Trainable Parameters", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    out_path = plots_dir / "rank_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Rank sensitivity plot saved to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    run_rank_sweep(max_steps=args.max_steps, results_dir=args.results_dir)
