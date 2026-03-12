"""
CLI entry point for training a single PEFT method.

Usage:
    python scripts/train.py --method lora --base-model llama3 --seed 42
    python scripts/train.py --config configs/qlora.yaml --seed 0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_tokenized_dataset, load_dataset
from src.training.trainer import RegimeTrainer
from src.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PEFT model for market regime detection."
    )
    parser.add_argument(
        "--method",
        choices=["lora", "qlora", "prefix_tuning", "ia3"],
        help="PEFT method to use (overrides config file).",
    )
    parser.add_argument(
        "--base-model",
        choices=["llama3", "mistral7b"],
        default="llama3",
        help="Base model to fine-tune (default: llama3).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file. If not provided, uses configs/{method}.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (-1 = use num_epochs from config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine config path
    if args.config:
        config_path = args.config
    elif args.method:
        config_path = f"configs/{args.method}.yaml"
    else:
        raise ValueError("Provide either --method or --config.")

    logger.info(f"Loading config from {config_path}")
    trainer = RegimeTrainer.from_yaml(config_path)

    # Override config with CLI args
    if args.method:
        trainer.config["method"] = args.method
        trainer.method = args.method
    if args.base_model:
        trainer.config["base_model"] = args.base_model
        trainer.base_model_name = args.base_model
    trainer.config["seed"] = args.seed
    if args.max_steps != -1:
        trainer.config["max_steps"] = args.max_steps

    # Setup model
    trainer.setup()

    # Load and tokenize data
    logger.info("Loading dataset...")
    dataset_dict = load_dataset(seed=args.seed)
    tokenized = get_tokenized_dataset(dataset_dict, trainer.tokenizer, max_length=trainer.config.get("max_length", 128))

    from src.data.loader import get_class_distribution

    for split in ["train", "validation", "test"]:
        dist = get_class_distribution(dataset_dict[split])
        logger.info(f"{split} distribution: {dist}")

    # Train
    checkpoint_dir = trainer.train(
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )
    logger.info(f"Training complete. Checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
