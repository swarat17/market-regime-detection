"""
CLI entry point for evaluating all trained PEFT checkpoints.

Usage:
    python scripts/evaluate.py --checkpoint-dirs models/lora_llama3/... models/qlora_llama3/...
    python scripts/evaluate.py --auto-discover  # scans models/ directory
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_tokenized_dataset, load_dataset
from src.evaluation.evaluator import Evaluator
from src.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PEFT benchmark checkpoints.")
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        help="Explicit list of checkpoint directories to evaluate.",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Auto-discover all checkpoints under models/ directory.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save benchmark_results.csv and plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_checkpoints(models_dir: str = "models") -> list:
    """Find all directories containing a training_config.json sidecar."""
    root = Path(models_dir)
    if not root.exists():
        return []
    return [str(p.parent) for p in root.rglob("training_config.json")]


def main():
    args = parse_args()

    if args.auto_discover:
        checkpoint_dirs = discover_checkpoints()
        logger.info(f"Auto-discovered {len(checkpoint_dirs)} checkpoints.")
    elif args.checkpoint_dirs:
        checkpoint_dirs = args.checkpoint_dirs
    else:
        logger.error("Provide --checkpoint-dirs or --auto-discover.")
        sys.exit(1)

    if not checkpoint_dirs:
        logger.error("No checkpoints found.")
        sys.exit(1)

    # Load test split
    logger.info("Loading test dataset...")
    dataset_dict = load_dataset(seed=args.seed)

    # We need the tokenizer from one of the checkpoints
    # Use the first valid checkpoint's sidecar to find model name
    import json

    tokenizer = None
    for ckpt_dir in checkpoint_dirs:
        sidecar_path = Path(ckpt_dir) / "training_config.json"
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                sidecar = json.load(f)
            model_path = sidecar.get("model_name_or_path", "meta-llama/Llama-3.2-3B")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            break

    if tokenizer is None:
        logger.error("Could not load tokenizer from any checkpoint sidecar.")
        sys.exit(1)

    tokenized = get_tokenized_dataset(dataset_dict, tokenizer, max_length=128)

    evaluator = Evaluator(
        checkpoint_dirs=checkpoint_dirs,
        test_dataset=tokenized["test"],
        tokenizer=tokenizer,
        results_dir=args.results_dir,
    )

    df = evaluator.evaluate_all()
    logger.info(f"\n{df.to_string()}")

    evaluator.save_results(df)
    evaluator.generate_plots(df)


if __name__ == "__main__":
    main()
