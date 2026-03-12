"""
Unified training pipeline for the PEFT Regime Benchmark.

Reads a config (dict or YAML path), loads model + PEFT adapter,
trains with HuggingFace Trainer, saves checkpoint + JSON sidecar.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data.loader import get_tokenized_dataset, load_dataset
from src.models.base_loader import load_base_model
from src.models.peft_factory import create_peft_model
from src.training.callbacks import RegimeBenchmarkCallback
from src.utils.logger import finish_run, init_wandb, logger


class RegimeTrainer:
    """
    Orchestrates loading, training, and checkpointing for a PEFT experiment.

    Usage:
        trainer = RegimeTrainer(config)
        trainer.setup()
        checkpoint_dir = trainer.train(train_ds, eval_ds)
    """

    def __init__(self, config: dict):
        self.config = config
        self.method = config["method"]
        self.base_model_name = config.get("base_model", "llama3")
        self.model = None
        self.tokenizer = None
        self.peft_info = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RegimeTrainer":
        """Instantiate RegimeTrainer from a YAML config file."""
        path = Path(yaml_path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def setup(self):
        """Load base model, apply PEFT adapter."""
        logger.info(f"Setting up trainer | method={self.method} | base={self.base_model_name}")

        self.model, self.tokenizer = load_base_model(
            model_name=self.config.get("model_name_or_path", self.base_model_name),
            quantize=self.config.get("quantize", True),
            quant_type=self.config.get("bnb_4bit_quant_type", "fp4"),
            use_double_quant=self.config.get("bnb_4bit_use_double_quant", False),
        )

        self.peft_info = create_peft_model(self.model, self.config)
        self.model = self.peft_info["model"]

    def train(self, train_dataset, eval_dataset) -> str:
        """
        Run training and save checkpoint.

        Args:
            train_dataset: Tokenized training dataset.
            eval_dataset: Tokenized validation dataset.

        Returns:
            Path to saved checkpoint directory.
        """
        if self.model is None:
            raise RuntimeError("Call setup() before train()")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(
            self.config.get("output_dir", f"models/{self.method}_{self.base_model_name}")
        ) / f"{self.method}_{self.base_model_name}_{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Init W&B
        run_name = f"{self.method}_{self.base_model_name}_{timestamp}"
        try:
            init_wandb(
                project="peft-regime-benchmark",
                config=self.config,
                run_name=run_name,
            )
        except Exception as e:
            logger.warning(f"W&B init failed (continuing without): {e}")

        training_args = self._build_training_args(checkpoint_dir)

        callback = RegimeBenchmarkCallback(
            trainable_params=self.peft_info.get("trainable_params", 0)
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, pad_to_multiple_of=8
        )

        hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[callback],
        )

        logger.info(f"Starting training | steps_per_epoch={len(train_dataset)}")
        hf_trainer.train()

        # Save best model
        hf_trainer.save_model(str(checkpoint_dir))

        # Write JSON sidecar
        sidecar = {
            "method": self.method,
            "base_model": self.base_model_name,
            "model_name_or_path": self.config.get("model_name_or_path", ""),
            "trainable_params": self.peft_info.get("trainable_params", 0),
            "total_params": self.peft_info.get("total_params", 0),
            "trainable_pct": self.peft_info.get("trainable_pct", 0.0),
            "lora_r": self.config.get("lora_r", None),
            "training_completed": timestamp,
            "best_val_f1": self._get_best_val_f1(hf_trainer),
            "checkpoint_dir": str(checkpoint_dir),
        }
        sidecar_path = checkpoint_dir / "training_config.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        finish_run()
        return str(checkpoint_dir)

    def _build_training_args(self, output_dir: Path) -> TrainingArguments:
        """Build HuggingFace TrainingArguments from config."""
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 1),
            per_device_eval_batch_size=self.config.get("batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 16),
            learning_rate=self.config.get("learning_rate", 3e-4),
            fp16=self.config.get("fp16", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="wandb",
            dataloader_num_workers=0,  # Windows: no multiprocessing fork
            optim=self.config.get("optim", "paged_adamw_8bit"),
            seed=self.config.get("seed", 42),
            max_steps=self.config.get("max_steps", -1),  # -1 = use num_epochs
        )

    def _compute_metrics(self, eval_pred) -> dict:
        """Compute accuracy and macro F1 for HuggingFace Trainer."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"eval_accuracy": acc, "eval_f1_macro": f1}

    def _get_best_val_f1(self, hf_trainer) -> float:
        """Extract best val F1 from trainer state."""
        try:
            if hf_trainer.state and hf_trainer.state.best_metric:
                return float(hf_trainer.state.best_metric)
        except Exception:
            pass
        return 0.0
