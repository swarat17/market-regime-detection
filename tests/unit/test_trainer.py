"""
Phase 3 unit tests — Training Pipeline.

Tests instantiation and checkpoint saving without real GPU or large model downloads.
Uses distilgpt2 as a mock base model for fast iteration.
W&B is mocked to avoid real API calls.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_config(tmp_dir: str, max_steps: int = 10) -> dict:
    """Minimal config using distilgpt2 for fast unit tests."""
    return {
        "method": "lora",
        "base_model": "distilgpt2",
        "model_name_or_path": "distilgpt2",
        "dataset": "financial_phrasebank",
        "max_length": 32,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "num_epochs": 1,
        "output_dir": tmp_dir,
        "fp16": False,  # CPU test — no CUDA
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "optim": "adamw_torch",
        "seed": 42,
        "quantize": False,
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["c_attn"],
        "max_steps": max_steps,
    }


def _make_tiny_dataset(tokenizer, n: int = 20):
    """Create a tiny tokenized dataset for unit tests."""
    texts = ["This is a bull market example."] * (n // 2) + [
        "Bearish sentiment dominates."
    ] * (n - n // 2)
    labels = [0] * (n // 2) + [1] * (n - n // 2)

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=32,
        padding="max_length",
        return_tensors="pt",
    )
    data = {
        "input_ids": enc["input_ids"].tolist(),
        "attention_mask": enc["attention_mask"].tolist(),
        "label": labels,
    }
    ds = Dataset.from_dict(data)
    ds.set_format("torch")
    return ds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("wandb.init")
@patch("wandb.log")
@patch("wandb.finish")
def test_trainer_instantiates_from_config(mock_finish, mock_log, mock_init):
    """RegimeTrainer(config) sets correct method without needing GPU."""
    from src.training.trainer import RegimeTrainer

    with tempfile.TemporaryDirectory() as tmp:
        config = _make_tiny_config(tmp)
        trainer = RegimeTrainer(config)
        assert trainer.method == "lora"
        assert trainer.config["base_model"] == "distilgpt2"


@patch("wandb.init")
@patch("wandb.log")
@patch("wandb.finish")
def test_checkpoint_saved_after_short_run(mock_finish, mock_log, mock_init):
    """10-step mini-run creates a checkpoint directory with saved files."""
    from src.training.trainer import RegimeTrainer
    from src.models.base_loader import load_base_model

    with tempfile.TemporaryDirectory() as tmp:
        config = _make_tiny_config(tmp, max_steps=10)

        # Load tiny model directly to avoid calling load_base_model with Llama
        from transformers import AutoModelForSequenceClassification
        from src.data.preprocessor import ID2LABEL, LABEL2ID

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilgpt2",
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.is_quantized = False

        trainer = RegimeTrainer(config)
        trainer.tokenizer = tokenizer

        # Apply PEFT manually
        from src.models.peft_factory import create_peft_model

        peft_info = create_peft_model(model, config)
        trainer.model = peft_info["model"]
        trainer.peft_info = peft_info

        train_ds = _make_tiny_dataset(tokenizer, n=20)
        eval_ds = _make_tiny_dataset(tokenizer, n=6)

        checkpoint_dir = trainer.train(train_ds, eval_ds)

        assert Path(checkpoint_dir).exists(), f"Checkpoint dir not found: {checkpoint_dir}"


@patch("wandb.init")
@patch("wandb.log")
@patch("wandb.finish")
def test_sidecar_json_contains_method(mock_finish, mock_log, mock_init):
    """JSON sidecar saved alongside checkpoint has 'method' key matching config."""
    from src.training.trainer import RegimeTrainer
    from transformers import AutoModelForSequenceClassification
    from src.data.preprocessor import ID2LABEL, LABEL2ID
    from src.models.peft_factory import create_peft_model

    with tempfile.TemporaryDirectory() as tmp:
        config = _make_tiny_config(tmp, max_steps=5)

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilgpt2",
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.is_quantized = False

        trainer = RegimeTrainer(config)
        trainer.tokenizer = tokenizer

        peft_info = create_peft_model(model, config)
        trainer.model = peft_info["model"]
        trainer.peft_info = peft_info

        train_ds = _make_tiny_dataset(tokenizer, n=20)
        eval_ds = _make_tiny_dataset(tokenizer, n=6)

        checkpoint_dir = trainer.train(train_ds, eval_ds)

        sidecar_path = Path(checkpoint_dir) / "training_config.json"
        assert sidecar_path.exists(), f"Sidecar JSON not found at {sidecar_path}"

        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert "method" in sidecar, "Sidecar JSON missing 'method' key"
        assert sidecar["method"] == config["method"]
