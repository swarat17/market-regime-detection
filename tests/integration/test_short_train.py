"""
Phase 3 integration tests — require GPU and model downloads.

Run with: pytest tests/integration/ -v -m integration
"""

import pytest


@pytest.mark.integration
def test_lora_trains_50_steps_no_crash():
    """LoRA + Llama-3.2-3B runs 50 steps without exception."""
    from src.training.trainer import RegimeTrainer
    from src.data.loader import get_tokenized_dataset, load_dataset

    config = {
        "method": "lora",
        "base_model": "llama3",
        "model_name_or_path": "meta-llama/Llama-3.2-3B",
        "dataset": "financial_phrasebank",
        "max_length": 128,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "output_dir": "models/integration_test",
        "fp16": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "optim": "paged_adamw_8bit",
        "seed": 42,
        "quantize": True,
        "bnb_4bit_quant_type": "fp4",
        "bnb_4bit_use_double_quant": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "max_steps": 50,
    }

    trainer = RegimeTrainer(config)
    trainer.setup()

    dataset_dict = load_dataset(seed=42)
    tokenized = get_tokenized_dataset(
        dataset_dict, trainer.tokenizer, max_length=128
    )

    checkpoint_dir = trainer.train(
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )
    assert checkpoint_dir is not None
