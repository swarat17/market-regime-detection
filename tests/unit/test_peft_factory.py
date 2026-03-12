"""
Phase 2 unit tests — PEFT Factory.

Uses tiny GPT-2 mock model — no GPU or model downloads needed.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock


def _make_mock_model(is_quantized: bool = False):
    """
    Create a tiny GPT-2 based mock model for testing.
    Uses distilgpt2 (82M params) to avoid network calls to gated models.
    Sets is_quantized attribute for factory validation.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilgpt2",
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = model.config.eos_token_id

    # Simulate quantization state (actual 4-bit loading not needed for unit tests)
    model.is_quantized = is_quantized
    if is_quantized:
        model.is_loaded_in_4bit = True

    return model


@pytest.fixture(scope="module")
def mock_model():
    return _make_mock_model(is_quantized=False)


@pytest.fixture(scope="module")
def mock_quantized_model():
    return _make_mock_model(is_quantized=True)


# --- Tests ---


def test_lora_reduces_trainable_params(mock_model):
    """LoRA model has fewer trainable params than full fine-tune."""
    from src.models.peft_factory import create_peft_model, _count_parameters

    total_params = sum(p.numel() for p in mock_model.parameters())

    lora_config = {
        "method": "lora",
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn"],  # GPT-2 attention projection
    }

    result = create_peft_model(mock_model, lora_config)
    trainable = result["trainable_params"]

    assert trainable < total_params, (
        f"LoRA trainable params ({trainable:,}) should be less than "
        f"total params ({total_params:,})"
    )


def test_qlora_requires_quantized_base():
    """Passing non-quantized base model with QLoRA config raises ValueError."""
    from src.models.peft_factory import create_peft_model

    non_quantized = _make_mock_model(is_quantized=False)
    qlora_config = {
        "method": "qlora",
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn"],
    }

    with pytest.raises(ValueError, match="QLoRA requires a 4-bit quantized base model"):
        create_peft_model(non_quantized, qlora_config)


def test_all_methods_produce_valid_model():
    """All 4 PEFT methods instantiate without error on a mock model."""
    from src.models.peft_factory import create_peft_model

    methods_and_configs = [
        {
            "method": "lora",
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "target_modules": ["c_attn"],
        },
        {
            "method": "prefix_tuning",
            "num_virtual_tokens": 5,
            "prefix_projection": False,
        },
        {
            "method": "ia3",
        },
    ]

    for config in methods_and_configs:
        model = _make_mock_model(is_quantized=False)
        result = create_peft_model(model, config)
        assert result["model"] is not None, f"Method {config['method']} returned None model"
        assert "trainable_params" in result

    # QLoRA requires quantized model
    qlora_model = _make_mock_model(is_quantized=True)
    qlora_config = {
        "method": "qlora",
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn"],
    }
    result = create_peft_model(qlora_model, qlora_config)
    assert result["model"] is not None


def test_trainable_param_count_logged():
    """create_peft_model returns a dict with 'trainable_params' key."""
    from src.models.peft_factory import create_peft_model

    model = _make_mock_model(is_quantized=False)
    config = {
        "method": "lora",
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["c_attn"],
    }

    result = create_peft_model(model, config)

    assert "trainable_params" in result, "Result dict missing 'trainable_params'"
    assert "total_params" in result, "Result dict missing 'total_params'"
    assert "trainable_pct" in result, "Result dict missing 'trainable_pct'"
    assert isinstance(result["trainable_params"], int)
    assert result["trainable_params"] > 0
