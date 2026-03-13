"""
Base model loading for the PEFT Regime Benchmark.

Supports Llama-3.2-3B and Mistral-7B with optional 4-bit quantization.
Always uses fp16 (GTX 1650 / Turing architecture — no bfloat16 support).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from src.data.preprocessor import ID2LABEL, LABEL2ID
from src.utils.logger import logger

MODEL_REGISTRY = {
    "llama3": "meta-llama/Llama-3.2-3B",
    "mistral7b": "mistralai/Mistral-7B-v0.3",
}

MISTRAL_VRAM_WARNING = (
    "Mistral-7B requires >8GB VRAM. This run may OOM on GTX 1650 (4GB). "
    "Defaulting to llama3 is recommended."
)


def get_bnb_config(quant_type: str = "fp4", use_double_quant: bool = False) -> BitsAndBytesConfig:
    """
    Build a BitsAndBytesConfig for 4-bit quantization.

    Args:
        quant_type: 'fp4' (for LoRA) or 'nf4' (for QLoRA per Dettmers et al.).
        use_double_quant: True for QLoRA double quantization.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 only — Turing arch
    )


def load_base_model(
    model_name: str,
    quantize: bool = False,
    quant_type: str = "fp4",
    use_double_quant: bool = False,
    num_labels: int = 3,
) -> tuple:
    """
    Load a base model and tokenizer for sequence classification.

    Args:
        model_name: Key from MODEL_REGISTRY ('llama3' or 'mistral7b') or full HF path.
        quantize: If True, load in 4-bit via bitsandbytes.
        quant_type: '4fp' or 'nf4' — only used when quantize=True.
        use_double_quant: Only used when quantize=True.
        num_labels: Number of classification labels (3 for Bull/Bear/Volatile).

    Returns:
        (model, tokenizer) tuple.
        model.is_quantized is set to `quantize` for downstream validation.
    """
    if model_name in MODEL_REGISTRY:
        if model_name == "mistral7b":
            logger.warning(MISTRAL_VRAM_WARNING)
        model_path = MODEL_REGISTRY[model_name]
    else:
        model_path = model_name  # allow passing full HF path or local path

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Llama (and some other models) have no pad token — set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build model kwargs
    model_kwargs = {
        "num_labels": num_labels,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "torch_dtype": torch.float16,  # Hard-coded: no bfloat16 on Turing
        "device_map": "auto",
        "ignore_mismatched_sizes": True,
        "low_cpu_mem_usage": True,  # Stream shards one at a time — reduces peak RAM
    }

    if quantize:
        bnb_config = get_bnb_config(quant_type=quant_type, use_double_quant=use_double_quant)
        model_kwargs["quantization_config"] = bnb_config

    logger.info(f"Loading model from {model_path} (quantize={quantize}, quant_type={quant_type})")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)

    # Needed for Llama — set pad token id on model config too
    model.config.pad_token_id = tokenizer.pad_token_id

    # Mark quantization state for downstream validation in peft_factory
    model.is_quantized = quantize

    logger.info(
        f"Model loaded: {model_path} | quantized={quantize} | "
        f"dtype={model.dtype}"
    )

    return model, tokenizer
