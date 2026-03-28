"""
PEFT model factory for the PEFT Regime Benchmark.

Supports: LoRA, QLoRA, Prefix Tuning, (IA)³
"""

from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from src.utils.logger import logger


def create_peft_model(base_model, config: dict) -> dict:
    """
    Wrap a base model with a PEFT adapter based on config.

    Args:
        base_model: A loaded HuggingFace model (from base_loader.load_base_model).
        config: Dict with at minimum a 'method' key. Method-specific keys:
                LoRA/QLoRA: lora_r, lora_alpha, lora_dropout, target_modules
                QLoRA validation: base model must have is_loaded_in_4bit=True
                Prefix Tuning: num_virtual_tokens, prefix_projection
                (IA)³: no extra keys needed

    Returns:
        dict with keys:
            - model: PEFT-wrapped model
            - trainable_params: int
            - total_params: int
            - trainable_pct: float

    Raises:
        ValueError: If method is 'qlora' and base model is not 4-bit quantized.
        ValueError: If method is unknown.
    """
    method = config.get("method", "").lower()

    # For quantized models, prepare for k-bit training BEFORE applying PEFT
    is_quantized = getattr(base_model, "is_quantized", False) or getattr(
        base_model, "is_loaded_in_4bit", False
    )

    if method == "qlora":
        if not is_quantized:
            raise ValueError(
                "QLoRA requires a 4-bit quantized base model. "
                "Load the base model with quantize=True before applying QLoRA."
            )

    if is_quantized:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )

    peft_config = _build_peft_config(method, config)
    peft_model = get_peft_model(base_model, peft_config)

    # Prefix tuning compatibility fix for transformers >= 4.44:
    # PEFT passes past_key_values as a legacy tuple, but Llama 4.44+ expects a
    # DynamicCache object with .get_seq_length(). Patch the inner LlamaModel
    # forward to convert tuples on the fly.
    if method == "prefix_tuning":
        _patch_llama_legacy_cache(peft_model)

    trainable_params, total_params = _count_parameters(peft_model)
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    logger.info(
        f"PEFT model created | method={method} | "
        f"trainable={trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)"
    )

    # Log to W&B if a run is active
    try:
        from src.utils.logger import log_metrics

        log_metrics(
            {
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_pct": trainable_pct,
            }
        )
    except Exception:
        pass

    return {
        "model": peft_model,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
    }


def _build_peft_config(method: str, config: dict):
    """Build the appropriate PeftConfig for the given method."""
    if method in ("lora", "qlora"):
        return LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            task_type=TaskType.SEQ_CLS,
            bias="none",
        )

    elif method == "prefix_tuning":
        # Note: PrefixTuningConfig may require CAUSAL_LM for some model architectures.
        # If SEQ_CLS raises errors, fall back to CAUSAL_LM (Llama is a causal model).
        try:
            return PrefixTuningConfig(
                num_virtual_tokens=config.get("num_virtual_tokens", 10),
                prefix_projection=config.get("prefix_projection", False),
                task_type=TaskType.SEQ_CLS,
            )
        except Exception:
            return PrefixTuningConfig(
                num_virtual_tokens=config.get("num_virtual_tokens", 10),
                prefix_projection=config.get("prefix_projection", False),
                task_type=TaskType.CAUSAL_LM,
            )

    elif method == "ia3":
        return IA3Config(
            task_type=TaskType.SEQ_CLS,
        )

    else:
        raise ValueError(
            f"Unknown PEFT method: '{method}'. "
            "Supported: 'lora', 'qlora', 'prefix_tuning', 'ia3'."
        )


def _patch_llama_legacy_cache(peft_model) -> None:
    """
    Monkey-patch the inner LlamaModel.forward to convert legacy tuple
    past_key_values to DynamicCache.

    Root cause: PEFT 0.12 prefix tuning passes past_key_values as a tuple,
    but transformers 4.44+ Llama expects a Cache object with .get_seq_length().
    This patch converts on the fly without modifying transformers or PEFT source.
    """
    try:
        from transformers.cache_utils import DynamicCache

        # Navigate to the inner LlamaModel (base_model.model for SEQ_CLS)
        inner = None
        if hasattr(peft_model, "base_model"):
            bm = peft_model.base_model
            if hasattr(bm, "model") and hasattr(bm.model, "model"):
                inner = bm.model.model  # LlamaForSequenceClassification.model.model
            elif hasattr(bm, "model"):
                inner = bm.model

        if inner is None:
            logger.warning("Prefix tuning cache patch: could not locate inner LlamaModel — skipping.")
            return

        original_forward = inner.forward

        def patched_forward(*args, **kwargs):
            pkv = kwargs.get("past_key_values", None)
            if isinstance(pkv, tuple):
                kwargs["past_key_values"] = DynamicCache.from_legacy_cache(pkv)
            return original_forward(*args, **kwargs)

        inner.forward = patched_forward
        logger.info("Prefix tuning: applied legacy cache → DynamicCache patch for transformers 4.44+.")
    except Exception as e:
        logger.warning(f"Prefix tuning cache patch failed (will attempt training anyway): {e}")


def _count_parameters(model) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
