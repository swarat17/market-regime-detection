"""
Data loading utilities for the PEFT Regime Benchmark.

Loads FinancialPhraseBank, applies regime labeling, and returns stratified splits.
"""

from pathlib import Path

from datasets import DatasetDict, load_dataset as hf_load_dataset

from src.data.preprocessor import LABEL2ID, ID2LABEL, preprocess_dataset


def load_dataset(
    split_ratios: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
    balance_classes: bool = True,
    agreement: str = "sentences_50agree",
) -> DatasetDict:
    """
    Load FinancialPhraseBank, apply regime labeling, and return stratified splits.

    Args:
        split_ratios: (train, val, test) fractions — must sum to 1.0.
        seed: Random seed for reproducible splits.
        balance_classes: Undersample Volatile to reduce class imbalance.
        agreement: FPB agreement level ('sentences_50agree', 'sentences_66agree',
                   'sentences_75agree', 'sentences_allagree').

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits.
        Each example has 'text' (str) and 'label' (int: 0=Bull, 1=Bear, 2=Volatile).
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"

    train_ratio, val_ratio, _ = split_ratios

    # Load raw FPB — it only has a 'train' split
    # trust_remote_code required: FPB uses a custom dataset script on HF Hub
    raw = hf_load_dataset("takala/financial_phrasebank", agreement, trust_remote_code=True)
    raw_train = raw["train"]

    # Preprocess (regime mapping + cleaning + balancing)
    processed = preprocess_dataset(raw_train, balance_classes=balance_classes)

    # Stratified train/val/test split
    # First split off test
    train_val_test = processed.train_test_split(
        test_size=(1.0 - train_ratio),
        seed=seed,
        stratify_by_column="label",
    )
    # Then split val from remaining
    val_fraction = val_ratio / (1.0 - train_ratio)
    val_test = train_val_test["test"].train_test_split(
        test_size=(1.0 - val_fraction),
        seed=seed,
        stratify_by_column="label",
    )

    dataset_dict = DatasetDict(
        {
            "train": train_val_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    # Attach label metadata
    for split in dataset_dict.values():
        split._info.features["label"]  # access to verify exists

    return dataset_dict


def get_class_distribution(dataset_split) -> dict:
    """
    Compute class distribution for a dataset split.

    Returns:
        dict with keys 'bull', 'bear', 'volatile' and percentage values summing to ~100.
    """
    labels = dataset_split["label"]
    total = len(labels)
    if total == 0:
        return {"bull": 0.0, "bear": 0.0, "volatile": 0.0}

    counts = {0: 0, 1: 0, 2: 0}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return {
        "bull": round(counts[0] / total * 100, 2),
        "bear": round(counts[1] / total * 100, 2),
        "volatile": round(counts[2] / total * 100, 2),
    }


def get_tokenized_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in the dataset.

    Args:
        dataset_dict: DatasetDict with 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer (already configured with pad_token).
        max_length: Maximum sequence length.

    Returns:
        DatasetDict with tokenized inputs (input_ids, attention_mask, label).
    """

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset_dict.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    tokenized = tokenized.with_format("torch")
    return tokenized
