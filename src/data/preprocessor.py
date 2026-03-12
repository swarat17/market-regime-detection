"""
Regime labeling and text preprocessing for financial text classification.

FinancialPhraseBank labels: 0=negative, 1=neutral, 2=positive
Regime mapping: positive→Bull(0), negative→Bear(1), neutral→Volatile(2)
"""

from collections import Counter

from datasets import Dataset

# Label mappings
SENTIMENT_TO_REGIME = {2: 0, 0: 1, 1: 2}  # positive→Bull, negative→Bear, neutral→Volatile
REGIME_NAMES = {0: "bull", 1: "bear", 2: "volatile"}
LABEL2ID = {"bull": 0, "bear": 1, "volatile": 2}
ID2LABEL = {0: "bull", 1: "bear", 2: "volatile"}


def map_sentiment_to_regime(example: dict) -> dict:
    """Map FPB sentiment label (0/1/2) to regime label (0=Bull, 1=Bear, 2=Volatile)."""
    return {"label": SENTIMENT_TO_REGIME[example["label"]]}


def clean_text(example: dict) -> dict:
    """Strip whitespace and normalize text field."""
    text = example.get("sentence", example.get("text", ""))
    if text is None:
        text = ""
    return {"text": text.strip()}


def preprocess_dataset(dataset: Dataset, balance_classes: bool = True) -> Dataset:
    """
    Apply regime mapping, text cleaning, and optional class balancing.

    Args:
        dataset: Raw FPB dataset split.
        balance_classes: If True, undersample Volatile to ~35% to reduce imbalance.
                         Original FPB is ~59% neutral → ~59% Volatile after mapping.

    Returns:
        Processed dataset with 'text' and 'label' columns.
    """
    # Rename 'sentence' to 'text' if needed
    if "sentence" in dataset.column_names:
        dataset = dataset.rename_column("sentence", "text")

    # Clean text
    dataset = dataset.map(clean_text)

    # Map sentiment → regime
    dataset = dataset.map(map_sentiment_to_regime)

    # Remove examples with empty text
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)

    # Keep only text and label columns
    cols_to_keep = ["text", "label"]
    cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    if balance_classes:
        dataset = _balance_classes(dataset)

    return dataset


def _balance_classes(dataset: Dataset) -> Dataset:
    """
    Undersample the majority class (Volatile) to achieve rough balance (~33% each).
    Target: each class gets roughly equal representation.
    """
    # Group indices by label
    label_to_indices = {0: [], 1: [], 2: []}
    for i, label in enumerate(dataset["label"]):
        label_to_indices[label].append(i)

    counts = {k: len(v) for k, v in label_to_indices.items()}
    # Target size: median count to avoid excessive undersampling
    target = sorted(counts.values())[1]  # median
    target = max(target, min(counts.values()))  # at least the minimum

    selected_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) > target:
            # Deterministic subsample (first target indices for reproducibility)
            selected_indices.extend(indices[:target])
        else:
            selected_indices.extend(indices)

    selected_indices = sorted(selected_indices)
    return dataset.select(selected_indices)
