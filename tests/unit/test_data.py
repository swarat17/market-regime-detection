"""
Phase 1 unit tests — Data Pipeline.

Tests run without GPU and without model downloads.
Uses real FinancialPhraseBank dataset (small, fast to download).
"""

import pytest

from src.data.loader import load_dataset, get_class_distribution
from src.data.preprocessor import preprocess_dataset, SENTIMENT_TO_REGIME


@pytest.fixture(scope="module")
def dataset():
    """Load the dataset once for all tests in this module."""
    return load_dataset(seed=42, balance_classes=True)


def test_regime_labels_are_three_classes(dataset):
    """Output dataset has exactly 3 unique labels."""
    all_labels = set()
    for split in ["train", "validation", "test"]:
        all_labels.update(set(dataset[split]["label"]))
    assert all_labels == {0, 1, 2}, f"Expected labels {{0, 1, 2}}, got {all_labels}"


def test_no_null_text_after_preprocessing(dataset):
    """No null or empty strings in any split."""
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        for i, text in enumerate(texts):
            assert text is not None, f"Found None text in {split}[{i}]"
            assert len(text.strip()) > 0, f"Found empty text in {split}[{i}]"


def test_stratified_split_preserves_distribution(dataset):
    """Train/val/test class distributions are within 5 percentage points of each other."""
    distributions = {}
    for split in ["train", "validation", "test"]:
        distributions[split] = get_class_distribution(dataset[split])

    for regime in ["bull", "bear", "volatile"]:
        pcts = [distributions[split][regime] for split in ["train", "validation", "test"]]
        max_diff = max(pcts) - min(pcts)
        assert max_diff <= 5.0, (
            f"Class '{regime}' distribution varies by {max_diff:.1f}pp across splits "
            f"(train={pcts[0]:.1f}%, val={pcts[1]:.1f}%, test={pcts[2]:.1f}%)"
        )


def test_get_class_distribution_sums_to_100(dataset):
    """Percentages in distribution dict sum to ~100."""
    for split in ["train", "validation", "test"]:
        dist = get_class_distribution(dataset[split])
        total = sum(dist.values())
        assert abs(total - 100.0) < 0.5, (
            f"Distribution for {split} sums to {total:.2f}, expected ~100"
        )
