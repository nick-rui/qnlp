"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "toy_sentiment.csv"


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the toy sentiment dataset."""
    dataset = pd.read_csv(path)
    required = {"sentence", "label", "split", "pattern"}
    missing = required.difference(dataset.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

    labels = set(dataset["label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1 values, found: {sorted(labels)}")

    splits = set(dataset["split"].unique())
    if not splits.issubset({"train", "test"}):
        raise ValueError(f"Splits must be train/test values, found: {sorted(splits)}")

    return dataset


def train_test_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic train/test DataFrames from the split column."""
    train = dataset[dataset["split"] == "train"].reset_index(drop=True)
    test = dataset[dataset["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Dataset must contain both train and test rows.")
    return train, test

