"""Inspect a grasp CSV and separate input features from binary labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["x", "y", "z", "roll", "pitch", "yaw"]
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent / "output" / "pr2_cube_demo" / "grasps.csv"
)


def inspect_dataset(data_path):
    """Validate the training table, display a summary, and return X and y."""
    df = pd.read_csv(data_path)
    expected_columns = FEATURE_COLUMNS + ["success"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected columns in this order: {expected_columns}")
    if df.empty:
        raise ValueError("The dataset contains no completed trials.")
    if df.isna().any().any():
        raise ValueError("The dataset contains missing values.")
    if not all(pd.api.types.is_numeric_dtype(df[column]) for column in df.columns):
        raise ValueError("All features and labels must be numeric.")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("The dataset contains non-finite values.")
    if not df["success"].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1.")

    X = df[FEATURE_COLUMNS]
    y = df["success"]

    print(f"Dataset: {Path(data_path).resolve()}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print(f"\nFeature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    print("\nClass counts (0 = failure, 1 = success):")
    print(counts.to_string())
    print("\nValidation passed: numeric finite values, no missing data, binary labels.")
    if (counts == 0).any():
        print("Only one class is present. Collect both outcomes before training.")
    elif counts.loc[0] != counts.loc[1]:
        print("Classes are imbalanced. Retain raw records and plan balancing separately.")
    print("No model has been trained and no files have been changed.")
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    inspect_dataset(args.csv_path)
