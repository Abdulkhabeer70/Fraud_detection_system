"""
preprocess.py - Data Loading, Scaling, SMOTE, and Train/Test Split
==================================================================

This module handles the full data preprocessing pipeline for the
Credit Card Fraud Detection project:
  1. Load the raw CSV dataset
  2. Explore basic statistics and class distribution
  3. Scale 'Time' and 'Amount' features (V1-V28 are already PCA-transformed)
  4. Split into train/test sets (stratified)
  5. Apply SMOTE oversampling on the training set to handle class imbalance

The dataset columns:
  - Time:   Seconds elapsed since the first transaction
  - V1-V28: PCA-transformed features (anonymized)
  - Amount: Transaction amount
  - Class:  0 = legitimate, 1 = fraud
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "ml", "figures")


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================================
# 1. DATA LOADING
# ===================================================================
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the credit card transaction dataset from CSV.

    Parameters
    ----------
    filepath : str
        Path to the creditcard.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place creditcard.csv in the data/ directory."
        )
    print(f"[preprocess] Loading dataset from {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"[preprocess] Dataset shape: {df.shape}")
    return df


# ===================================================================
# 2. EXPLORATORY STATISTICS
# ===================================================================
def explore_data(df: pd.DataFrame) -> dict:
    """
    Print and return basic statistics about the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    dict
        Dictionary containing summary statistics.
    """
    total = len(df)
    fraud_count = int(df["Class"].sum())
    legit_count = total - fraud_count
    fraud_pct = fraud_count / total * 100

    stats = {
        "total_transactions": total,
        "legitimate_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "feature_count": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "amount_mean": round(float(df["Amount"].mean()), 2),
        "amount_std": round(float(df["Amount"].std()), 2),
        "amount_max": round(float(df["Amount"].max()), 2),
        "time_span_hours": round(float(df["Time"].max()) / 3600, 2),
    }

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>25s}: {v}")
    print("=" * 60 + "\n")

    return stats


# ===================================================================
# 3. FEATURE SCALING
# ===================================================================
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the 'Time' and 'Amount' columns using RobustScaler
    (robust to outliers). V1-V28 are already PCA-transformed and
    roughly standardized, so we leave them as-is.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with raw Time and Amount columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with scaled Time and Amount columns.
    """
    print("[preprocess] Scaling 'Time' and 'Amount' with RobustScaler ...")
    df = df.copy()

    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()

    df["Amount"] = scaler_amount.fit_transform(df[["Amount"]])
    df["Time"] = scaler_time.fit_transform(df[["Time"]])

    # Save scalers for later use / inference
    joblib.dump(scaler_amount, os.path.join(MODELS_DIR, "scaler_amount.joblib"))
    joblib.dump(scaler_time, os.path.join(MODELS_DIR, "scaler_time.joblib"))
    print("[preprocess] Scalers saved to ml/models/")

    return df


# ===================================================================
# 4. TRAIN / TEST SPLIT
# ===================================================================
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified train/test split preserving the class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled dataframe.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    print(f"[preprocess] Splitting data (test_size={test_size}) ...")

    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    feature_names = [c for c in df.columns if c != "Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"  Train set : {X_train.shape[0]:>7,} samples  "
          f"(fraud: {int(y_train.sum()):,})")
    print(f"  Test set  : {X_test.shape[0]:>7,} samples  "
          f"(fraud: {int(y_test.sum()):,})")

    return X_train, X_test, y_train, y_test, feature_names


# ===================================================================
# 5. SMOTE OVERSAMPLING
# ===================================================================
def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    sampling_strategy: float = 0.5,
):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to the
    training set to mitigate class imbalance.

    Using sampling_strategy=0.5 means the minority class will be
    oversampled to 50% of the majority class count (not full 1:1),
    which provides a good balance between information gain and
    overfitting risk.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    random_state : int
        Random seed.
    sampling_strategy : float
        Target ratio of minority to majority class after resampling.

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    print(f"\n[preprocess] Applying SMOTE (strategy={sampling_strategy}) ...")
    print(f"  Before SMOTE - Class 0: {int((y_train == 0).sum()):,}, "
          f"Class 1: {int((y_train == 1).sum()):,}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE  - Class 0: {int((y_resampled == 0).sum()):,}, "
          f"Class 1: {int((y_resampled == 1).sum()):,}")
    print(f"  Total training samples: {len(y_resampled):,}\n")

    return X_resampled, y_resampled


# ===================================================================
# FULL PREPROCESSING PIPELINE
# ===================================================================
def run_preprocessing(data_path: str = DATA_PATH):
    """
    Execute the full preprocessing pipeline end-to-end.

    Returns
    -------
    dict
        Dictionary with all preprocessed data and metadata:
        {
            "X_train", "X_test", "y_train", "y_test",
            "X_train_smote", "y_train_smote",
            "feature_names", "stats", "df_scaled"
        }
    """
    ensure_dirs()

    # Step 1 – Load
    df = load_data(data_path)

    # Step 2 – Explore
    stats = explore_data(df)

    # Step 3 – Scale
    df_scaled = scale_features(df)

    # Step 4 – Split
    X_train, X_test, y_train, y_test, feature_names = split_data(df_scaled)

    # Step 5 – SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Save the preprocessed splits for reproducibility
    preprocessed = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "feature_names": feature_names,
        "stats": stats,
        "df_scaled": df_scaled,
    }

    # Save stats to JSON
    stats_path = os.path.join(MODELS_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[preprocess] Dataset stats saved to {stats_path}")

    # Save preprocessed arrays
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_smote": X_train_smote,
            "y_train_smote": y_train_smote,
            "feature_names": feature_names,
        },
        os.path.join(MODELS_DIR, "preprocessed_data.joblib"),
    )
    print("[preprocess] Preprocessed data saved to ml/models/preprocessed_data.joblib")

    return preprocessed


# ===================================================================
# STANDALONE EXECUTION
# ===================================================================
if __name__ == "__main__":
    try:
        result = run_preprocessing()
        print("\n[preprocess] ✓ Preprocessing pipeline completed successfully!")
    except FileNotFoundError as e:
        print(f"\n[preprocess] ✗ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[preprocess] ✗ Unexpected error: {e}")
        raise
