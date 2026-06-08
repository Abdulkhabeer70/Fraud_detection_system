"""
train_models.py - Supervised Model Training
============================================

Train three supervised classifiers for credit card fraud detection:
  1. Logistic Regression  (baseline, interpretable)
  2. Random Forest         (ensemble, handles non-linearity)
  3. XGBoost               (gradient boosting, state-of-the-art)

Each model is trained on the SMOTE-resampled training data and
evaluated on the original (imbalanced) test set.

Models are saved as .joblib files in ml/models/.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import joblib
import warnings

warnings.filterwarnings("ignore")

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[train_models] WARNING: xgboost not installed. Skipping XGBoost.")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "ml", "figures")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================================
# MODEL DEFINITIONS
# ===================================================================

def get_models() -> dict:
    """
    Return a dictionary of model name -> (model_instance, description).
    """
    models = {
        "logistic_regression": (
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "L2-regularized logistic regression with balanced class weights",
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "Random Forest with 200 trees, balanced subsample weighting",
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = (
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=50,  # approximate ratio of neg/pos
                use_label_encoder=False,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            "XGBoost with tuned hyperparameters and positive-class weighting",
        )

    return models


# ===================================================================
# TRAINING
# ===================================================================

def train_single_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train a single model and return performance metrics.

    Parameters
    ----------
    name : str
        Model name (for logging/saving).
    model : sklearn estimator
        Unfitted model instance.
    X_train, y_train : np.ndarray
        Training data (typically SMOTE-resampled).
    X_test, y_test : np.ndarray
        Test data (original, imbalanced).

    Returns
    -------
    dict
        Training results including metrics, training time, and model path.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)
    print(f"  Training time: {train_time}s")

    # Predict
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    # Metrics
    f1 = round(f1_score(y_test, y_pred), 4)
    roc_auc = round(roc_auc_score(y_test, y_proba), 4)
    avg_precision = round(average_precision_score(y_test, y_proba), 4)

    print(f"  F1 Score         : {f1}")
    print(f"  ROC-AUC          : {roc_auc}")
    print(f"  Avg Precision    : {avg_precision}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Cross-validation (on SMOTE data for consistency)
    print(f"  Running 5-fold stratified CV ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
    )
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std = round(float(cv_scores.std()), 4)
    print(f"  CV F1: {cv_mean} ± {cv_std}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")

    result = {
        "model_name": name,
        "train_time_seconds": train_time,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc,
        "test_avg_precision": avg_precision,
        "cv_f1_mean": cv_mean,
        "cv_f1_std": cv_std,
        "model_path": model_path,
    }

    return result


# ===================================================================
# MAIN TRAINING PIPELINE
# ===================================================================

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train all supervised models and return a summary of results.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        SMOTE-resampled training data.
    X_test, y_test : np.ndarray
        Original test data.

    Returns
    -------
    dict
        Mapping of model names to their result dictionaries.
    """
    ensure_dirs()
    models = get_models()
    all_results = {}

    for name, (model, desc) in models.items():
        print(f"\n  Description: {desc}")
        result = train_single_model(name, model, X_train, y_train, X_test, y_test)
        all_results[name] = result

    # Save summary
    summary_path = os.path.join(MODELS_DIR, "training_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[train_models] Training summary saved to {summary_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<25s} {'F1':>8s} {'ROC-AUC':>10s} {'Avg-PR':>10s} {'Time(s)':>10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for name, res in all_results.items():
        print(
            f"  {name:<25s} {res['test_f1_score']:>8.4f} "
            f"{res['test_roc_auc']:>10.4f} "
            f"{res['test_avg_precision']:>10.4f} "
            f"{res['train_time_seconds']:>10.2f}"
        )
    print("=" * 70)

    return all_results


# ===================================================================
# STANDALONE EXECUTION
# ===================================================================

if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_path = os.path.join(MODELS_DIR, "preprocessed_data.joblib")
    if not os.path.isfile(preprocessed_path):
        print("[train_models] Preprocessed data not found. Running preprocessing ...")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
        from preprocessing.preprocess import run_preprocessing
        data = run_preprocessing()
    else:
        print("[train_models] Loading preprocessed data ...")
        data = joblib.load(preprocessed_path)

    results = train_all_models(
        X_train=data["X_train_smote"],
        y_train=data["y_train_smote"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    print("\n[train_models] ✓ All models trained successfully!")
