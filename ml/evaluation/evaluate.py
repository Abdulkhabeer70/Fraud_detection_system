"""
evaluate.py - Model Evaluation & Visualization
================================================

Comprehensive evaluation of all trained models:
  - Classification metrics (precision, recall, F1, accuracy)
  - Confusion matrices (with visualization)
  - ROC curves (with AUC comparison)
  - Precision-Recall curves (critical for imbalanced data)
  - Threshold analysis

Results are saved as:
  - ml/models/evaluation_results.json
  - ml/figures/confusion_matrices.png
  - ml/figures/roc_curves.png
  - ml/figures/precision_recall_curves.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)
import joblib
import warnings

warnings.filterwarnings("ignore")

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
# METRIC COMPUTATION
# ===================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_proba : np.ndarray, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary of metric name -> value.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
    }

    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["avg_precision"] = round(average_precision_score(y_true, y_proba), 4)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)

    # Specificity (true negative rate)
    metrics["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0

    return metrics


# ===================================================================
# CONFUSION MATRIX VISUALIZATION
# ===================================================================

def plot_confusion_matrices(
    model_results: dict,
    y_test: np.ndarray,
    save_path: str = None,
):
    """
    Plot confusion matrices for all models side by side.

    Parameters
    ----------
    model_results : dict
        Mapping of model_name -> {"y_pred": ..., "metrics": ...}.
    y_test : np.ndarray
        True labels.
    save_path : str
        Path to save the figure.
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for i, (name, res) in enumerate(model_results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        ax = axes[i]

        # Normalize for display
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Display values
        for row in range(2):
            for col in range(2):
                color = "white" if cm_norm[row, col] > 0.5 else "black"
                ax.text(col, row, f"{cm[row, col]:,}\n({cm_norm[row, col]:.1%})",
                        ha="center", va="center", fontsize=11, color=color,
                        fontweight="bold")

        # Title with F1 score
        f1 = res["metrics"]["f1_score"]
        ax.set_title(f"{name}\n(F1={f1:.4f})", fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.close()


# ===================================================================
# ROC CURVE VISUALIZATION
# ===================================================================

def plot_roc_curves(
    model_results: dict,
    y_test: np.ndarray,
    save_path: str = None,
):
    """
    Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    model_results : dict
        Model name -> {"y_proba": ..., "metrics": ...}.
    y_test : np.ndarray
        True labels.
    save_path : str
        Path to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    for i, (name, res) in enumerate(model_results.items()):
        if res.get("y_proba") is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            auc = res["metrics"].get("roc_auc", 0)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label=f"{name} (AUC={auc:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.close()


# ===================================================================
# PRECISION-RECALL CURVE
# ===================================================================

def plot_precision_recall_curves(
    model_results: dict,
    y_test: np.ndarray,
    save_path: str = None,
):
    """
    Plot Precision-Recall curves — more informative than ROC for
    highly imbalanced datasets.

    Parameters
    ----------
    model_results : dict
        Model results with y_proba.
    y_test : np.ndarray
        True labels.
    save_path : str
        Path to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    for i, (name, res) in enumerate(model_results.items()):
        if res.get("y_proba") is not None:
            prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
            ap = res["metrics"].get("avg_precision", 0)
            ax.plot(rec, prec, color=colors[i % len(colors)], linewidth=2,
                    label=f"{name} (AP={ap:.4f})")

    # Baseline: prevalence line
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence={prevalence:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.close()


# ===================================================================
# MAIN EVALUATION PIPELINE
# ===================================================================

def evaluate_all_models(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Load all trained models and evaluate them comprehensively.

    Parameters
    ----------
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    dict
        Evaluation results for all models.
    """
    ensure_dirs()

    model_files = {
        "logistic_regression": "logistic_regression.joblib",
        "random_forest": "random_forest.joblib",
        "xgboost": "xgboost.joblib",
    }

    model_results = {}

    print("\n" + "=" * 70)
    print("  MODEL EVALUATION")
    print("=" * 70)

    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(model_path):
            print(f"\n  [SKIP] {name}: model file not found at {model_path}")
            continue

        print(f"\n  Evaluating: {name}")
        print(f"  {'-'*50}")

        model = joblib.load(model_path)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # Print report
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
        print(f"  MCC: {metrics['mcc']}")
        if y_proba is not None:
            print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
            print(f"  Avg Precision: {metrics.get('avg_precision', 'N/A')}")

        model_results[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": metrics,
        }

    if not model_results:
        print("\n  [WARNING] No models found to evaluate!")
        return {}

    # --- Visualizations ---
    print("\n  Generating evaluation visualizations ...")

    plot_confusion_matrices(
        model_results, y_test,
        save_path=os.path.join(FIGURES_DIR, "confusion_matrices.png"),
    )
    plot_roc_curves(
        model_results, y_test,
        save_path=os.path.join(FIGURES_DIR, "roc_curves.png"),
    )
    plot_precision_recall_curves(
        model_results, y_test,
        save_path=os.path.join(FIGURES_DIR, "precision_recall_curves.png"),
    )

    # --- Summary Table ---
    print("\n" + "=" * 90)
    print("  EVALUATION SUMMARY")
    print("=" * 90)
    header = (f"  {'Model':<25s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} "
              f"{'F1':>7s} {'MCC':>7s} {'AUC':>7s} {'AP':>7s}")
    print(header)
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, res in model_results.items():
        m = res["metrics"]
        print(
            f"  {name:<25s} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {m['mcc']:>7.4f} "
            f"{m.get('roc_auc', 0):>7.4f} {m.get('avg_precision', 0):>7.4f}"
        )
    print("=" * 90)

    # --- Save results JSON ---
    # Remove numpy arrays before JSON serialization
    json_results = {}
    for name, res in model_results.items():
        json_results[name] = res["metrics"]

    results_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return model_results


# ===================================================================
# STANDALONE EXECUTION
# ===================================================================

if __name__ == "__main__":
    preprocessed_path = os.path.join(MODELS_DIR, "preprocessed_data.joblib")

    if not os.path.isfile(preprocessed_path):
        print("[evaluate] Preprocessed data not found. Running preprocessing ...")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
        from preprocessing.preprocess import run_preprocessing
        try:
            data = run_preprocessing()
        except FileNotFoundError as e:
            print(f"[evaluate] ✗ {e}")
            sys.exit(1)
    else:
        data = joblib.load(preprocessed_path)

    results = evaluate_all_models(
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    if results:
        print("\n[evaluate] ✓ Evaluation completed successfully!")
    else:
        print("\n[evaluate] ✗ No models were evaluated. Train models first.")
