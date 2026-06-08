"""
shap_analysis.py - SHAP Explainability Analysis
=================================================

Generate SHAP (SHapley Additive exPlanations) values for the trained
models to understand feature contributions to predictions.

Outputs:
  - SHAP summary plots (beeswarm / bar)
  - Feature importance rankings
  - Per-class feature importance
  - Saved as ml/figures/shap_*.png and ml/models/shap_results.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

# Try importing SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[shap_analysis] WARNING: shap not installed. Install with: pip install shap")

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
# SHAP ANALYSIS FOR A SINGLE MODEL
# ===================================================================

def analyze_model_shap(
    model,
    model_name: str,
    X_sample: np.ndarray,
    feature_names: list,
    max_display: int = 20,
) -> dict:
    """
    Compute SHAP values and generate visualizations for a single model.

    Parameters
    ----------
    model : sklearn/xgboost estimator
        Trained model.
    model_name : str
        Name for file naming and titles.
    X_sample : np.ndarray
        Subsample of test data for SHAP computation.
    feature_names : list
        Feature column names.
    max_display : int
        Max features to display in plots.

    Returns
    -------
    dict
        SHAP analysis results (feature importances, etc.).
    """
    print(f"\n  {'='*50}")
    print(f"  SHAP Analysis: {model_name}")
    print(f"  {'='*50}")

    X_df = pd.DataFrame(X_sample, columns=feature_names)

    # Choose the appropriate explainer
    model_type = type(model).__name__

    try:
        if "XGB" in model_type or "Booster" in model_type:
            print("  Using TreeExplainer (XGBoost) ...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
        elif "Forest" in model_type or "Tree" in model_type:
            print("  Using TreeExplainer (tree-based) ...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            # For binary classification, RandomForest returns [class0, class1] or 3D array
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 (fraud)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # Use class 1 (fraud)
        else:
            print("  Using LinearExplainer / KernelExplainer ...")
            # Use a background sample for KernelExplainer
            background = shap.sample(X_df, min(100, len(X_df)))
            try:
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(X_df)
            except Exception:
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

        print(f"  SHAP values shape: {np.array(shap_values).shape}")

    except Exception as e:
        print(f"  ERROR computing SHAP values: {e}")
        return {"error": str(e)}

    # --- Feature Importance (mean |SHAP|) ---
    shap_array = np.array(shap_values)
    # Handle 3D arrays from tree-based models (n_samples, n_features, n_classes)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]  # Use class 1 (fraud)
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    # Ensure importance values are scalar floats
    mean_abs_shap = np.array([float(v) if np.ndim(v) == 0 else float(np.mean(np.abs(v))) for v in mean_abs_shap])

    importance_ranking = sorted(
        zip(feature_names, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n  Top {min(10, len(importance_ranking))} Features by SHAP Importance:")
    for feat, imp in importance_ranking[:10]:
        print(f"    {feat:>10s}: {imp:.6f}")

    # --- Plot 1: Summary Plot (Beeswarm) ---
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
        plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        summary_path = os.path.join(FIGURES_DIR, f"shap_summary_{model_name}.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Figure saved: {summary_path}")
    except Exception as e:
        summary_path = None
        print(f"  Could not generate summary plot: {e}")

    # --- Plot 2: Bar Plot (Mean |SHAP|) ---
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, plot_type="bar",
                          max_display=max_display, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        bar_path = os.path.join(FIGURES_DIR, f"shap_importance_{model_name}.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Figure saved: {bar_path}")
    except Exception as e:
        bar_path = None
        print(f"  Could not generate bar plot: {e}")

    result = {
        "model_name": model_name,
        "n_samples_analyzed": int(X_sample.shape[0]),
        "feature_importance": {feat: round(imp, 6) for feat, imp in importance_ranking},
        "top_10_features": [feat for feat, _ in importance_ranking[:10]],
        "figures": {
            "summary_plot": summary_path,
            "bar_plot": bar_path,
        },
    }

    return result


# ===================================================================
# MAIN SHAP PIPELINE
# ===================================================================

def run_shap_analysis(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    sample_size: int = 1000,
) -> dict:
    """
    Run SHAP analysis on all available trained models.

    Parameters
    ----------
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    feature_names : list
        Feature names.
    sample_size : int
        Number of samples for SHAP computation (SHAP can be slow).

    Returns
    -------
    dict
        SHAP results for all models.
    """
    if not HAS_SHAP:
        print("[shap_analysis] SHAP library not available. Skipping.")
        return {"error": "shap not installed"}

    ensure_dirs()

    print("\n" + "#" * 70)
    print("#  SHAP EXPLAINABILITY ANALYSIS")
    print("#" * 70)

    # Subsample for SHAP efficiency
    if X_test.shape[0] > sample_size:
        np.random.seed(42)
        idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    print(f"  Using {X_sample.shape[0]} samples for SHAP analysis")

    model_files = {
        "logistic_regression": "logistic_regression.joblib",
        "random_forest": "random_forest.joblib",
        "xgboost": "xgboost.joblib",
    }

    all_results = {}

    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if not os.path.isfile(model_path):
            print(f"\n  [SKIP] {name}: model not found")
            continue

        model = joblib.load(model_path)
        result = analyze_model_shap(model, name, X_sample, feature_names)
        all_results[name] = result

    if not all_results:
        print("\n  No models available for SHAP analysis.")
        return {}

    # --- Cross-Model Feature Importance Comparison ---
    print("\n  " + "=" * 50)
    print("  CROSS-MODEL FEATURE IMPORTANCE COMPARISON")
    print("  " + "=" * 50)

    try:
        all_importances = {}
        for name, res in all_results.items():
            if "feature_importance" in res:
                all_importances[name] = res["feature_importance"]

        if len(all_importances) > 1:
            # Create comparison plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Get union of top features across all models
            top_features = set()
            for imp_dict in all_importances.values():
                sorted_feats = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
                top_features.update([f for f, _ in sorted_feats[:15]])
            top_features = sorted(top_features)

            x = np.arange(len(top_features))
            width = 0.8 / len(all_importances)
            colors = ["#2196F3", "#4CAF50", "#FF5722"]

            for i, (model_name, imp_dict) in enumerate(all_importances.items()):
                vals = [imp_dict.get(f, 0) for f in top_features]
                ax.barh(x + i * width, vals, width, label=model_name,
                        color=colors[i % len(colors)], alpha=0.8)

            ax.set_yticks(x + width * (len(all_importances) - 1) / 2)
            ax.set_yticklabels(top_features)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("Cross-Model Feature Importance Comparison", fontsize=14, fontweight="bold")
            ax.legend()
            ax.invert_yaxis()

            plt.tight_layout()
            comparison_path = os.path.join(FIGURES_DIR, "shap_model_comparison.png")
            plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"  Figure saved: {comparison_path}")

            all_results["cross_model_comparison"] = {
                "figure": comparison_path,
            }
    except Exception as e:
        print(f"  Cross-model comparison error: {e}")

    # Save results JSON
    # Clean up non-serializable entries
    json_results = {}
    for name, res in all_results.items():
        json_results[name] = {
            k: v for k, v in res.items()
            if k != "shap_values"
        }

    results_path = os.path.join(MODELS_DIR, "shap_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n  SHAP results saved to: {results_path}")

    return all_results


# ===================================================================
# STANDALONE EXECUTION
# ===================================================================

if __name__ == "__main__":
    preprocessed_path = os.path.join(MODELS_DIR, "preprocessed_data.joblib")

    if not os.path.isfile(preprocessed_path):
        print("[shap_analysis] Preprocessed data not found. Running preprocessing ...")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
        from preprocessing.preprocess import run_preprocessing
        try:
            data = run_preprocessing()
        except FileNotFoundError as e:
            print(f"[shap_analysis] ✗ {e}")
            sys.exit(1)
    else:
        data = joblib.load(preprocessed_path)

    results = run_shap_analysis(
        X_test=data["X_test"],
        y_test=data["y_test"],
        feature_names=data["feature_names"],
    )

    if results:
        print("\n[shap_analysis] ✓ SHAP analysis completed successfully!")
    else:
        print("\n[shap_analysis] ✗ SHAP analysis failed or no models found.")
