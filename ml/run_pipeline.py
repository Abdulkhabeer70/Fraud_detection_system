"""
run_pipeline.py - Master Pipeline Script
==========================================

End-to-end execution of the complete ML & Data Mining pipeline:

  Stage 1: Preprocessing  (load, scale, split, SMOTE)
  Stage 2: Model Training (Logistic Regression, Random Forest, XGBoost)
  Stage 3: Data Mining     (Isolation Forest, LOF, K-Means, DBSCAN, patterns)
  Stage 4: Evaluation      (metrics, confusion matrices, ROC/PR curves)
  Stage 5: Explainability  (SHAP values and feature importance)

Usage:
    python ml/run_pipeline.py                     # Run full pipeline
    python ml/run_pipeline.py --skip-shap         # Skip SHAP (faster)
    python ml/run_pipeline.py --skip-mining       # Skip mining module
    python ml/run_pipeline.py --only-mining       # Only run mining

Output:
    ml/models/  — Saved models (.joblib), metrics (.json)
    ml/figures/ — All visualizations (.png)
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))

MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "ml", "figures")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")


def banner(text: str, char: str = "=", width: int = 70):
    """Print a formatted banner."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def run_full_pipeline(skip_shap: bool = False, skip_mining: bool = False,
                      only_mining: bool = False):
    """
    Execute the full pipeline end-to-end.

    Parameters
    ----------
    skip_shap : bool
        Skip the SHAP explainability stage (faster).
    skip_mining : bool
        Skip the data mining stage.
    only_mining : bool
        Only run preprocessing + data mining.
    """
    pipeline_start = time.time()

    banner("CREDIT CARD FRAUD DETECTION — ML & DATA MINING PIPELINE", "█")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Data path    : {DATA_PATH}")
    print(f"  Models dir   : {MODELS_DIR}")
    print(f"  Figures dir  : {FIGURES_DIR}")
    print(f"  Skip SHAP    : {skip_shap}")
    print(f"  Skip Mining  : {skip_mining}")
    print(f"  Only Mining  : {only_mining}")

    # ---------------------------------------------------------------
    # STAGE 1: PREPROCESSING
    # ---------------------------------------------------------------
    banner("STAGE 1: DATA PREPROCESSING", "▓")

    try:
        from preprocessing.preprocess import run_preprocessing

        stage_start = time.time()
        preprocessed = run_preprocessing(DATA_PATH)
        stage_time = round(time.time() - stage_start, 1)
        print(f"\n  ✓ Preprocessing completed in {stage_time}s")

    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        print("  Pipeline cannot continue without data.")
        return
    except Exception as e:
        print(f"\n  ✗ Preprocessing failed: {e}")
        raise

    # ---------------------------------------------------------------
    # STAGE 2: SUPERVISED MODEL TRAINING
    # ---------------------------------------------------------------
    if not only_mining:
        banner("STAGE 2: SUPERVISED MODEL TRAINING", "▓")

        try:
            from training.train_models import train_all_models

            stage_start = time.time()
            training_results = train_all_models(
                X_train=preprocessed["X_train_smote"],
                y_train=preprocessed["y_train_smote"],
                X_test=preprocessed["X_test"],
                y_test=preprocessed["y_test"],
            )
            stage_time = round(time.time() - stage_start, 1)
            print(f"\n  ✓ Model training completed in {stage_time}s")

        except Exception as e:
            print(f"\n  ✗ Model training failed: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------
    # STAGE 3: DATA MINING
    # ---------------------------------------------------------------
    if not skip_mining:
        banner("STAGE 3: DATA MINING ANALYSIS", "▓")

        try:
            from training.train_mining import run_mining_pipeline
            import numpy as np

            # Use the full dataset for mining (not just train/test)
            X_full = np.vstack([preprocessed["X_train"], preprocessed["X_test"]])
            y_full = np.concatenate([preprocessed["y_train"], preprocessed["y_test"]])

            stage_start = time.time()
            mining_results = run_mining_pipeline(
                X=X_full,
                y=y_full,
                df_scaled=preprocessed.get("df_scaled"),
                feature_names=preprocessed["feature_names"],
            )
            stage_time = round(time.time() - stage_start, 1)
            print(f"\n  ✓ Data mining completed in {stage_time}s")

        except Exception as e:
            print(f"\n  ✗ Data mining failed: {e}")
            import traceback
            traceback.print_exc()

    if only_mining:
        total_time = round(time.time() - pipeline_start, 1)
        banner(f"PIPELINE COMPLETE (Mining Only) — Total: {total_time}s", "█")
        return

    # ---------------------------------------------------------------
    # STAGE 4: MODEL EVALUATION
    # ---------------------------------------------------------------
    banner("STAGE 4: MODEL EVALUATION", "▓")

    try:
        from evaluation.evaluate import evaluate_all_models

        stage_start = time.time()
        eval_results = evaluate_all_models(
            X_test=preprocessed["X_test"],
            y_test=preprocessed["y_test"],
        )
        stage_time = round(time.time() - stage_start, 1)
        print(f"\n  ✓ Evaluation completed in {stage_time}s")

    except Exception as e:
        print(f"\n  ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------------
    # STAGE 5: SHAP EXPLAINABILITY
    # ---------------------------------------------------------------
    if not skip_shap:
        banner("STAGE 5: SHAP EXPLAINABILITY", "▓")

        try:
            from explainability.shap_analysis import run_shap_analysis

            stage_start = time.time()
            shap_results = run_shap_analysis(
                X_test=preprocessed["X_test"],
                y_test=preprocessed["y_test"],
                feature_names=preprocessed["feature_names"],
            )
            stage_time = round(time.time() - stage_start, 1)
            print(f"\n  ✓ SHAP analysis completed in {stage_time}s")

        except Exception as e:
            print(f"\n  ✗ SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n  [SKIP] SHAP explainability (--skip-shap flag)")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    total_time = round(time.time() - pipeline_start, 1)

    banner(f"PIPELINE COMPLETE — Total Runtime: {total_time}s", "█")

    print("  Output files:")
    for subdir in [MODELS_DIR, FIGURES_DIR]:
        if os.path.isdir(subdir):
            files = os.listdir(subdir)
            for f in sorted(files):
                fpath = os.path.join(subdir, f)
                if os.path.isfile(fpath):
                    size_kb = round(os.path.getsize(fpath) / 1024, 1)
                    print(f"    {os.path.relpath(fpath, PROJECT_ROOT):<50s} ({size_kb} KB)")

    print(f"\n  Total pipeline runtime: {total_time}s")
    print("  Done! ✓")


# ===================================================================
# ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — ML & Data Mining Pipeline"
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explainability analysis (faster execution)",
    )
    parser.add_argument(
        "--skip-mining",
        action="store_true",
        help="Skip the data mining analysis stage",
    )
    parser.add_argument(
        "--only-mining",
        action="store_true",
        help="Only run preprocessing and data mining (skip supervised models, eval, SHAP)",
    )

    args = parser.parse_args()

    run_full_pipeline(
        skip_shap=args.skip_shap,
        skip_mining=args.skip_mining,
        only_mining=args.only_mining,
    )
