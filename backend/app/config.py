"""
Configuration settings for the Credit Card Fraud Detection API.

Centralizes all paths, filenames, and application settings so that
other modules never hard-code locations.
"""

from pathlib import Path
from typing import Dict, List


# ── Project root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:/data_mining_project")

# ── Data paths ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "creditcard.csv"

# ── ML artefact paths ──────────────────────────────────────────────────
ML_DIR = PROJECT_ROOT / "ml"
MODELS_DIR = ML_DIR / "models"
FIGURES_DIR = ML_DIR / "figures"

# ── Database ────────────────────────────────────────────────────────────
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'fraud_detection.db'}"

# ── Model filenames (joblib / pickle) ───────────────────────────────────
MODEL_FILES: Dict[str, str] = {
    "xgboost": "xgboost.joblib",
    "random_forest": "random_forest.joblib",
    "logistic_regression": "logistic_regression.joblib",
}

SCALER_FILE = "scaler_amount.joblib"
DEFAULT_MODEL = "logistic_regression"

# ── Evaluation / mining result files ────────────────────────────────────
EVALUATION_RESULTS_FILE = MODELS_DIR / "evaluation_results.json"
MINING_RESULTS_FILE = MODELS_DIR / "mining_results.json"

# ── Ordered list of feature columns expected by the models ──────────────
# Time is included because models were trained with it; API defaults Time=0
FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ── Application settings ───────────────────────────────────────────────
APP_TITLE = "Credit Card Fraud Detection API"
APP_DESCRIPTION = (
    "A data-mining-driven REST API for real-time credit card fraud "
    "prediction, exploratory statistics, model performance comparison, "
    "and advanced mining-result retrieval (anomaly detection, clustering, "
    "pattern analysis)."
)
APP_VERSION = "1.0.0"

# Risk-level probability thresholds
RISK_THRESHOLDS = {
    "high": 0.75,   # probability >= 0.75  → High
    "medium": 0.40,  # probability >= 0.40  → Medium
    # everything below                     → Low
}
