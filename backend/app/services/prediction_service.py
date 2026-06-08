"""
Prediction service – model loading, scaling, and inference.

All model artefacts live under ``ml/models/``.  The service caches loaded
objects in module-level dictionaries so that models are loaded **once** and
reused across requests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from app.config import (
    DEFAULT_MODEL,
    FEATURE_COLUMNS,
    MODEL_FILES,
    MODELS_DIR,
    RISK_THRESHOLDS,
    SCALER_FILE,
)

logger = logging.getLogger(__name__)

# ── In-memory caches ───────────────────────────────────────────────────
_loaded_models: Dict[str, Any] = {}
_scaler: Optional[Any] = None


# ═══════════════════════════════════════════════════════════════════════
# Loading helpers
# ═══════════════════════════════════════════════════════════════════════

def load_scaler() -> Any:
    """Load the feature scaler (StandardScaler / RobustScaler) from disk.

    Returns ``None`` if the file does not exist yet (models not trained).
    """
    global _scaler
    if _scaler is not None:
        return _scaler

    scaler_path = MODELS_DIR / SCALER_FILE
    if not scaler_path.exists():
        logger.warning("Scaler file not found at %s", scaler_path)
        return None

    _scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded from %s", scaler_path)
    return _scaler


def load_model(model_name: str | None = None) -> Tuple[Any, str]:
    """Load a trained model by name.

    Parameters
    ----------
    model_name : str, optional
        Key into ``MODEL_FILES``.  Falls back to ``DEFAULT_MODEL``.

    Returns
    -------
    (model_object, model_name) or (None, model_name) if unavailable.
    """
    name = (model_name or DEFAULT_MODEL).lower()

    # Return from cache
    if name in _loaded_models:
        return _loaded_models[name], name

    filename = MODEL_FILES.get(name)
    if filename is None:
        logger.error("Unknown model name: %s", name)
        return None, name

    model_path = MODELS_DIR / filename
    if not model_path.exists():
        logger.warning("Model file not found: %s", model_path)
        return None, name

    model = joblib.load(model_path)
    _loaded_models[name] = model
    logger.info("Model '%s' loaded from %s", name, model_path)
    return model, name


def get_available_models() -> List[str]:
    """Return names of models whose artefact files exist on disk."""
    available: List[str] = []
    for name, filename in MODEL_FILES.items():
        if (MODELS_DIR / filename).exists():
            available.append(name)
    return available


def preload_all_models() -> None:
    """Eagerly load every available model + the scaler at startup."""
    load_scaler()
    for name in get_available_models():
        load_model(name)
    logger.info(
        "Pre-loaded %d model(s): %s",
        len(_loaded_models),
        list(_loaded_models.keys()),
    )


# ═══════════════════════════════════════════════════════════════════════
# Prediction logic
# ═══════════════════════════════════════════════════════════════════════

def _features_to_array(features: Dict[str, float]) -> np.ndarray:
    """Convert a dict of feature values into a 1-D numpy array in the
    correct column order expected by the model.
    Time defaults to 0.0 if not provided by the user."""
    row = []
    for col in FEATURE_COLUMNS:
        row.append(float(features.get(col, 0.0)))
    return np.array([row])


def predict_transaction(
    features: Dict[str, float],
    model_name: str | None = None,
) -> Dict[str, Any]:
    """Run inference on a single transaction.

    Parameters
    ----------
    features : dict
        Must contain keys ``V1`` … ``V28`` and ``Amount``.
    model_name : str, optional
        Which model to use. Defaults to ``DEFAULT_MODEL``.

    Returns
    -------
    dict with ``prediction``, ``probability``, ``risk_level``,
    ``model_used``, and ``features_hash``.

    Raises
    ------
    RuntimeError
        If the requested model or the scaler is not available.
    """
    model, used_name = load_model(model_name)
    if model is None:
        raise RuntimeError(
            f"Model '{used_name}' is not available. Train models first."
        )

    # Build feature array
    X = _features_to_array(features)

    # Scale the Amount column (last column) if a scaler is available
    scaler = load_scaler()
    if scaler is not None:
        try:
            # The scaler was fitted on the Amount column only
            amount_idx = FEATURE_COLUMNS.index("Amount")
            X[0, amount_idx] = scaler.transform(
                X[0, amount_idx].reshape(-1, 1)
            )[0, 0]
        except Exception:
            # If the scaler was fitted on all features, transform the whole row
            try:
                X = scaler.transform(X)
            except Exception as exc:
                logger.warning("Scaler transform failed, using raw values: %s", exc)

    # Predict
    prediction: int = int(model.predict(X)[0])

    # Probability
    if hasattr(model, "predict_proba"):
        probability: float = float(model.predict_proba(X)[0][1])
    elif hasattr(model, "decision_function"):
        # For SVM-style models, map decision function to [0, 1]
        raw = float(model.decision_function(X)[0])
        probability = 1.0 / (1.0 + np.exp(-raw))  # sigmoid
    else:
        probability = float(prediction)

    risk_level = get_risk_level(probability)

    # Hash of V1–V28 for deduplication
    v_values = json.dumps([features[f"V{i}"] for i in range(1, 29)], sort_keys=True)
    features_hash = hashlib.sha256(v_values.encode()).hexdigest()[:16]

    return {
        "prediction": prediction,
        "probability": round(probability, 6),
        "risk_level": risk_level,
        "model_used": used_name,
        "features_hash": features_hash,
    }


# ═══════════════════════════════════════════════════════════════════════
# Risk classification
# ═══════════════════════════════════════════════════════════════════════

def get_risk_level(probability: float) -> str:
    """Map a fraud probability to a human-readable risk level.

    Uses thresholds from ``config.RISK_THRESHOLDS``.
    """
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"
