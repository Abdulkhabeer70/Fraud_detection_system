"""
Model performance route – GET /model-performance.

Reads ``evaluation_results.json`` from the ml/models/ directory and returns
structured metrics for every trained model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.config import EVALUATION_RESULTS_FILE

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model Performance"])


def _parse_model_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw JSON dict into frontend-compatible field names.
    Frontend expects: name, accuracy, precision, recall, f1, auc
    """
    return {
        "name": raw.get("model_name", raw.get("name", "unknown")),
        "accuracy": raw.get("accuracy"),
        "precision": raw.get("precision", raw.get("precision_score")),
        "recall": raw.get("recall"),
        "f1": raw.get("f1", raw.get("f1_score")),
        "auc": raw.get("roc_auc", raw.get("auc")),
        "training_time": raw.get("training_time", raw.get("train_time")),
        "true_positives": raw.get("true_positives"),
        "false_positives": raw.get("false_positives"),
        "true_negatives": raw.get("true_negatives"),
        "false_negatives": raw.get("false_negatives"),
        "mcc": raw.get("mcc"),
        "specificity": raw.get("specificity"),
    }


def _determine_best_model(models: List[Dict[str, Any]]) -> Optional[str]:
    """Pick the model with the highest F1 score (or AUC as fallback)."""
    best = None
    best_score = -1.0
    for m in models:
        score = m.get("f1") or m.get("auc") or 0.0
        if score > best_score:
            best_score = score
            best = m
    return best["name"] if best else None


@router.get(
    "/model-performance",
    summary="Model evaluation metrics",
    description=(
        "Returns accuracy, precision, recall, F1, ROC-AUC, and other "
        "metrics for every trained model. Reads from "
        "``ml/models/evaluation_results.json``."
    ),
)
def get_model_performance():
    """Load and return evaluation metrics for all models as raw JSON."""

    if not EVALUATION_RESULTS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Evaluation results not found. Train and evaluate the "
                "models first to generate evaluation_results.json."
            ),
        )

    try:
        with open(EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.exception("Malformed evaluation_results.json")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse evaluation_results.json: {exc}",
        )
    except Exception as exc:
        logger.exception("Error reading evaluation_results.json")
        raise HTTPException(status_code=500, detail=str(exc))

    # Support multiple JSON formats:
    #   1. A bare list of model dicts
    #   2. A wrapper object with a "models" key
    #   3. A dict keyed by model name (e.g. {"logistic_regression": {...}, ...})
    if isinstance(data, list):
        raw_models = data
        notes = None
    elif isinstance(data, dict):
        if "models" in data or "results" in data:
            raw_models = data.get("models", data.get("results", []))
            notes = data.get("comparison_notes", data.get("notes"))
        else:
            # Dict keyed by model name → convert to list
            raw_models = []
            for model_name, metrics in data.items():
                if isinstance(metrics, dict):
                    metrics["model_name"] = model_name
                    raw_models.append(metrics)
            notes = None
    else:
        raise HTTPException(
            status_code=500,
            detail="Unexpected format in evaluation_results.json.",
        )

    models = [_parse_model_metrics(m) for m in raw_models]
    best = _determine_best_model(models)

    # Build confusion matrix from best model's data
    best_model_data = next((m for m in models if m["name"] == best), None)
    confusion_matrix = None
    if best_model_data:
        confusion_matrix = {
            "true_positives": best_model_data.get("true_positives"),
            "false_positives": best_model_data.get("false_positives"),
            "true_negatives": best_model_data.get("true_negatives"),
            "false_negatives": best_model_data.get("false_negatives"),
        }

    return {
        "models": models,
        "best_model": best,
        "comparison_notes": notes,
        "confusion_matrix": confusion_matrix,
    }
