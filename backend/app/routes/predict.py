"""
Prediction routes – POST /predict and GET /predictions/history.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import Prediction, get_db
from app.models.schemas import (
    PredictionHistoryResponse,
    PredictionHistoryItem,
    PredictionResponse,
    TransactionInput,
)
from app.services.prediction_service import predict_transaction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


# ── POST /predict ───────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud for a single transaction",
    description=(
        "Accepts V1–V28 (PCA components) and Amount, runs the selected "
        "model (default: XGBoost), and returns the prediction, fraud "
        "probability, and risk level."
    ),
)
def predict(
    transaction: TransactionInput,
    model_name: Optional[str] = Query(
        default=None,
        description="Model to use (xgboost, random_forest, logistic_regression, …). "
                    "Defaults to xgboost.",
    ),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Run fraud prediction and log to the database."""
    features = transaction.model_dump()

    try:
        result = predict_transaction(features, model_name=model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {exc}",
        )

    # Persist to DB
    try:
        record = Prediction(
            features_hash=result["features_hash"],
            amount=features["Amount"],
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["risk_level"],
            model_used=result["model_used"],
        )
        db.add(record)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to log prediction to DB: %s", exc)
        db.rollback()

    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        model_used=result["model_used"],
    )


# ── GET /predictions/history ────────────────────────────────────────────

@router.get(
    "/predictions/history",
    response_model=PredictionHistoryResponse,
    summary="Retrieve past prediction history",
    description="Returns a paginated list of past predictions, newest first.",
)
def prediction_history(
    skip: int = Query(0, ge=0, description="Records to skip"),
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Return prediction history from the database."""
    try:
        total = db.query(Prediction).count()
        rows = (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
    except Exception as exc:
        logger.exception("Failed to query prediction history")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    items = [
        PredictionHistoryItem.model_validate(row)
        for row in rows
    ]

    return PredictionHistoryResponse(total=total, predictions=items)
