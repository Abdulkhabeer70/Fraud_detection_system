"""
Dataset statistics route – GET /stats.

Loads ``creditcard.csv`` once and caches the computed statistics in memory
so subsequent requests are instant.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import DATASET_PATH
from app.models.schemas import AmountStats, StatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Statistics"])

# ── In-memory cache ────────────────────────────────────────────────────
_cached_stats: Optional[StatsResponse] = None


def _compute_amount_stats(series: pd.Series) -> AmountStats:
    """Compute descriptive statistics for a pandas Series."""
    return AmountStats(
        mean=round(float(series.mean()), 4),
        median=round(float(series.median()), 4),
        std=round(float(series.std()), 4),
        min=round(float(series.min()), 4),
        max=round(float(series.max()), 4),
        q25=round(float(series.quantile(0.25)), 4),
        q75=round(float(series.quantile(0.75)), 4),
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Dataset statistics",
    description=(
        "Returns high-level statistics about the credit-card transactions "
        "dataset, including class balance and amount distributions for "
        "overall, fraud, and legitimate transactions."
    ),
)
def get_stats() -> StatsResponse:
    """Load dataset statistics, caching after first computation."""
    global _cached_stats

    if _cached_stats is not None:
        return _cached_stats

    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Dataset not found at {DATASET_PATH}. "
                "Please place creditcard.csv in the data/ directory."
            ),
        )

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as exc:
        logger.exception("Failed to read dataset")
        raise HTTPException(
            status_code=500,
            detail=f"Error reading dataset: {exc}",
        )

    if "Class" not in df.columns or "Amount" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Dataset must contain 'Class' and 'Amount' columns.",
        )

    total = len(df)
    fraud = int(df["Class"].sum())
    legit = total - fraud

    _cached_stats = StatsResponse(
        total_transactions=total,
        fraud_count=fraud,
        legit_count=legit,
        fraud_percentage=round(fraud / total * 100, 4) if total else 0.0,
        amount_stats_overall=_compute_amount_stats(df["Amount"]),
        amount_stats_fraud=_compute_amount_stats(df.loc[df["Class"] == 1, "Amount"]),
        amount_stats_legit=_compute_amount_stats(df.loc[df["Class"] == 0, "Amount"]),
    )

    logger.info("Dataset stats computed and cached (%d rows)", total)
    return _cached_stats
