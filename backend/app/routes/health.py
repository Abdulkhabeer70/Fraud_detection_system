"""
Health-check route – GET /health.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.prediction_service import get_available_models, load_scaler

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description=(
        "Returns the current API status, whether models are loaded, and "
        "database connectivity."
    ),
)
def health_check() -> HealthResponse:
    """Return a snapshot of system health."""
    available = get_available_models()
    scaler_ok = load_scaler() is not None

    # Quick DB connectivity test
    db_ok = False
    try:
        from app.database import SessionLocal

        session = SessionLocal()
        session.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )
        session.close()
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if available else "degraded",
        timestamp=datetime.now(timezone.utc),
        models_loaded=bool(available) and scaler_ok,
        available_models=available,
        database_connected=db_ok,
    )
