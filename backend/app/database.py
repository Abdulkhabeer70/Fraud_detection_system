"""
Database layer – SQLAlchemy + SQLite.

Tables
------
- **predictions** : stores every prediction request & result.
- **model_metrics** : stores per-model evaluation metrics.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_DIR, DATABASE_URL


# ── Engine & session factory ────────────────────────────────────────────
# Create the database directory if it doesn't exist yet.
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Declarative base ───────────────────────────────────────────────────
class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


# ── ORM Models ─────────────────────────────────────────────────────────

class Prediction(Base):
    """A single fraud-detection prediction log."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # We store a hash of V1–V28 to avoid 28 float columns;
    # this lets us detect duplicate transactions quickly.
    features_hash = Column(String, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)          # 0 or 1
    probability = Column(Float, nullable=False)            # fraud prob
    risk_level = Column(String, nullable=False)            # Low / Medium / High
    model_used = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} prediction={self.prediction} "
            f"risk={self.risk_level} model={self.model_used}>"
        )


class ModelMetric(Base):
    """Evaluation metrics for a trained model."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False, index=True)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<ModelMetric model={self.model_name} f1={self.f1}>"


# ── Helper ──────────────────────────────────────────────────────────────

def create_tables() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency – yields a DB session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
