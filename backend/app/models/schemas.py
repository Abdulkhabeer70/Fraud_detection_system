"""
Pydantic models (schemas) for request validation and response serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════════════════════════

class TransactionInput(BaseModel):
    """Input features for a single credit-card transaction."""

    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    model_config = {"json_schema_extra": {
        "examples": [{
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": -0.34,
            "V25": 0.17, "V26": 0.13, "V27": -0.19, "V28": -0.14,
            "Amount": 149.62,
        }]
    }}


class PredictionResponse(BaseModel):
    """Result of a fraud-detection prediction."""

    prediction: int = Field(..., description="0 = legitimate, 1 = fraud")
    probability: float = Field(..., description="Probability of fraud (0–1)")
    risk_level: str = Field(..., description="Low / Medium / High")
    model_used: str = Field(..., description="Name of the model that produced the prediction")


class PredictionHistoryItem(BaseModel):
    """A single past prediction record."""

    id: int
    amount: float
    prediction: int
    probability: float
    risk_level: str
    model_used: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionHistoryResponse(BaseModel):
    """Paginated list of prediction history."""

    total: int
    predictions: List[PredictionHistoryItem]


# ═══════════════════════════════════════════════════════════════════════
# Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════

class AmountStats(BaseModel):
    """Descriptive statistics for the Amount column."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float


class StatsResponse(BaseModel):
    """High-level statistics about the credit-card dataset."""

    total_transactions: int
    fraud_count: int
    legit_count: int
    fraud_percentage: float
    amount_stats_overall: AmountStats
    amount_stats_fraud: AmountStats
    amount_stats_legit: AmountStats


# ═══════════════════════════════════════════════════════════════════════
# Model Performance
# ═══════════════════════════════════════════════════════════════════════

class SingleModelMetrics(BaseModel):
    """Metrics for one model."""

    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    training_time: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    additional_info: Optional[Dict[str, Any]] = None


class ModelPerformanceResponse(BaseModel):
    """Evaluation results for all trained models."""

    models: List[SingleModelMetrics]
    best_model: Optional[str] = None
    comparison_notes: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Data Mining Results
# ═══════════════════════════════════════════════════════════════════════

class AnomalyDetectionResult(BaseModel):
    """Results from a single anomaly-detection method."""

    method: str = Field(..., description="E.g. Isolation Forest, LOF, One-Class SVM")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    num_anomalies_detected: Optional[int] = None
    contamination: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ClusterAnalysisResult(BaseModel):
    """Results from cluster analysis (e.g. K-Means, DBSCAN)."""

    method: str
    num_clusters: Optional[int] = None
    silhouette_score: Optional[float] = None
    fraud_distribution_per_cluster: Optional[Dict[str, Any]] = None
    cluster_sizes: Optional[List[int]] = None
    parameters: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class PatternFinding(BaseModel):
    """A single mined pattern or insight."""

    category: str = Field(
        ..., description="E.g. 'Feature Correlation', 'Association Rule', 'Temporal Pattern'"
    )
    description: str
    importance: Optional[str] = None  # High / Medium / Low
    supporting_metrics: Optional[Dict[str, Any]] = None


class FeatureImportanceItem(BaseModel):
    """Importance ranking for a single feature."""

    feature: str
    importance: float
    rank: int


class DimensionalityReductionResult(BaseModel):
    """Results from PCA / t-SNE / UMAP exploration."""

    method: str
    explained_variance: Optional[List[float]] = None
    num_components: Optional[int] = None
    notes: Optional[str] = None


class MiningResultsResponse(BaseModel):
    """Comprehensive data-mining results – the centrepiece of the project."""

    anomaly_detection: Optional[List[AnomalyDetectionResult]] = None
    cluster_analysis: Optional[List[ClusterAnalysisResult]] = None
    pattern_findings: Optional[List[PatternFinding]] = None
    feature_importance: Optional[List[FeatureImportanceItem]] = None
    dimensionality_reduction: Optional[List[DimensionalityReductionResult]] = None
    association_rules: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """API health-check payload."""

    status: str
    timestamp: datetime
    models_loaded: bool
    available_models: List[str]
    database_connected: bool
