"""
Data Mining results route – GET /mining-results.

Reads the mining_results.json produced by the offline pipeline and
transforms it into a structured response.  The JSON uses flat top-level
keys (isolation_forest, lof, kmeans, dbscan, pattern_analysis, …) which
are remapped here into the expected schema sections.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.config import MINING_RESULTS_FILE

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Data Mining"])


def _best(results_list: list, key: str = "f1_score"):
    """Return the entry with the highest value for *key*."""
    if not results_list:
        return {}
    return max(results_list, key=lambda r: r.get(key, 0))


@router.get(
    "/mining-results",
    summary="Comprehensive data-mining results",
    description=(
        "Returns the full suite of data-mining outputs: anomaly detection "
        "(Isolation Forest, LOF), cluster analysis (K-Means, DBSCAN), "
        "pattern analysis, feature importance, and statistical summaries."
    ),
)
def get_mining_results() -> Dict[str, Any]:
    """Load and return all data-mining results from disk as raw JSON."""

    if not MINING_RESULTS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Mining results not found. Run the data-mining pipeline "
                "first to generate mining_results.json."
            ),
        )

    try:
        with open(MINING_RESULTS_FILE, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.exception("Malformed mining_results.json")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse mining_results.json: {exc}",
        )
    except Exception as exc:
        logger.exception("Error reading mining_results.json")
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Normalise to the structure the frontend expects ──────────────
    out: Dict[str, Any] = {}

    # Anomaly Detection
    if_raw = data.get("isolation_forest", {})
    lof_raw = data.get("lof", {})
    best_if = _best(if_raw.get("results_per_contamination", []))
    best_lof_list = lof_raw.get("results_per_n_neighbors", [])
    best_lof = _best(best_lof_list)

    out["anomaly_detection"] = {
        "isolation_forest": {
            "method": "Isolation Forest",
            "best_contamination": if_raw.get("best_contamination"),
            "detection_rate": best_if.get("recall", 0),
            "false_positive_rate": (
                best_if.get("false_positives", 0)
                / max(best_if.get("false_positives", 0) + best_if.get("true_negatives", 1), 1)
            ),
            "precision": best_if.get("precision"),
            "recall": best_if.get("recall"),
            "f1_score": if_raw.get("best_f1_score"),
            "anomalies_found": best_if.get("anomalies_detected"),
            "true_positives": best_if.get("true_positives"),
            "false_positives": best_if.get("false_positives"),
            "true_negatives": best_if.get("true_negatives"),
            "false_negatives": best_if.get("false_negatives"),
            "contamination": if_raw.get("best_contamination"),
        },
        "local_outlier_factor": {
            "method": "Local Outlier Factor",
            "best_n_neighbors": lof_raw.get("best_n_neighbors"),
            "detection_rate": best_lof.get("recall", 0),
            "false_positive_rate": (
                best_lof.get("false_positives", 0)
                / max(best_lof.get("false_positives", 0) + best_lof.get("true_negatives", 1), 1)
            ),
            "precision": best_lof.get("precision"),
            "recall": best_lof.get("recall"),
            "f1_score": lof_raw.get("best_f1_score"),
            "anomalies_found": best_lof.get("anomalies_detected"),
            "true_positives": best_lof.get("true_positives"),
            "false_positives": best_lof.get("false_positives"),
            "true_negatives": best_lof.get("true_negatives"),
            "false_negatives": best_lof.get("false_negatives"),
        },
        # Contamination sweep data for the performance curve chart
        "if_sweep": if_raw.get("results_per_contamination", []),
        "lof_sweep": best_lof_list,
    }

    # Clustering
    km_raw = data.get("kmeans", {})
    db_raw = data.get("dbscan", {})
    out["clustering"] = {
        "kmeans": {
            "n_clusters": km_raw.get("n_clusters"),
            "silhouette_score": km_raw.get("silhouette_score"),
            "inertia": km_raw.get("inertia"),
            "cluster_distribution": km_raw.get("cluster_fraud_distribution", []),
        },
        "dbscan": {
            "eps": db_raw.get("eps"),
            "min_samples": db_raw.get("min_samples"),
            "n_clusters_found": db_raw.get("n_clusters"),
            "noise_points": db_raw.get("n_noise"),
            "noise_points_fraud_pct": db_raw.get("noise_fraud_rate", 0) * 100,
            "silhouette_score": db_raw.get("silhouette_score"),
        },
    }

    # Pattern Analysis
    pat_raw = data.get("pattern_analysis", {})
    out["pattern_discovery"] = {
        "association_rules": pat_raw.get("association_rules", []),
        "temporal_patterns": pat_raw.get("temporal_patterns", {}),
        "key_insights": pat_raw.get("key_insights", []),
        "feature_correlations": pat_raw.get("feature_correlations", {}),
        "statistical_summary": pat_raw.get("statistical_summary", {}),
    }

    # Anomaly comparison table (pre-computed)
    out["anomaly_comparison"] = data.get("anomaly_comparison", {})

    # Pipeline metadata
    out["pipeline_metadata"] = data.get("pipeline_metadata", {})

    return out
