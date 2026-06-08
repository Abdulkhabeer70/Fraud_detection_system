"""
train_mining.py - Data Mining & Unsupervised Analysis
======================================================

THE CORE DATA MINING MODULE for the Credit Card Fraud Detection project.

This module implements a comprehensive suite of data mining techniques:

  A. ANOMALY DETECTION
     1. Isolation Forest       — tree-based anomaly scoring with contamination tuning
     2. Local Outlier Factor   — density-based novelty detection (LOF)

  B. CLUSTERING ANALYSIS
     3. K-Means Clustering     — k=2,3,4 with silhouette & Calinski-Harabasz analysis
     4. DBSCAN Clustering      — density-based clustering with epsilon optimization

  C. PATTERN ANALYSIS
     5. Transaction Patterns   — amount-based, temporal, and feature-based patterns
     6. Anomaly Scoring Comparison — quantitative comparison across all methods

All results are saved as:
  - ml/models/mining_results.json   (comprehensive JSON with all findings)
  - ml/figures/*.png                (visualizations for each technique)

This file is designed to be the most detailed and comprehensive module
in the project, reflecting the data mining focus of the semester project.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from collections import OrderedDict

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "ml", "figures")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================================
#  HELPER: Dimensionality Reduction for Visualization
# ===================================================================

def reduce_for_visualization(X: np.ndarray, method: str = "pca", n_samples: int = 5000,
                             random_state: int = 42) -> np.ndarray:
    """
    Reduce data to 2D for scatter-plot visualization.

    For large datasets, we subsample before applying t-SNE (which is O(n²)).
    PCA is used as the default since it's fast and deterministic.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    method : str
        'pca' or 'tsne'.
    n_samples : int
        Maximum number of samples (for t-SNE performance).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    np.ndarray
        2D embedding of shape (n, 2).
    """
    if method == "tsne" and X.shape[0] > n_samples:
        idx = np.random.RandomState(random_state).choice(X.shape[0], n_samples, replace=False)
        X = X[idx]

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        # First reduce with PCA to 30 dims for t-SNE efficiency
        if X.shape[1] > 30:
            X = PCA(n_components=30, random_state=random_state).fit_transform(X)
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)

    return reducer.fit_transform(X)


# ###################################################################
#  SECTION A: ANOMALY DETECTION
# ###################################################################

# ===================================================================
#  A1. ISOLATION FOREST
# ===================================================================

def run_isolation_forest(X: np.ndarray, y: np.ndarray,
                         contamination_levels: list = None) -> dict:
    """
    Isolation Forest anomaly detection with multiple contamination levels.

    Isolation Forest isolates observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and
    minimum values of that feature. Anomalies are isolated in fewer
    splits (shorter path length in the tree), yielding lower anomaly scores.

    We sweep over multiple contamination levels to find the optimal
    trade-off between precision and recall for fraud detection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels (0=legit, 1=fraud) — used for evaluation only.
    contamination_levels : list of float
        Contamination fractions to try.

    Returns
    -------
    dict
        Results including per-level metrics and best configuration.
    """
    print("\n" + "=" * 70)
    print("  A1. ISOLATION FOREST — Anomaly Detection")
    print("=" * 70)

    if contamination_levels is None:
        contamination_levels = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    results_per_level = []
    best_f1 = 0
    best_level = None
    best_model = None
    best_preds = None

    for cont in contamination_levels:
        print(f"\n  Contamination = {cont:.4f} ...")

        iso_forest = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=cont,
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )

        start = time.time()
        iso_forest.fit(X)
        fit_time = round(time.time() - start, 2)

        # Predict: -1 = anomaly, 1 = normal
        preds = iso_forest.predict(X)
        scores = iso_forest.decision_function(X)

        # Convert to binary: anomaly (-1) -> 1 (fraud), normal (1) -> 0
        y_pred = (preds == -1).astype(int)

        # Evaluate against true labels
        f1 = round(f1_score(y, y_pred, zero_division=0), 4)
        precision = round(precision_score(y, y_pred, zero_division=0), 4)
        recall = round(recall_score(y, y_pred, zero_division=0), 4)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        level_result = {
            "contamination": cont,
            "fit_time_seconds": fit_time,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "anomalies_detected": int(y_pred.sum()),
            "actual_frauds": int(y.sum()),
        }
        results_per_level.append(level_result)

        print(f"    F1={f1:.4f}  Prec={precision:.4f}  Recall={recall:.4f}  "
              f"TP={tp}  FP={fp}  Detected={y_pred.sum()}")

        if f1 > best_f1:
            best_f1 = f1
            best_level = cont
            best_model = iso_forest
            best_preds = y_pred
            best_scores = scores

    # Save the best model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    print(f"\n  ★ Best contamination: {best_level} (F1={best_f1:.4f})")

    # --- Visualization: Anomaly Score Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Score distribution by class
    scores_legit = best_scores[y == 0]
    scores_fraud = best_scores[y == 1]

    axes[0].hist(scores_legit, bins=80, alpha=0.6, label="Legitimate", color="steelblue", density=True)
    axes[0].hist(scores_fraud, bins=80, alpha=0.7, label="Fraud", color="crimson", density=True)
    axes[0].set_xlabel("Anomaly Score (decision function)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Isolation Forest — Anomaly Score Distribution")
    axes[0].legend()
    axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Threshold")

    # Plot 2: F1 vs contamination level
    cont_vals = [r["contamination"] for r in results_per_level]
    f1_vals = [r["f1_score"] for r in results_per_level]
    prec_vals = [r["precision"] for r in results_per_level]
    rec_vals = [r["recall"] for r in results_per_level]

    axes[1].plot(cont_vals, f1_vals, "o-", color="green", linewidth=2, label="F1 Score")
    axes[1].plot(cont_vals, prec_vals, "s--", color="blue", alpha=0.7, label="Precision")
    axes[1].plot(cont_vals, rec_vals, "^--", color="red", alpha=0.7, label="Recall")
    axes[1].set_xlabel("Contamination Level")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Isolation Forest — Contamination Tuning")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "isolation_forest_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    return {
        "method": "Isolation Forest",
        "best_contamination": best_level,
        "best_f1_score": best_f1,
        "results_per_contamination": results_per_level,
        "figure": fig_path,
    }


# ===================================================================
#  A2. LOCAL OUTLIER FACTOR (LOF)
# ===================================================================

def run_lof(X: np.ndarray, y: np.ndarray,
            n_neighbors_list: list = None) -> dict:
    """
    Local Outlier Factor for anomaly/novelty detection.

    LOF measures the local deviation of the density of a given sample
    with respect to its neighbors. Samples that have a substantially
    lower density than their neighbors are considered outliers.

    We test multiple neighborhood sizes to find the optimal configuration.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels (for evaluation).
    n_neighbors_list : list of int
        Neighborhood sizes to evaluate.

    Returns
    -------
    dict
        LOF results with metrics for each configuration.
    """
    print("\n" + "=" * 70)
    print("  A2. LOCAL OUTLIER FACTOR (LOF) — Density-Based Anomaly Detection")
    print("=" * 70)

    if n_neighbors_list is None:
        n_neighbors_list = [5, 10, 20, 30, 50]

    # Estimate contamination from true fraud ratio
    true_contamination = y.sum() / len(y)
    print(f"  True fraud ratio: {true_contamination:.6f}")

    results_per_k = []
    best_f1 = 0
    best_k = None
    best_scores = None

    for k in n_neighbors_list:
        print(f"\n  n_neighbors = {k} ...")

        lof = LocalOutlierFactor(
            n_neighbors=k,
            contamination=true_contamination,
            metric="euclidean",
            n_jobs=-1,
            novelty=False,  # Use fit_predict for transductive mode
        )

        start = time.time()
        preds = lof.fit_predict(X)
        fit_time = round(time.time() - start, 2)

        # Get negative outlier factor (more negative = more anomalous)
        scores = lof.negative_outlier_factor_

        # Convert: -1 -> fraud (1), 1 -> legit (0)
        y_pred = (preds == -1).astype(int)

        f1 = round(f1_score(y, y_pred, zero_division=0), 4)
        precision = round(precision_score(y, y_pred, zero_division=0), 4)
        recall = round(recall_score(y, y_pred, zero_division=0), 4)

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        level_result = {
            "n_neighbors": k,
            "fit_time_seconds": fit_time,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "anomalies_detected": int(y_pred.sum()),
        }
        results_per_k.append(level_result)

        print(f"    F1={f1:.4f}  Prec={precision:.4f}  Recall={recall:.4f}  "
              f"TP={tp}  FP={fp}")

        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_scores = scores

    print(f"\n  ★ Best n_neighbors: {best_k} (F1={best_f1:.4f})")

    # --- Visualization: LOF Score Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: LOF score distribution
    lof_legit = best_scores[y == 0]
    lof_fraud = best_scores[y == 1]

    axes[0].hist(lof_legit, bins=80, alpha=0.6, label="Legitimate", color="steelblue", density=True)
    axes[0].hist(lof_fraud, bins=80, alpha=0.7, label="Fraud", color="crimson", density=True)
    axes[0].set_xlabel("Negative Outlier Factor")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"LOF Score Distribution (k={best_k})")
    axes[0].legend()
    axes[0].set_xlim(np.percentile(best_scores, 0.5), np.percentile(best_scores, 99.5))

    # Plot 2: F1 vs n_neighbors
    k_vals = [r["n_neighbors"] for r in results_per_k]
    f1_vals = [r["f1_score"] for r in results_per_k]
    prec_vals = [r["precision"] for r in results_per_k]
    rec_vals = [r["recall"] for r in results_per_k]

    axes[1].plot(k_vals, f1_vals, "o-", color="green", linewidth=2, label="F1 Score")
    axes[1].plot(k_vals, prec_vals, "s--", color="blue", alpha=0.7, label="Precision")
    axes[1].plot(k_vals, rec_vals, "^--", color="red", alpha=0.7, label="Recall")
    axes[1].set_xlabel("Number of Neighbors (k)")
    axes[1].set_ylabel("Score")
    axes[1].set_title("LOF — Neighborhood Size Tuning")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "lof_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # Train a novelty-detection LOF for future use (fit on legit only)
    print("\n  Training novelty-mode LOF on legitimate transactions only ...")
    X_legit = X[y == 0]
    lof_novelty = LocalOutlierFactor(
        n_neighbors=best_k,
        contamination=true_contamination,
        novelty=True,
        n_jobs=-1,
    )
    lof_novelty.fit(X_legit)
    joblib.dump(lof_novelty, os.path.join(MODELS_DIR, "lof_novelty.joblib"))
    print("  Novelty LOF saved to ml/models/lof_novelty.joblib")

    return {
        "method": "Local Outlier Factor",
        "best_n_neighbors": best_k,
        "best_f1_score": best_f1,
        "true_contamination_ratio": round(float(true_contamination), 6),
        "results_per_n_neighbors": results_per_k,
        "figure": fig_path,
    }


# ###################################################################
#  SECTION B: CLUSTERING ANALYSIS
# ###################################################################

# ===================================================================
#  B1. K-MEANS CLUSTERING
# ===================================================================

def run_kmeans(X: np.ndarray, y: np.ndarray,
               k_values: list = None) -> dict:
    """
    K-Means clustering with silhouette analysis.

    We cluster the transaction space with k=2,3,4 and evaluate
    cluster quality using:
      - Silhouette Score (cohesion vs separation, [-1, 1])
      - Calinski-Harabasz Index (higher = better defined clusters)
      - Davies-Bouldin Index (lower = better separation)

    We also analyze how fraud transactions distribute across clusters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels for fraud-distribution analysis.
    k_values : list of int
        Number of clusters to try.

    Returns
    -------
    dict
        Clustering results with quality metrics.
    """
    print("\n" + "=" * 70)
    print("  B1. K-MEANS CLUSTERING — Partitional Analysis")
    print("=" * 70)

    if k_values is None:
        k_values = [2, 3, 4]

    # Use a subsample for efficiency (full dataset is ~280K)
    max_samples = min(30000, X.shape[0])
    np.random.seed(42)
    idx = np.random.choice(X.shape[0], max_samples, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    print(f"  Using {max_samples:,} samples for clustering analysis")

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sub)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    results_per_k = []
    cluster_labels_all = {}

    for k in k_values:
        print(f"\n  K = {k} ...")

        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )

        start = time.time()
        labels = kmeans.fit_predict(X_sub)
        fit_time = round(time.time() - start, 2)

        # Quality metrics
        sil_score = round(silhouette_score(X_sub, labels), 4)
        ch_score = round(calinski_harabasz_score(X_sub, labels), 2)
        db_score = round(davies_bouldin_score(X_sub, labels), 4)
        inertia = round(float(kmeans.inertia_), 2)

        print(f"    Silhouette Score       : {sil_score}")
        print(f"    Calinski-Harabasz Index: {ch_score}")
        print(f"    Davies-Bouldin Index   : {db_score}")
        print(f"    Inertia (WCSS)         : {inertia}")
        print(f"    Fit time               : {fit_time}s")

        # Analyze fraud distribution per cluster
        cluster_fraud_info = []
        print(f"\n    Cluster fraud distribution:")
        for c in range(k):
            mask = labels == c
            cluster_size = int(mask.sum())
            cluster_frauds = int(y_sub[mask].sum())
            fraud_rate = round(cluster_frauds / cluster_size * 100, 4) if cluster_size > 0 else 0

            cluster_fraud_info.append({
                "cluster_id": c,
                "size": cluster_size,
                "fraud_count": cluster_frauds,
                "fraud_rate_pct": fraud_rate,
            })
            print(f"      Cluster {c}: {cluster_size:>6,} samples, "
                  f"{cluster_frauds:>4} frauds ({fraud_rate:.4f}%)")

        # Per-sample silhouette scores for silhouette plot
        sample_sil = silhouette_samples(X_sub, labels)

        results_per_k.append({
            "k": k,
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score,
            "inertia": inertia,
            "fit_time_seconds": fit_time,
            "cluster_fraud_distribution": cluster_fraud_info,
        })

        cluster_labels_all[k] = (labels, kmeans, sample_sil)

    # --- Visualization: Multi-panel K-Means Analysis ---
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0"]

    for i, k in enumerate(k_values):
        labels, kmeans, sample_sil = cluster_labels_all[k]

        # Scatter plots (top row)
        ax = fig.add_subplot(gs[0, i])
        for c in range(k):
            mask = labels == c
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[c], alpha=0.3,
                       s=5, label=f"Cluster {c}")
        # Overlay fraud points
        fraud_mask = y_sub == 1
        ax.scatter(X_2d[fraud_mask, 0], X_2d[fraud_mask, 1], c="red",
                   marker="x", s=25, alpha=0.8, label="Fraud", zorder=5)

        centers_2d = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black",
                   marker="*", s=200, edgecolors="white", linewidths=1.5,
                   zorder=10, label="Centroids")
        ax.set_title(f"K-Means (k={k})", fontsize=12, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="upper right")

        # Silhouette plots (bottom row)
        ax2 = fig.add_subplot(gs[1, i])
        y_lower = 10
        for c in range(k):
            cluster_sil = np.sort(sample_sil[labels == c])
            size = cluster_sil.shape[0]
            y_upper = y_lower + size
            ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                              facecolor=colors[c], alpha=0.7)
            ax2.text(-0.05, y_lower + 0.5 * size, str(c), fontsize=10, fontweight="bold")
            y_lower = y_upper + 10

        ax2.axvline(x=results_per_k[i]["silhouette_score"], color="red",
                    linestyle="--", linewidth=1.5)
        ax2.set_title(f"Silhouette Plot (k={k}, avg={results_per_k[i]['silhouette_score']:.3f})",
                      fontsize=11)
        ax2.set_xlabel("Silhouette Coefficient")
        ax2.set_ylabel("Cluster")
        ax2.set_yticks([])

    fig.suptitle("K-Means Clustering Analysis", fontsize=16, fontweight="bold", y=1.01)
    fig_path = os.path.join(FIGURES_DIR, "kmeans_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    # --- Elbow Plot ---
    fig_elbow, ax_elbow = plt.subplots(1, 1, figsize=(8, 5))
    k_range = range(2, 9)
    inertias = []
    sil_scores_range = []
    for kk in k_range:
        km = KMeans(n_clusters=kk, init="k-means++", n_init=5, random_state=42)
        km.fit(X_sub)
        inertias.append(km.inertia_)
        sil_scores_range.append(silhouette_score(X_sub, km.labels_))

    ax_elbow.plot(list(k_range), inertias, "bo-", linewidth=2, label="Inertia (WCSS)")
    ax_elbow.set_xlabel("Number of Clusters (k)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method for Optimal k")
    ax_elbow.grid(True, alpha=0.3)

    ax2 = ax_elbow.twinx()
    ax2.plot(list(k_range), sil_scores_range, "rs--", linewidth=2, label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score")

    lines1, labels1 = ax_elbow.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_elbow.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig_elbow_path = os.path.join(FIGURES_DIR, "kmeans_elbow.png")
    plt.savefig(fig_elbow_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_elbow_path}")

    # Find best k
    best_k = k_values[np.argmax([r["silhouette_score"] for r in results_per_k])]
    print(f"\n  ★ Best k={best_k} (highest silhouette score)")

    return {
        "method": "K-Means Clustering",
        "best_k": best_k,
        "results_per_k": results_per_k,
        "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "samples_used": max_samples,
        "figures": [fig_path, fig_elbow_path],
    }


# ===================================================================
#  B2. DBSCAN CLUSTERING
# ===================================================================

def run_dbscan(X: np.ndarray, y: np.ndarray) -> dict:
    """
    DBSCAN density-based clustering with epsilon neighborhood analysis.

    DBSCAN finds clusters of arbitrary shape and identifies noise points
    (points not belonging to any cluster), making it naturally suited
    for anomaly detection — noise points may correspond to fraudulent
    transactions.

    We use the k-distance graph (sorted k-nearest-neighbor distances)
    to estimate optimal epsilon, then run DBSCAN with multiple epsilon values.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels for evaluation.

    Returns
    -------
    dict
        DBSCAN results.
    """
    print("\n" + "=" * 70)
    print("  B2. DBSCAN — Density-Based Clustering")
    print("=" * 70)

    # Subsample for tractability
    max_samples = min(15000, X.shape[0])
    np.random.seed(42)
    idx = np.random.choice(X.shape[0], max_samples, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    print(f"  Using {max_samples:,} samples")

    # --- Epsilon estimation via k-distance graph ---
    print("  Computing k-distance graph for epsilon estimation ...")
    min_samples = 5  # standard default
    nn = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nn.fit(X_sub)
    distances, _ = nn.kneighbors(X_sub)
    k_distances = np.sort(distances[:, -1])

    # Estimate the "elbow" — use percentiles of the k-distance distribution
    eps_candidates = [
        round(float(np.percentile(k_distances, p)), 3)
        for p in [85, 90, 95, 97]
    ]
    # Remove duplicates and sort
    eps_candidates = sorted(set(eps_candidates))
    print(f"  Epsilon candidates: {eps_candidates}")

    # --- K-Distance Plot ---
    fig_kdist, ax_kdist = plt.subplots(1, 1, figsize=(8, 5))
    ax_kdist.plot(k_distances, linewidth=1.5, color="steelblue")
    for eps in eps_candidates:
        ax_kdist.axhline(y=eps, linestyle="--", alpha=0.5, label=f"ε={eps}")
    ax_kdist.set_xlabel("Points (sorted by distance)")
    ax_kdist.set_ylabel(f"{min_samples}-Distance")
    ax_kdist.set_title(f"k-Distance Graph (k={min_samples})")
    ax_kdist.legend()
    ax_kdist.grid(True, alpha=0.3)

    kdist_path = os.path.join(FIGURES_DIR, "dbscan_kdistance.png")
    plt.savefig(kdist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {kdist_path}")

    # --- Run DBSCAN with each epsilon ---
    results_per_eps = []
    best_fraud_capture = 0
    best_eps = None
    best_labels = None

    for eps in eps_candidates:
        print(f"\n  ε = {eps}, min_samples = {min_samples} ...")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)

        start = time.time()
        labels = dbscan.fit_predict(X_sub)
        fit_time = round(time.time() - start, 2)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = round(n_noise / len(labels) * 100, 2)

        # Fraud in noise points
        noise_mask = labels == -1
        fraud_in_noise = int(y_sub[noise_mask].sum())
        total_fraud = int(y_sub.sum())
        fraud_capture_rate = round(fraud_in_noise / total_fraud * 100, 2) if total_fraud > 0 else 0

        # Cluster analysis
        cluster_info = []
        for c in sorted(set(labels)):
            c_mask = labels == c
            c_size = int(c_mask.sum())
            c_fraud = int(y_sub[c_mask].sum())
            c_fraud_rate = round(c_fraud / c_size * 100, 4) if c_size > 0 else 0
            cluster_info.append({
                "cluster_id": int(c),
                "label": "noise" if c == -1 else f"cluster_{c}",
                "size": c_size,
                "fraud_count": c_fraud,
                "fraud_rate_pct": c_fraud_rate,
            })

        # Silhouette score (only if >1 cluster and not all noise)
        sil = None
        if n_clusters >= 2 and n_noise < len(labels) - 1:
            non_noise = labels != -1
            if len(set(labels[non_noise])) >= 2:
                sil = round(silhouette_score(X_sub[non_noise], labels[non_noise]), 4)

        eps_result = {
            "epsilon": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_percentage": noise_pct,
            "fraud_in_noise": fraud_in_noise,
            "fraud_capture_rate_pct": fraud_capture_rate,
            "silhouette_score": sil,
            "fit_time_seconds": fit_time,
            "cluster_details": cluster_info,
        }
        results_per_eps.append(eps_result)

        print(f"    Clusters: {n_clusters}, Noise: {n_noise} ({noise_pct}%), "
              f"Fraud in noise: {fraud_in_noise}/{total_fraud} ({fraud_capture_rate}%)")

        if fraud_capture_rate > best_fraud_capture:
            best_fraud_capture = fraud_capture_rate
            best_eps = eps
            best_labels = labels

    print(f"\n  ★ Best ε={best_eps} (fraud capture in noise: {best_fraud_capture}%)")

    # --- Visualization: DBSCAN clusters with best eps ---
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sub)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Cluster assignment
    unique_labels = set(best_labels)
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(sorted(unique_labels)):
        mask = best_labels == label
        if label == -1:
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c="gray", alpha=0.3,
                            s=5, label="Noise")
        else:
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap_colors[i]], alpha=0.4,
                            s=8, label=f"Cluster {label}")

    fraud_mask = y_sub == 1
    axes[0].scatter(X_2d[fraud_mask, 0], X_2d[fraud_mask, 1], c="red",
                    marker="x", s=30, alpha=0.8, label="Fraud", zorder=5)
    axes[0].set_title(f"DBSCAN (ε={best_eps}, min_samples={min_samples})", fontweight="bold")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=7, loc="upper right")

    # Plot 2: Fraud capture vs epsilon
    eps_vals = [r["epsilon"] for r in results_per_eps]
    capture_vals = [r["fraud_capture_rate_pct"] for r in results_per_eps]
    noise_vals = [r["noise_percentage"] for r in results_per_eps]

    axes[1].bar(range(len(eps_vals)), capture_vals, alpha=0.7, color="crimson",
                label="Fraud Capture Rate (%)")
    ax2 = axes[1].twinx()
    ax2.plot(range(len(eps_vals)), noise_vals, "bo-", linewidth=2, label="Noise %")
    axes[1].set_xticks(range(len(eps_vals)))
    axes[1].set_xticklabels([str(e) for e in eps_vals], rotation=45)
    axes[1].set_xlabel("Epsilon (ε)")
    axes[1].set_ylabel("Fraud Capture Rate (%)")
    ax2.set_ylabel("Noise Percentage (%)")
    axes[1].set_title("DBSCAN — Fraud Capture vs Epsilon")

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "dbscan_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    return {
        "method": "DBSCAN Clustering",
        "best_epsilon": best_eps,
        "best_fraud_capture_pct": best_fraud_capture,
        "min_samples": min_samples,
        "results_per_epsilon": results_per_eps,
        "samples_used": max_samples,
        "figures": [kdist_path, fig_path],
    }


# ###################################################################
#  SECTION C: PATTERN ANALYSIS
# ###################################################################

# ===================================================================
#  C1. TRANSACTION PATTERN ANALYSIS
# ===================================================================

def run_pattern_analysis(df: pd.DataFrame) -> dict:
    """
    Comprehensive transaction pattern mining:
      1. Amount-based patterns (distributions, statistical tests)
      2. Temporal patterns (time-of-day, periodic behavior)
      3. Feature correlation patterns
      4. Extreme-value analysis

    Parameters
    ----------
    df : pd.DataFrame
        The scaled dataframe with all features.

    Returns
    -------
    dict
        Pattern analysis findings.
    """
    print("\n" + "=" * 70)
    print("  C1. TRANSACTION PATTERN ANALYSIS")
    print("=" * 70)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    # -----------------------------------------------------------------
    # 1. AMOUNT-BASED PATTERNS
    # -----------------------------------------------------------------
    print("\n  1. Amount-Based Patterns")
    print("  " + "-" * 40)

    amount_patterns = {
        "legitimate": {
            "mean": round(float(legit["Amount"].mean()), 4),
            "median": round(float(legit["Amount"].median()), 4),
            "std": round(float(legit["Amount"].std()), 4),
            "min": round(float(legit["Amount"].min()), 4),
            "max": round(float(legit["Amount"].max()), 4),
            "q25": round(float(legit["Amount"].quantile(0.25)), 4),
            "q75": round(float(legit["Amount"].quantile(0.75)), 4),
            "q95": round(float(legit["Amount"].quantile(0.95)), 4),
            "q99": round(float(legit["Amount"].quantile(0.99)), 4),
        },
        "fraud": {
            "mean": round(float(fraud["Amount"].mean()), 4),
            "median": round(float(fraud["Amount"].median()), 4),
            "std": round(float(fraud["Amount"].std()), 4),
            "min": round(float(fraud["Amount"].min()), 4),
            "max": round(float(fraud["Amount"].max()), 4),
            "q25": round(float(fraud["Amount"].quantile(0.25)), 4),
            "q75": round(float(fraud["Amount"].quantile(0.75)), 4),
            "q95": round(float(fraud["Amount"].quantile(0.95)), 4),
            "q99": round(float(fraud["Amount"].quantile(0.99)), 4),
        },
    }

    for cls in ["legitimate", "fraud"]:
        print(f"    {cls.title()} Transactions:")
        for stat, val in amount_patterns[cls].items():
            print(f"      {stat:>8s}: {val}")

    # -----------------------------------------------------------------
    # 2. TEMPORAL PATTERNS
    # -----------------------------------------------------------------
    print("\n  2. Temporal Patterns")
    print("  " + "-" * 40)

    # Time in hours from start
    df_time = df.copy()
    # Time column is already scaled, so we use the original for temporal analysis
    # We'll work with the scaled values and analyze their distribution

    temporal_patterns = {
        "legitimate_time_mean": round(float(legit["Time"].mean()), 4),
        "legitimate_time_std": round(float(legit["Time"].std()), 4),
        "fraud_time_mean": round(float(fraud["Time"].mean()), 4),
        "fraud_time_std": round(float(fraud["Time"].std()), 4),
    }

    # Binned time analysis (quartiles)
    time_bins = pd.qcut(df["Time"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    time_fraud_dist = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        mask = time_bins == q
        total_in_bin = int(mask.sum())
        fraud_in_bin = int(df.loc[mask, "Class"].sum())
        rate = round(fraud_in_bin / total_in_bin * 100, 4) if total_in_bin > 0 else 0
        time_fraud_dist[q] = {
            "total": total_in_bin,
            "fraud_count": fraud_in_bin,
            "fraud_rate_pct": rate,
        }
        print(f"    Time {q}: {total_in_bin:>7,} txns, "
              f"{fraud_in_bin:>4} frauds ({rate:.4f}%)")

    temporal_patterns["time_quartile_fraud_distribution"] = time_fraud_dist

    # -----------------------------------------------------------------
    # 3. FEATURE CORRELATION PATTERNS
    # -----------------------------------------------------------------
    print("\n  3. Feature Correlation Patterns")
    print("  " + "-" * 40)

    # Find features most correlated with fraud
    feature_cols = [c for c in df.columns if c != "Class"]
    correlations = df[feature_cols + ["Class"]].corr()["Class"].drop("Class")
    top_positive = correlations.nlargest(5)
    top_negative = correlations.nsmallest(5)

    correlation_patterns = {
        "top_positive_correlations": {k: round(v, 4) for k, v in top_positive.items()},
        "top_negative_correlations": {k: round(v, 4) for k, v in top_negative.items()},
    }

    print("    Top positively correlated features with Fraud:")
    for feat, corr in top_positive.items():
        print(f"      {feat:>8s}: {corr:+.4f}")
    print("    Top negatively correlated features with Fraud:")
    for feat, corr in top_negative.items():
        print(f"      {feat:>8s}: {corr:+.4f}")

    # -----------------------------------------------------------------
    # 4. FEATURE DISTRIBUTION DIVERGENCE (Fraud vs Legit)
    # -----------------------------------------------------------------
    print("\n  4. Feature Distribution Divergence (KS-Statistic)")
    print("  " + "-" * 40)

    from scipy.stats import ks_2samp

    ks_results = {}
    pca_features = [f"V{i}" for i in range(1, 29)]
    for feat in pca_features:
        stat, pval = ks_2samp(legit[feat].values, fraud[feat].values)
        ks_results[feat] = {"ks_statistic": round(stat, 4), "p_value": round(pval, 6)}

    # Sort by KS statistic (most divergent first)
    sorted_ks = sorted(ks_results.items(), key=lambda x: x[1]["ks_statistic"], reverse=True)
    print("    Most divergent features (Kolmogorov-Smirnov test):")
    for feat, vals in sorted_ks[:10]:
        print(f"      {feat:>4s}: KS={vals['ks_statistic']:.4f}  p={vals['p_value']:.6f}")

    feature_divergence = {feat: vals for feat, vals in sorted_ks}

    # -----------------------------------------------------------------
    # 5. EXTREME VALUE / OUTLIER PATTERN ANALYSIS
    # -----------------------------------------------------------------
    print("\n  5. Extreme Value Analysis")
    print("  " + "-" * 40)

    extreme_patterns = {}
    for feat in ["V1", "V2", "V3", "V4", "V14", "V17", "Amount"]:
        q99 = float(df[feat].quantile(0.99))
        q01 = float(df[feat].quantile(0.01))
        extreme_high = df[df[feat] > q99]
        extreme_low = df[df[feat] < q01]

        fraud_rate_high = round(
            extreme_high["Class"].mean() * 100, 4
        ) if len(extreme_high) > 0 else 0
        fraud_rate_low = round(
            extreme_low["Class"].mean() * 100, 4
        ) if len(extreme_low) > 0 else 0

        extreme_patterns[feat] = {
            "q01_threshold": round(q01, 4),
            "q99_threshold": round(q99, 4),
            "extreme_high_fraud_rate_pct": fraud_rate_high,
            "extreme_low_fraud_rate_pct": fraud_rate_low,
        }
        print(f"    {feat:>8s}: High-extreme fraud rate={fraud_rate_high:.2f}%,  "
              f"Low-extreme fraud rate={fraud_rate_low:.2f}%")

    # -----------------------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------------------
    print("\n  Generating pattern analysis visualizations ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (0,0) Amount distribution
    axes[0, 0].hist(legit["Amount"], bins=80, alpha=0.6, label="Legitimate",
                    color="steelblue", density=True)
    axes[0, 0].hist(fraud["Amount"], bins=80, alpha=0.7, label="Fraud",
                    color="crimson", density=True)
    axes[0, 0].set_xlabel("Scaled Amount")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Amount Distribution by Class")
    axes[0, 0].legend()
    axes[0, 0].set_xlim(-3, 10)

    # (0,1) Time distribution
    axes[0, 1].hist(legit["Time"], bins=80, alpha=0.6, label="Legitimate",
                    color="steelblue", density=True)
    axes[0, 1].hist(fraud["Time"], bins=80, alpha=0.7, label="Fraud",
                    color="crimson", density=True)
    axes[0, 1].set_xlabel("Scaled Time")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Time Distribution by Class")
    axes[0, 1].legend()

    # (0,2) Feature correlations bar chart
    top_feats = pd.concat([top_positive, top_negative]).sort_values()
    colors_bar = ["crimson" if v > 0 else "steelblue" for v in top_feats.values]
    axes[0, 2].barh(range(len(top_feats)), top_feats.values, color=colors_bar, alpha=0.8)
    axes[0, 2].set_yticks(range(len(top_feats)))
    axes[0, 2].set_yticklabels(top_feats.index)
    axes[0, 2].set_xlabel("Correlation with Fraud")
    axes[0, 2].set_title("Top Feature Correlations with Fraud")
    axes[0, 2].axvline(x=0, color="black", linewidth=0.5)

    # (1,0) Top-3 most divergent feature distributions
    top3_feats = [feat for feat, _ in sorted_ks[:3]]
    for feat in top3_feats:
        axes[1, 0].hist(fraud[feat], bins=50, alpha=0.5, label=f"Fraud-{feat}", density=True)
    axes[1, 0].set_xlabel("Feature Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Top 3 Most Divergent Features (Fraud)")
    axes[1, 0].legend()

    # (1,1) KS statistic bar chart
    ks_feats = [f[0] for f in sorted_ks[:15]]
    ks_vals = [f[1]["ks_statistic"] for f in sorted_ks[:15]]
    axes[1, 1].barh(range(len(ks_feats)), ks_vals, color="darkorange", alpha=0.8)
    axes[1, 1].set_yticks(range(len(ks_feats)))
    axes[1, 1].set_yticklabels(ks_feats)
    axes[1, 1].set_xlabel("KS Statistic")
    axes[1, 1].set_title("Feature Distribution Divergence (KS Test)")
    axes[1, 1].invert_yaxis()

    # (1,2) Fraud rate by time quartile
    quartiles = list(time_fraud_dist.keys())
    fraud_rates = [time_fraud_dist[q]["fraud_rate_pct"] for q in quartiles]
    axes[1, 2].bar(quartiles, fraud_rates, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"],
                   alpha=0.8)
    axes[1, 2].set_xlabel("Time Quartile")
    axes[1, 2].set_ylabel("Fraud Rate (%)")
    axes[1, 2].set_title("Fraud Rate by Time Period")
    for i, (q, r) in enumerate(zip(quartiles, fraud_rates)):
        axes[1, 2].text(i, r + 0.005, f"{r:.3f}%", ha="center", fontsize=9)

    plt.suptitle("Transaction Pattern Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "pattern_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    return {
        "method": "Transaction Pattern Analysis",
        "amount_patterns": amount_patterns,
        "temporal_patterns": temporal_patterns,
        "feature_correlations": correlation_patterns,
        "feature_divergence_ks": {k: v for k, v in sorted_ks[:15]},
        "extreme_value_patterns": extreme_patterns,
        "figure": fig_path,
    }


# ===================================================================
#  C2. ANOMALY SCORING COMPARISON
# ===================================================================

def run_anomaly_comparison(X: np.ndarray, y: np.ndarray,
                           iso_results: dict, lof_results: dict) -> dict:
    """
    Quantitative comparison of anomaly detection methods.

    Compares Isolation Forest and LOF across multiple metrics and
    generates a unified comparison visualization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels.
    iso_results : dict
        Results from Isolation Forest.
    lof_results : dict
        Results from LOF.

    Returns
    -------
    dict
        Comparison summary.
    """
    print("\n" + "=" * 70)
    print("  C2. ANOMALY DETECTION METHOD COMPARISON")
    print("=" * 70)

    methods = []

    # Extract best results for each method
    if iso_results:
        best_iso = max(iso_results["results_per_contamination"],
                       key=lambda x: x["f1_score"])
        methods.append({
            "method": "Isolation Forest",
            "f1_score": best_iso["f1_score"],
            "precision": best_iso["precision"],
            "recall": best_iso["recall"],
            "true_positives": best_iso["true_positives"],
            "false_positives": best_iso["false_positives"],
            "best_param": f"contamination={iso_results['best_contamination']}",
        })

    if lof_results:
        best_lof = max(lof_results["results_per_n_neighbors"],
                       key=lambda x: x["f1_score"])
        methods.append({
            "method": "Local Outlier Factor",
            "f1_score": best_lof["f1_score"],
            "precision": best_lof["precision"],
            "recall": best_lof["recall"],
            "true_positives": best_lof["true_positives"],
            "false_positives": best_lof["false_positives"],
            "best_param": f"n_neighbors={lof_results['best_n_neighbors']}",
        })

    # Rank methods
    methods_sorted = sorted(methods, key=lambda x: x["f1_score"], reverse=True)

    print("\n  ANOMALY DETECTION RANKING:")
    print(f"  {'Rank':<5} {'Method':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} {'TP':>6} {'FP':>6}")
    print(f"  {'-'*5} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for i, m in enumerate(methods_sorted, 1):
        print(f"  {i:<5} {m['method']:<25} {m['f1_score']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['true_positives']:>6} {m['false_positives']:>6}")

    # --- Comparison visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Grouped bar chart of metrics
    method_names = [m["method"] for m in methods_sorted]
    x = np.arange(len(method_names))
    width = 0.25

    f1_vals = [m["f1_score"] for m in methods_sorted]
    prec_vals = [m["precision"] for m in methods_sorted]
    rec_vals = [m["recall"] for m in methods_sorted]

    bars1 = axes[0].bar(x - width, f1_vals, width, label="F1 Score", color="#2196F3", alpha=0.8)
    bars2 = axes[0].bar(x, prec_vals, width, label="Precision", color="#4CAF50", alpha=0.8)
    bars3 = axes[0].bar(x + width, rec_vals, width, label="Recall", color="#FF5722", alpha=0.8)

    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Anomaly Detection — Method Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_names, rotation=15)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.2, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    # Plot 2: TP/FP comparison
    tp_vals = [m["true_positives"] for m in methods_sorted]
    fp_vals = [m["false_positives"] for m in methods_sorted]

    axes[1].bar(x - 0.15, tp_vals, 0.3, label="True Positives", color="green", alpha=0.7)
    axes[1].bar(x + 0.15, fp_vals, 0.3, label="False Positives", color="red", alpha=0.7)
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Anomaly Detection — TP vs FP")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_names, rotation=15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "anomaly_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    return {
        "method": "Anomaly Detection Comparison",
        "ranking": methods_sorted,
        "best_method": methods_sorted[0]["method"] if methods_sorted else None,
        "best_f1": methods_sorted[0]["f1_score"] if methods_sorted else None,
        "figure": fig_path,
    }


# ###################################################################
#  MASTER MINING PIPELINE
# ###################################################################

def run_mining_pipeline(X: np.ndarray, y: np.ndarray,
                        df_scaled: pd.DataFrame = None,
                        feature_names: list = None) -> dict:
    """
    Execute the full data mining pipeline:
      A1. Isolation Forest
      A2. Local Outlier Factor
      B1. K-Means Clustering
      B2. DBSCAN Clustering
      C1. Pattern Analysis
      C2. Method Comparison

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (full dataset, not just train or test).
    y : np.ndarray
        True labels.
    df_scaled : pd.DataFrame
        Scaled dataframe (for pattern analysis).
    feature_names : list
        Feature column names.

    Returns
    -------
    dict
        Comprehensive mining results.
    """
    ensure_dirs()
    start_total = time.time()

    all_results = OrderedDict()

    print("\n" + "#" * 70)
    print("#  CREDIT CARD FRAUD DETECTION — DATA MINING ANALYSIS")
    print("#" * 70)

    # --- A. Anomaly Detection ---
    print("\n" + "#" * 70)
    print("#  SECTION A: ANOMALY DETECTION")
    print("#" * 70)

    try:
        iso_results = run_isolation_forest(X, y)
        all_results["isolation_forest"] = iso_results
    except Exception as e:
        print(f"  ERROR in Isolation Forest: {e}")
        iso_results = None

    try:
        lof_results = run_lof(X, y)
        all_results["lof"] = lof_results
    except Exception as e:
        print(f"  ERROR in LOF: {e}")
        lof_results = None

    # --- B. Clustering ---
    print("\n" + "#" * 70)
    print("#  SECTION B: CLUSTERING ANALYSIS")
    print("#" * 70)

    try:
        kmeans_results = run_kmeans(X, y)
        all_results["kmeans"] = kmeans_results
    except Exception as e:
        print(f"  ERROR in K-Means: {e}")

    try:
        dbscan_results = run_dbscan(X, y)
        all_results["dbscan"] = dbscan_results
    except Exception as e:
        print(f"  ERROR in DBSCAN: {e}")

    # --- C. Pattern Analysis ---
    print("\n" + "#" * 70)
    print("#  SECTION C: PATTERN & COMPARISON ANALYSIS")
    print("#" * 70)

    if df_scaled is not None:
        try:
            pattern_results = run_pattern_analysis(df_scaled)
            all_results["pattern_analysis"] = pattern_results
        except Exception as e:
            print(f"  ERROR in Pattern Analysis: {e}")
    else:
        print("  [SKIP] Pattern analysis requires df_scaled (DataFrame format)")

    try:
        comparison = run_anomaly_comparison(X, y, iso_results, lof_results)
        all_results["anomaly_comparison"] = comparison
    except Exception as e:
        print(f"  ERROR in Anomaly Comparison: {e}")

    # --- Summary ---
    total_time = round(time.time() - start_total, 2)
    all_results["pipeline_metadata"] = {
        "total_runtime_seconds": total_time,
        "dataset_size": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "fraud_count": int(y.sum()),
        "fraud_ratio": round(float(y.mean()), 6),
    }

    # Save comprehensive results JSON
    results_path = os.path.join(MODELS_DIR, "mining_results.json")

    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, OrderedDict):
            return dict(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=make_serializable)

    print("\n" + "=" * 70)
    print(f"  DATA MINING PIPELINE COMPLETE")
    print(f"  Total runtime: {total_time:.1f}s")
    print(f"  Results saved: {results_path}")
    print(f"  Figures saved: {FIGURES_DIR}")
    print("=" * 70)

    return all_results


# ===================================================================
# STANDALONE EXECUTION
# ===================================================================

if __name__ == "__main__":
    # Load preprocessed data or run preprocessing
    preprocessed_path = os.path.join(MODELS_DIR, "preprocessed_data.joblib")

    if os.path.isfile(preprocessed_path):
        print("[train_mining] Loading preprocessed data ...")
        data = joblib.load(preprocessed_path)
        X = np.vstack([data["X_train"], data["X_test"]])
        y = np.concatenate([data["y_train"], data["y_test"]])
        feature_names = data["feature_names"]
    else:
        print("[train_mining] Preprocessed data not found. Running preprocessing ...")
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
        from preprocessing.preprocess import run_preprocessing

        try:
            result = run_preprocessing()
            X = np.vstack([result["X_train"], result["X_test"]])
            y = np.concatenate([result["y_train"], result["y_test"]])
            feature_names = result["feature_names"]
        except FileNotFoundError as e:
            print(f"[train_mining] ✗ {e}")
            sys.exit(1)

    # Try to load the scaled DataFrame for pattern analysis
    df_scaled = None
    try:
        data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
        if os.path.isfile(data_path):
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
            from preprocessing.preprocess import load_data, scale_features
            df_raw = load_data(data_path)
            df_scaled = scale_features(df_raw)
    except Exception:
        print("[train_mining] Could not load DataFrame for pattern analysis")

    results = run_mining_pipeline(X, y, df_scaled=df_scaled, feature_names=feature_names)
    print("\n[train_mining] ✓ Data mining pipeline completed successfully!")
