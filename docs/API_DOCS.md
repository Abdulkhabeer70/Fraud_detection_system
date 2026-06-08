# API Documentation

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### 1. Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "models_loaded": true,
  "database_connected": true
}
```

---

### 2. Predict Transaction

```
POST /predict
```

**Request Body:**
```json
{
  "Amount": 149.62,
  "V1": -1.36,
  "V2": -0.07,
  "V3": 2.54,
  "V4": 1.38,
  "V5": -0.34,
  "V6": 0.46,
  "V7": 0.24,
  "V8": 0.10,
  "V9": 0.36,
  "V10": 0.09,
  "V11": -0.55,
  "V12": -0.62,
  "V13": -0.99,
  "V14": -0.31,
  "V15": 1.47,
  "V16": -0.47,
  "V17": 0.21,
  "V18": 0.03,
  "V19": 0.40,
  "V20": 0.25,
  "V21": -0.02,
  "V22": 0.28,
  "V23": -0.11,
  "V24": 0.07,
  "V25": 0.13,
  "V26": -0.19,
  "V27": 0.13,
  "V28": -0.02
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "Legitimate",
  "probability": 0.03,
  "risk_level": "Low",
  "model_used": "xgboost"
}
```

**Risk Levels:**
- `Low`: probability < 0.3
- `Medium`: 0.3 ≤ probability < 0.7
- `High`: probability ≥ 0.7

---

### 3. Dataset Statistics

```
GET /stats
```

**Response:**
```json
{
  "total_transactions": 284807,
  "fraud_count": 492,
  "legit_count": 284315,
  "fraud_percentage": 0.17,
  "amount_stats": {
    "mean": 88.35,
    "std": 250.12,
    "min": 0.0,
    "max": 25691.16,
    "median": 22.0
  }
}
```

---

### 4. Model Performance

```
GET /model-performance
```

**Response:**
```json
{
  "logistic_regression": {
    "accuracy": 0.97,
    "precision": 0.85,
    "recall": 0.62,
    "f1_score": 0.72,
    "roc_auc": 0.95
  },
  "random_forest": {
    "accuracy": 0.99,
    "precision": 0.93,
    "recall": 0.78,
    "f1_score": 0.85,
    "roc_auc": 0.98
  },
  "xgboost": {
    "accuracy": 0.99,
    "precision": 0.95,
    "recall": 0.82,
    "f1_score": 0.88,
    "roc_auc": 0.99
  }
}
```

---

### 5. Data Mining Results

```
GET /mining-results
```

**Response:**
```json
{
  "anomaly_detection": {
    "isolation_forest": {
      "anomalies_detected": 285,
      "precision": 0.45,
      "recall": 0.65,
      "contamination": 0.001
    },
    "lof": {
      "anomalies_detected": 310,
      "precision": 0.40,
      "recall": 0.70,
      "n_neighbors": 20
    }
  },
  "clustering": {
    "kmeans": {
      "n_clusters": 2,
      "silhouette_score": 0.45,
      "cluster_sizes": [280000, 4807],
      "fraud_distribution_per_cluster": {"0": 100, "1": 392}
    },
    "dbscan": {
      "n_clusters": 5,
      "noise_points": 1200,
      "silhouette_score": 0.38
    }
  },
  "pattern_analysis": {
    "high_amount_fraud_pct": 0.25,
    "avg_fraud_amount": 122.21,
    "avg_legit_amount": 88.29,
    "peak_fraud_hours": [2, 3, 11]
  }
}
```

---

### 6. Prediction History

```
GET /predictions/history?limit=50
```

**Response:**
```json
{
  "predictions": [
    {
      "id": 1,
      "amount": 149.62,
      "prediction": 0,
      "prediction_label": "Legitimate",
      "probability": 0.03,
      "risk_level": "Low",
      "model_used": "xgboost",
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "total": 1
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error description"
}
```

| Status Code | Meaning |
|-------------|---------|
| `400` | Bad request (invalid input) |
| `404` | Resource not found |
| `503` | Service unavailable (models not loaded) |
| `500` | Internal server error |
