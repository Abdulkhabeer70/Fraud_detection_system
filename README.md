# 🔍 Credit Card Fraud Detection

**Using Data Mining and Artificial Intelligence**

A semester project that applies data mining techniques, machine learning algorithms, and explainable AI to detect fraudulent credit card transactions. Built with a modern tech stack featuring a React dashboard, FastAPI backend, and a comprehensive ML/data mining pipeline.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Data Mining Techniques](#data-mining-techniques)
- [Machine Learning Models](#machine-learning-models)
- [Screenshots](#screenshots)

---

## 📖 Project Overview

Credit card fraud is a growing concern in the financial industry. This project demonstrates how data mining and AI techniques can be applied to detect fraudulent transactions in real-time. The system analyzes transaction patterns using both supervised ML models and unsupervised data mining algorithms.

### Dataset

This project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
- **284,807** transactions over 2 days
- **492** fraud cases (0.172% of all transactions)
- Features **V1–V28** are PCA-transformed (anonymized)
- **Time** and **Amount** are original features

> A synthetic dataset generator is included for development/testing without the full Kaggle dataset.

---

## ✨ Key Features

### Data Mining (Primary Focus)
- **Isolation Forest** — Tree-based anomaly detection
- **Local Outlier Factor (LOF)** — Density-based anomaly detection
- **K-Means Clustering** — Partition-based transaction grouping
- **DBSCAN Clustering** — Density-based spatial clustering
- **Transaction Pattern Analysis** — Temporal and amount-based pattern discovery
- **Anomaly Scoring Comparison** — Quantitative comparison across methods

### Machine Learning
- **Logistic Regression** — Baseline linear classifier
- **Random Forest** — Ensemble bagging classifier
- **XGBoost** — Gradient boosting classifier

### Explainable AI
- **SHAP** feature importance visualization
- Feature contribution analysis per prediction

### Web Application
- Real-time fraud prediction API
- Interactive analytics dashboard
- Transaction prediction form with risk scoring
- Comprehensive data mining visualizations

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React, Vite, Tailwind CSS, Recharts, Axios |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy |
| **ML/Data Mining** | Scikit-learn, XGBoost, Imbalanced-learn, SHAP |
| **Database** | SQLite |
| **Visualization** | Matplotlib, Seaborn, Recharts |
| **Data Processing** | Pandas, NumPy, SciPy |

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Dashboard, Predict, Analytics
│   │   ├── services/          # API client
│   │   └── App.jsx            # Main app with routing
│   └── package.json
│
├── backend/                   # FastAPI backend
│   └── app/
│       ├── main.py            # App entry point
│       ├── routes/            # API endpoints
│       ├── models/            # Pydantic schemas
│       ├── services/          # Business logic
│       └── database.py        # SQLite ORM
│
├── ml/                        # ML & Data Mining pipeline
│   ├── preprocessing/         # Data loading, scaling, SMOTE
│   ├── training/              # Model training & data mining
│   ├── evaluation/            # Metrics & evaluation
│   ├── explainability/        # SHAP analysis
│   ├── models/                # Saved trained models (.joblib)
│   └── figures/               # Generated plots (.png)
│
├── data/                      # Dataset directory
│   ├── creditcard.csv         # Kaggle dataset (not in Git)
│   └── generate_sample.py     # Synthetic data generator
│
├── database/                  # SQLite database
├── notebooks/                 # Jupyter notebooks for EDA
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare the Dataset

**Option A** — Use the real Kaggle dataset:
1. Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the `data/` directory

**Option B** — Generate synthetic data:
```bash
python data/generate_sample.py
```

### 4. Run the ML Pipeline

```bash
python ml/run_pipeline.py
```

This will:
- Preprocess the data (scaling, SMOTE, split)
- Train ML models (Logistic Regression, Random Forest, XGBoost)
- Run data mining algorithms (Isolation Forest, LOF, K-Means, DBSCAN)
- Evaluate all models and generate metrics
- Run SHAP analysis for explainability
- Save models to `ml/models/` and figures to `ml/figures/`

### 5. Start the Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

API will be available at `http://localhost:8000`

### 6. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`

---

## 🔌 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict fraud for a transaction |
| `GET` | `/stats` | Dataset statistics |
| `GET` | `/model-performance` | ML model evaluation metrics |
| `GET` | `/mining-results` | Data mining analysis results |
| `GET` | `/predictions/history` | Past prediction logs |

### Example: Predict a Transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
  }'
```

### Response

```json
{
  "prediction": 0,
  "prediction_label": "Legitimate",
  "probability": 0.03,
  "risk_level": "Low",
  "model_used": "xgboost"
}
```

---

## ⛏ Data Mining Techniques

### Anomaly Detection

| Technique | Approach | Key Parameter |
|-----------|----------|---------------|
| **Isolation Forest** | Tree-based isolation of anomalies | `contamination=0.001` |
| **Local Outlier Factor** | Local density comparison | `n_neighbors=20` |

### Clustering

| Technique | Approach | Use Case |
|-----------|----------|----------|
| **K-Means** | Centroid-based partitioning | Group transactions by behavior |
| **DBSCAN** | Density-based spatial clustering | Find natural transaction clusters |

### Pattern Analysis
- Amount distribution patterns across fraud/legit
- Temporal transaction patterns
- Feature correlation analysis

---

## 🤖 Machine Learning Models

| Model | Type | Key Advantage |
|-------|------|---------------|
| **Logistic Regression** | Linear | Interpretable baseline |
| **Random Forest** | Ensemble (Bagging) | Handles non-linearity |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art performance |

### Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

### Handling Class Imbalance
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied to training data

---

## 📜 License

This project is developed as a semester project for academic purposes.

---

## 👤 Author

**[Your Name]**  
Semester Project — Data Mining and Artificial Intelligence
