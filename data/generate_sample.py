"""
Synthetic Data Generator
========================
Generates a sample dataset that mirrors the Kaggle Credit Card Fraud dataset
schema so the project can run without the original 150 MB file.

Columns: Time, V1–V28, Amount, Class
- Class 0 = legitimate transaction
- Class 1 = fraudulent transaction (≈0.17% of total)

Usage:
    python data/generate_sample.py
"""

import numpy as np
import pandas as pd
import os

# ── Configuration ──────────────────────────────────────────────
N_SAMPLES = 5000          # Total number of transactions
FRAUD_RATIO = 0.017       # ~1.7% fraud (slightly higher than real 0.17% for better demo)
RANDOM_SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "creditcard.csv")


def generate_sample_dataset():
    """Generate a synthetic credit card transaction dataset."""
    np.random.seed(RANDOM_SEED)

    n_fraud = int(N_SAMPLES * FRAUD_RATIO)
    n_legit = N_SAMPLES - n_fraud

    print(f"Generating synthetic dataset...")
    print(f"  Total transactions : {N_SAMPLES}")
    print(f"  Legitimate         : {n_legit}")
    print(f"  Fraudulent         : {n_fraud}")

    # ── Time feature (seconds elapsed, 0 to ~172800 = 2 days) ──
    time_values = np.sort(np.random.uniform(0, 172800, N_SAMPLES))

    # ── V1–V28: Anonymized PCA features ──
    # Legitimate transactions: centered around 0, moderate spread
    legit_features = np.random.normal(loc=0, scale=1.0, size=(n_legit, 28))

    # Fraudulent transactions: shifted means and higher variance
    # Mimics the real dataset where fraud has distinct PCA signatures
    fraud_means = np.random.uniform(-2, 2, 28)
    fraud_features = np.random.normal(loc=fraud_means, scale=1.5, size=(n_fraud, 28))

    # Make some features more discriminative (like V14, V17 in real data)
    fraud_features[:, 13] -= 3.0   # V14 strongly negative for fraud
    fraud_features[:, 16] -= 2.5   # V17 strongly negative for fraud
    fraud_features[:, 11] += 2.0   # V12 shifted for fraud
    fraud_features[:, 9]  -= 2.0   # V10 shifted for fraud

    # ── Amount feature ──
    legit_amounts = np.abs(np.random.exponential(scale=88, size=n_legit))
    fraud_amounts = np.abs(np.random.exponential(scale=120, size=n_fraud))

    # ── Labels ──
    legit_labels = np.zeros(n_legit, dtype=int)
    fraud_labels = np.ones(n_fraud, dtype=int)

    # ── Combine ──
    features = np.vstack([legit_features, fraud_features])
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    labels = np.concatenate([legit_labels, fraud_labels])

    # Shuffle everything together
    indices = np.random.permutation(N_SAMPLES)
    features = features[indices]
    amounts = amounts[indices]
    labels = labels[indices]

    # ── Build DataFrame ──
    v_columns = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(features, columns=v_columns)
    df.insert(0, "Time", time_values)
    df["Amount"] = np.round(amounts, 2)
    df["Class"] = labels

    # ── Save ──
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Dataset saved to: {OUTPUT_FILE}")
    print(f"  Shape: {df.shape}")
    print(f"  Fraud distribution:\n{df['Class'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    generate_sample_dataset()
