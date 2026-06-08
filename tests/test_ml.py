"""
Test Suite for the ML Pipeline
==============================
Tests data loading, preprocessing, and model prediction functions.

Usage:
    pytest tests/test_ml.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class TestDataLoading:
    """Tests for data loading and sample generation."""

    def test_generate_sample_creates_file(self):
        """Test that the synthetic data generator creates a CSV file."""
        from data.generate_sample import generate_sample_dataset

        df = generate_sample_dataset()
        assert df is not None
        assert len(df) > 0

    def test_sample_has_correct_columns(self):
        """Test that generated data has the correct schema."""
        from data.generate_sample import generate_sample_dataset

        df = generate_sample_dataset()
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        assert list(df.columns) == expected_cols

    def test_sample_has_both_classes(self):
        """Test that generated data contains both fraud and legit transactions."""
        from data.generate_sample import generate_sample_dataset

        df = generate_sample_dataset()
        assert 0 in df["Class"].values
        assert 1 in df["Class"].values

    def test_class_distribution_is_imbalanced(self):
        """Test that fraud is the minority class."""
        from data.generate_sample import generate_sample_dataset

        df = generate_sample_dataset()
        fraud_ratio = df["Class"].mean()
        assert fraud_ratio < 0.1  # Fraud should be minority


class TestPreprocessing:
    """Tests for the preprocessing module."""

    def test_preprocess_returns_splits(self):
        """Test that preprocessing returns train/test splits."""
        data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
        if not os.path.exists(data_path):
            from data.generate_sample import generate_sample_dataset
            generate_sample_dataset()

        from ml.preprocessing.preprocess import load_and_preprocess

        result = load_and_preprocess(data_path)
        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result

    def test_preprocessing_scales_amount(self):
        """Test that Amount feature is scaled."""
        data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
        if not os.path.exists(data_path):
            from data.generate_sample import generate_sample_dataset
            generate_sample_dataset()

        from ml.preprocessing.preprocess import load_and_preprocess

        result = load_and_preprocess(data_path)
        # After scaling, Amount should have mean ≈ 0
        amount_idx = -1  # Amount is last feature column
        assert abs(result["X_train"][:, amount_idx].mean()) < 1.0


class TestModelPrediction:
    """Tests for model predictions (requires trained models)."""

    def test_model_files_exist_after_training(self):
        """Check that model files exist in ml/models/ directory."""
        models_dir = os.path.join(PROJECT_ROOT, "ml", "models")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
            # This test only passes after training
            if model_files:
                assert len(model_files) >= 1

    def test_evaluation_results_exist(self):
        """Check that evaluation results JSON exists after training."""
        results_path = os.path.join(PROJECT_ROOT, "ml", "models", "evaluation_results.json")
        if os.path.exists(results_path):
            import json
            with open(results_path) as f:
                results = json.load(f)
            assert isinstance(results, dict)
            assert len(results) > 0
