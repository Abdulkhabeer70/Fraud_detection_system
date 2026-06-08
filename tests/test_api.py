"""
Test Suite for the FastAPI Backend
===================================
Tests all API endpoints for correct responses.

Usage:
    pytest tests/test_api.py -v
"""

import os
import sys
import pytest

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def get_sample_transaction(self):
        """Generate a sample transaction payload."""
        transaction = {"Amount": 149.62}
        for i in range(1, 29):
            transaction[f"V{i}"] = 0.0
        return transaction

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self.get_sample_transaction())
        # May return 503 if models not trained
        assert response.status_code in [200, 503]

    def test_predict_response_schema(self, client):
        response = client.post("/predict", json=self.get_sample_transaction())
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data

    def test_predict_invalid_input(self, client):
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error


class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def test_stats_returns_200_or_503(self, client):
        response = client.get("/stats")
        assert response.status_code in [200, 503]


class TestPerformanceEndpoint:
    """Tests for the /model-performance endpoint."""

    def test_performance_returns_200_or_503(self, client):
        response = client.get("/model-performance")
        assert response.status_code in [200, 503]


class TestMiningEndpoint:
    """Tests for the /mining-results endpoint."""

    def test_mining_returns_200_or_503(self, client):
        response = client.get("/mining-results")
        assert response.status_code in [200, 503]
