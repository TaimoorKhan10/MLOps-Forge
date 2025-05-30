"""Unit tests for API."""

import json
import uuid
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from mlops_forge.api.api import app


@pytest.fixture
def test_client():
    """Create a test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_trainer():
    """Create mock model trainer."""
    with mock.patch("mlops_forge.api.api.MODEL") as mock_trainer:
        # Configure mock to return predictions
        mock_trainer.predict.return_value = [1]
        mock_trainer.predict_proba.return_value = [[0.2, 0.8]]
        mock_trainer.model = mock.MagicMock()
        yield mock_trainer


class TestAPI:
    """Test API endpoints."""
    
    def test_health_check(self, test_client, mock_model_trainer):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "model_version" in data
        assert "api_version" in data
    
    def test_health_check_model_not_loaded(self, test_client):
        """Test health check when model is not loaded."""
        # Patch model_trainer to None
        with mock.patch("mlops_production_system.api.api.model_trainer", None):
            response = test_client.get("/health")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_predict(self, test_client, mock_model_trainer):
        """Test predict endpoint."""
        # Create test data
        test_data = {
            "features": {
                "feature_1": 0.5,
                "feature_2": 10
            }
        }
        
        # Mock uuid generation for consistent testing
        with mock.patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")):
            response = test_client.post("/predict", json=test_data)
        
        # Verify response
        assert response.status_code == 200
        
        data = response.json()
        assert data["prediction"] == 1
        assert data["probability"] == 0.8
        assert data["prediction_id"] == "00000000-0000-0000-0000-000000000000"
        assert "model_version" in data
    
    def test_predict_model_not_loaded(self, test_client):
        """Test predict when model is not loaded."""
        # Create test data
        test_data = {
            "features": {
                "feature_1": 0.5,
                "feature_2": 10
            }
        }
        
        # Patch model_trainer to None
        with mock.patch("mlops_production_system.api.api.model_trainer", None):
            response = test_client.post("/predict", json=test_data)
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_batch_predict(self, test_client, mock_model_trainer):
        """Test batch predict endpoint."""
        # Create test data
        test_data = {
            "instances": [
                {
                    "feature_1": 0.5,
                    "feature_2": 10
                },
                {
                    "feature_1": 0.7,
                    "feature_2": 20
                }
            ]
        }
        
        # Configure mock to return multiple predictions
        mock_model_trainer.predict.return_value = [1, 0]
        mock_model_trainer.predict_proba.return_value = [[0.2, 0.8], [0.6, 0.4]]
        
        # Mock uuid generation for consistent testing
        uuid_values = [uuid.UUID("00000000-0000-0000-0000-000000000000"), uuid.UUID("11111111-1111-1111-1111-111111111111")]
        with mock.patch("uuid.uuid4", side_effect=uuid_values):
            response = test_client.post("/batch-predict", json=test_data)
        
        # Verify response
        assert response.status_code == 200
        
        data = response.json()
        assert data["predictions"] == [1, 0]
        assert data["probabilities"] == [0.8, 0.4]
        assert data["prediction_ids"] == ["00000000-0000-0000-0000-000000000000", "11111111-1111-1111-1111-111111111111"]
        assert "model_version" in data
    
    def test_metrics(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_model_metadata(self, test_client):
        """Test model metadata endpoint."""
        # Mock model_metadata
        mock_metadata = {
            "model_name": "test_model",
            "model_version": "1.0.0",
            "model_type": "random_forest",
            "created_at": "2023-09-15T12:00:00Z",
            "features": ["feature_1", "feature_2"],
            "metrics": {"accuracy": 0.95}
        }
        
        with mock.patch("mlops_production_system.api.api.model_metadata", mock_metadata):
            response = test_client.get("/metadata")
            
            # Verify response
            assert response.status_code == 200
            assert response.json() == mock_metadata
    
    def test_model_metadata_not_found(self, test_client):
        """Test model metadata when not found."""
        # Patch model_metadata to None
        with mock.patch("mlops_production_system.api.api.model_metadata", None):
            response = test_client.get("/metadata")
            assert response.status_code == 404
            assert "Model metadata not found" in response.json()["detail"]
