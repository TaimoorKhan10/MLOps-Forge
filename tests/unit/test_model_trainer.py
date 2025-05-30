"""Unit tests for model trainer."""

import os
import tempfile
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mlops_forge.models.model_trainer import ModelTrainer, get_model_trainer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create simple dataset with 2 features
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR function
    
    return X, y


class TestModelTrainer:
    """Test model trainer class."""
    
    def test_init(self):
        """Test initialization."""
        model_trainer = ModelTrainer(model_type="random_forest")
        assert model_trainer.model_type == "random_forest"
        assert model_trainer.model is None
    
    def test_get_model_instance(self):
        """Test getting model instance."""
        # Test random forest
        model_trainer = ModelTrainer(model_type="random_forest", hyperparameters={"n_estimators": 10})
        model = model_trainer._get_model_instance()
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
        
        # Test logistic regression
        model_trainer = ModelTrainer(model_type="logistic_regression", hyperparameters={"C": 0.1})
        model = model_trainer._get_model_instance()
        assert model.__class__.__name__ == "LogisticRegression"
        assert model.C == 0.1
        
        # Test unsupported model type
        model_trainer = ModelTrainer(model_type="unsupported_model")
        with pytest.raises(ValueError, match="Unsupported model type"):
            model_trainer._get_model_instance()
    
    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.log_params")
    @mock.patch("mlflow.log_param")
    @mock.patch("mlflow.log_metric")
    @mock.patch("mlflow.sklearn.log_model")
    def test_train(self, mock_log_model, mock_log_metric, mock_log_param, 
                  mock_log_params, mock_start_run, sample_data):
        """Test model training."""
        # Set up mock for MLflow start_run context manager
        mock_run = mock.MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        X, y = sample_data
        
        # Train model
        model_trainer = ModelTrainer(model_type="random_forest", hyperparameters={"n_estimators": 10})
        model = model_trainer.train(X, y)
        
        # Verify model was trained
        assert model is not None
        assert model_trainer.model is not None
        
        # Verify MLflow logging was called
        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once()
        mock_log_param.assert_called_once_with("model_type", "random_forest")
        assert mock_log_metric.call_count > 0
        mock_log_model.assert_called_once()
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Train model first
        model_trainer = ModelTrainer(model_type="random_forest", hyperparameters={"n_estimators": 10, "random_state": 42})
        model = model_trainer.train(X, y)
        
        # Evaluate on same data
        metrics = model_trainer.evaluate(X, y, prefix="test_")
        
        # Verify metrics
        assert "test_accuracy" in metrics
        assert "test_f1" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        
        # Test evaluation without training
        model_trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Model is not trained"):
            model_trainer.evaluate(X, y)
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        X, y = sample_data
        
        # Train model
        model_trainer = ModelTrainer(model_type="random_forest", hyperparameters={"n_estimators": 10, "random_state": 42})
        model_trainer.train(X, y)
        
        # Save model to temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.joblib")
            saved_path = model_trainer.save_model(model_path)
            
            assert os.path.exists(saved_path)
            
            # Load model in new trainer
            new_trainer = ModelTrainer()
            loaded_model = new_trainer.load_model(saved_path)
            
            # Verify model was loaded
            assert loaded_model is not None
            assert new_trainer.model is not None
            
            # Verify predictions match
            np.testing.assert_array_equal(
                model_trainer.predict(X),
                new_trainer.predict(X)
            )
    
    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        
        # Train model
        model_trainer = ModelTrainer(model_type="random_forest", hyperparameters={"n_estimators": 10, "random_state": 42})
        model_trainer.train(X, y)
        
        # Make predictions
        predictions = model_trainer.predict(X)
        
        # Verify predictions
        assert predictions.shape == y.shape
        
        # Test prediction without training
        model_trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Model is not trained"):
            model_trainer.predict(X)
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        
        # Train model
        model_trainer = ModelTrainer(model_type="random_forest", hyperparameters={"n_estimators": 10, "random_state": 42})
        model_trainer.train(X, y)
        
        # Make probability predictions
        proba = model_trainer.predict_proba(X)
        
        # Verify probability predictions
        assert proba.shape == (X.shape[0], 2)  # Binary classification
        
        # Test prediction without training
        model_trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Model is not trained"):
            model_trainer.predict_proba(X)
    
    def test_get_model_trainer(self):
        """Test get_model_trainer factory function."""
        model_trainer = get_model_trainer(model_type="random_forest")
        assert isinstance(model_trainer, ModelTrainer)
        assert model_trainer.model_type == "random_forest"
