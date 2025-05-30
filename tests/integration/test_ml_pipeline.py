"""Integration tests for ML pipeline."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_forge.data.data_loader import get_data_loader
from mlops_forge.data.feature_engineering import get_feature_engineer
from mlops_forge.models.model_trainer import get_model_trainer


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for integration testing."""
    # Create a simple dataset for binary classification
    np.random.seed(42)
    n_samples = 100
    
    # Generate numeric features
    feature_1 = np.random.normal(0, 1, n_samples)
    feature_2 = np.random.normal(0, 1, n_samples)
    
    # Generate categorical features
    categories = ['A', 'B', 'C']
    feature_3 = np.random.choice(categories, n_samples)
    
    # Generate target (binary classification)
    # Simple rule: if feature_1 + feature_2 > 0, then target = 1, else target = 0
    target = (feature_1 + feature_2 > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'target': target
    })
    
    return df


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestMLPipeline:
    """Integration tests for ML pipeline."""
    
    def test_end_to_end_pipeline(self, sample_dataset, temp_data_dir):
        """Test end-to-end ML pipeline."""
        # Step 1: Save sample dataset to CSV
        data_path = os.path.join(temp_data_dir, "sample_data.csv")
        sample_dataset.to_csv(data_path, index=False)
        
        # Step 2: Load data
        data_loader = get_data_loader(data_path=os.path.dirname(data_path))
        data = data_loader.load_data(os.path.basename(data_path))
        
        assert data.shape == sample_dataset.shape
        assert list(data.columns) == list(sample_dataset.columns)
        
        # Step 3: Split data
        splits = data_loader.split_data(
            data=data,
            target_column="target",
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        assert "X_train" in splits
        assert "y_train" in splits
        assert "X_val" in splits
        assert "y_val" in splits
        assert "X_test" in splits
        assert "y_test" in splits
        
        # Step 4: Feature engineering
        feature_engineer = get_feature_engineer()
        X_train_processed = feature_engineer.fit_transform(splits["X_train"])
        X_val_processed = feature_engineer.transform(splits["X_val"])
        X_test_processed = feature_engineer.transform(splits["X_test"])
        
        # Step 5: Train model
        model_trainer = get_model_trainer(
            model_type="random_forest",
            hyperparameters={"n_estimators": 10, "random_state": 42}
        )
        
        model = model_trainer.train(
            X_train=X_train_processed,
            y_train=splits["y_train"],
            X_val=X_val_processed,
            y_val=splits["y_val"]
        )
        
        # Step 6: Evaluate model on test data
        test_metrics = model_trainer.evaluate(X_test_processed, splits["y_test"], prefix="test_")
        
        # Verify metrics are reasonable (accuracy > 0.7 for this simple dataset)
        assert test_metrics["test_accuracy"] > 0.7
        
        # Step 7: Save and load model
        model_path = os.path.join(temp_data_dir, "model.joblib")
        saved_path = model_trainer.save_model(model_path)
        
        assert os.path.exists(saved_path)
        
        # Step 8: Load model in new trainer
        new_trainer = get_model_trainer()
        loaded_model = new_trainer.load_model(saved_path)
        
        # Step 9: Make predictions with loaded model
        test_predictions = new_trainer.predict(X_test_processed)
        
        # Verify predictions match
        np.testing.assert_array_equal(
            model_trainer.predict(X_test_processed),
            test_predictions
        )
