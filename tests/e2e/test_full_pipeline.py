"""End-to-end test for the complete ML pipeline."""

import os
import json
import pytest
import pandas as pd
import requests
from pathlib import Path

from mlops_forge.config.settings import settings
from mlops_forge.data.data_loader import get_data_loader
from mlops_forge.pipeline.training_pipeline import run_training_pipeline


@pytest.fixture(scope="module")
def sample_data_path():
    """Generate sample data for testing."""
    # Import here to avoid circular imports
    from scripts.generate_sample_data import generate_data
    
    # Create sample data directory if it doesn't exist
    os.makedirs(settings.DATA.RAW_DATA_DIR, exist_ok=True)
    
    # Generate sample data
    sample_data_path = os.path.join(settings.DATA.RAW_DATA_DIR, "test_sample_data.csv")
    generate_data(
        n_samples=1000,
        n_features=10,
        output_path=sample_data_path,
        random_state=42
    )
    
    return sample_data_path


@pytest.fixture(scope="module")
def trained_model(sample_data_path):
    """Train a model for testing."""
    # Set output directory for test artifacts
    test_output_dir = os.path.join(settings.MODEL.MODEL_DIR, "test")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Run training pipeline
    metrics = run_training_pipeline(
        data_path=sample_data_path,
        target_column="target",
        model_type="random_forest",
        output_dir=test_output_dir,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    return {
        "metrics": metrics,
        "model_path": os.path.join(test_output_dir, "random_forest_model.joblib"),
        "metadata_path": os.path.join(test_output_dir, "metadata.json"),
        "output_dir": test_output_dir
    }


@pytest.fixture(scope="module")
def api_client():
    """Start the API server for testing."""
    import subprocess
    import time
    import requests
    
    # Start the API server in a subprocess
    process = subprocess.Popen(
        ["python", "-m", "mlops_production_system.api.api"],
        env={**os.environ, "MODEL_DIR": os.path.join(settings.MODEL.MODEL_DIR, "test")}
    )
    
    # Wait for the API to start
    api_url = f"http://{settings.API.HOST}:{settings.API.PORT}/health"
    max_retries = 30
    retries = 0
    
    while retries < max_retries:
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        
        time.sleep(1)
        retries += 1
    
    if retries == max_retries:
        process.terminate()
        pytest.fail("API server failed to start")
    
    # Return API base URL
    api_base_url = f"http://{settings.API.HOST}:{settings.API.PORT}"
    
    yield api_base_url
    
    # Clean up
    process.terminate()


def test_data_generation(sample_data_path):
    """Test that sample data is generated correctly."""
    assert os.path.exists(sample_data_path)
    
    # Load the data
    df = pd.read_csv(sample_data_path)
    
    # Check basic properties
    assert len(df) == 1000
    assert "target" in df.columns
    assert len(df.columns) == 11  # 10 features + target


def test_model_training(trained_model):
    """Test that the model training pipeline works end-to-end."""
    # Check that model file exists
    assert os.path.exists(trained_model["model_path"])
    
    # Check that metadata file exists
    assert os.path.exists(trained_model["metadata_path"])
    
    # Check metrics
    assert "test_accuracy" in trained_model["metrics"]
    assert trained_model["metrics"]["test_accuracy"] > 0.7  # Reasonable accuracy threshold
    
    # Check metadata content
    with open(trained_model["metadata_path"], "r") as f:
        metadata = json.load(f)
    
    assert metadata["model_name"] == "random_forest"
    assert "metrics" in metadata
    assert "features" in metadata


def test_api_health(api_client):
    """Test that the API health endpoint works."""
    response = requests.get(f"{api_client}/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_model_info(api_client):
    """Test that the model info endpoint works."""
    response = requests.get(f"{api_client}/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "metrics" in data


def test_prediction(api_client, sample_data_path):
    """Test that the prediction endpoint works."""
    # Load sample data
    df = pd.read_csv(sample_data_path)
    
    # Get a single sample (without the target)
    sample = df.drop(columns=["target"]).iloc[0].to_dict()
    
    # Make prediction
    response = requests.post(
        f"{api_client}/predict",
        json={"features": sample}
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "prediction_probability" in data
    assert "model_version" in data


def test_full_pipeline_e2e(sample_data_path, trained_model, api_client):
    """Test the full end-to-end pipeline from data to prediction."""
    # This test ties everything together
    
    # 1. Verify data
    df = pd.read_csv(sample_data_path)
    assert len(df) > 0
    
    # 2. Verify model artifacts
    assert os.path.exists(trained_model["model_path"])
    with open(trained_model["metadata_path"], "r") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == "random_forest"
    
    # 3. Verify API
    health_response = requests.get(f"{api_client}/health")
    assert health_response.status_code == 200
    
    # 4. Make prediction with API
    sample = df.drop(columns=["target"]).iloc[0].to_dict()
    pred_response = requests.post(
        f"{api_client}/predict",
        json={"features": sample}
    )
    assert pred_response.status_code == 200
    assert "prediction" in pred_response.json()
    
    # The test passes if all components work together correctly
