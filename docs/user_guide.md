# MLOps-Forge User Guide

This user guide provides comprehensive documentation for using the MLOps-Forge framework, including all components, deployment options, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Getting Started](#getting-started)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Model Deployment](#model-deployment)
7. [Monitoring](#monitoring)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [API Documentation](#api-documentation)
10. [Troubleshooting](#troubleshooting)
11. [Cloud Deployment](#cloud-deployment)
12. [Advanced Features](#advanced-features)

## Overview

The MLOps-Forge framework is a comprehensive solution for building, deploying, and monitoring machine learning models in production. It follows MLOps best practices to ensure reproducibility, scalability, and reliability of machine learning pipelines.

### Key Features

- **End-to-End ML Pipeline**: From data ingestion to model deployment
- **Experiment Tracking**: Track experiments with MLflow
- **Model Registry**: Version and manage models
- **Monitoring**: Detect data drift and model performance degradation
- **CI/CD Integration**: Automate testing, building, and deployment
- **Scalable Architecture**: Support for distributed training and serving
- **A/B Testing**: Test models in production
- **Cloud Deployment**: Deploy to multiple cloud providers

## System Components

The MLOps Production System consists of the following core components:

### 1. Data Pipeline

- Data ingestion from various sources
- Data validation and quality checks
- Feature engineering
- Data versioning

### 2. Model Training

- Model training with various algorithms
- Hyperparameter tuning
- Experiment tracking
- Distributed training support

### 3. Model Registry

- Model versioning
- Model metadata storage
- Model promotion workflow

### 4. API Layer

- Model serving via REST API
- Input validation
- Authentication and authorization
- Performance optimization

### 5. Monitoring System

- Data drift detection
- Model performance monitoring
- Feature-level monitoring
- Automated retraining triggers

### 6. CI/CD Pipeline

- Automated testing
- Docker image building
- Kubernetes deployment
- Model validation

### 7. Infrastructure Components

- Docker containerization
- Kubernetes orchestration
- Terraform infrastructure as code
- Prometheus and Grafana monitoring

## Getting Started

### Prerequisites

- Python 3.9 or later
- Docker and Docker Compose
- Kubernetes (for production deployment)
- Git

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/MLOps-Production-System.git
cd MLOps-Production-System
```

2. Install dependencies:

```bash
pip install -e .
```

3. Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start the development environment:

```bash
docker-compose -f infrastructure/docker/docker-compose.yml up -d
```

### Quick Start

To quickly test the system with sample data:

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Train a model
python scripts/demo_ml_pipeline.py --train

# Make predictions with the trained model
python scripts/demo_ml_pipeline.py --predict
```

## Data Pipeline

The data pipeline handles all aspects of data processing, from ingestion to feature engineering.

### Data Loaders

The system includes several data loaders for different data sources:

```python
from mlops_forge.data import get_data_loader

# Load and preprocess data
data_loader = get_data_loader()
data = data_loader.load_data("data.csv")
processed_data = data_loader.preprocess_data(data)

# Split data for training
split_data = data_loader.split_data(processed_data, target_column="target")
```

### Feature Engineering

The feature engineering module transforms raw data into features suitable for machine learning:

```python
from mlops_forge.data.feature_engineering import get_feature_engineer

# Initialize feature engineer
feature_engineer = get_feature_engineer()

# Identify feature types
categorical_features = data.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Fit and transform the data
X_transformed = feature_engineer.fit_transform(
    data,
    categorical_features=categorical_features,
    numerical_features=numerical_features
)

# Save feature configuration for inference
feature_engineer.save_features("models/feature_metadata.json")
```

## Model Training

The model training component supports different types of models and integrates with MLflow for experiment tracking.

### Basic Training

```python
from mlops_forge.models.model_trainer import get_model_trainer

# Initialize model trainer
model_trainer = get_model_trainer(
    model_type="random_forest",
    hyperparameters={"n_estimators": 100, "random_state": 42}
)

# Train the model
model_trainer.train(X_train, y_train, X_val, y_val)

# Evaluate the model
metrics = model_trainer.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")

# Save the model
model_trainer.save_model("models/random_forest_model.joblib")
```

### Distributed Training

For large datasets, the system supports distributed training with PyTorch:

```bash
# Run on multiple GPUs
python -m mlops_forge.training.distributed_trainer \
  --master \
  --num-workers=4 \
  --dataset=/data/training/large_dataset.csv \
  --batch-size=64 \
  --epochs=10
```

### Experiment Tracking

The system integrates with MLflow for experiment tracking:

```python
from mlops_forge.tracking.mlflow_extensions import ExperimentTracker, track_experiment

# Manual tracking
tracker = ExperimentTracker(experiment_name="model_comparison")
tracker.start_run(run_name="random_forest_run")

# Log parameters
mlflow.log_params({
    "model_type": "random_forest",
    "n_estimators": 100
})

# Train and evaluate
metrics = model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

# Log metrics
tracker.log_model_performance(y_test, y_pred, y_prob)

# End run
tracker.end_run()

# Or use the decorator for automatic tracking
@track_experiment(experiment_name="model_training")
def train_model(X_train, y_train, X_test, y_test, model_type="random_forest"):
    model_trainer = get_model_trainer(model_type=model_type)
    model_trainer.train(X_train, y_train)
    return model_trainer.evaluate(X_test, y_test)
```

## Model Deployment

The system provides multiple options for model deployment.

### API Deployment

Deploy models as REST APIs using FastAPI:

```bash
# Start the API server
python -m mlops_forge.api.api run_server
```

### Docker Deployment

For containerized deployment:

```bash
# Build the Docker image
docker build -f infrastructure/docker/Dockerfile -t mlops_forge:latest .

# Run the container
docker run -p 8000:8000 -e MODEL_PATH=/app/models/model.joblib mlops_forge:latest
```

### Kubernetes Deployment

For production deployment:

```bash
# Deploy to Kubernetes
kubectl apply -f infrastructure/kubernetes/namespace.yaml
kubectl apply -f infrastructure/kubernetes/configmap.yaml
kubectl apply -f infrastructure/kubernetes/deployment.yaml
kubectl apply -f infrastructure/kubernetes/service.yaml
```

## Monitoring

The monitoring system tracks model performance and detects data drift in production.

### Metrics Collection

```python
from mlops_forge.monitoring.metrics import track_prediction_latency

# Track prediction latency
@track_prediction_latency
def predict(features):
    return model.predict(features)
```

### Drift Detection

```python
from mlops_forge.monitoring.drift_detection import FeatureDriftDetector

# Initialize drift detector with reference data
drift_detector = FeatureDriftDetector(
    reference_data=training_data,
    categorical_features=["category", "country"],
    numerical_features=["age", "income"],
    warning_threshold=0.1,
    drift_threshold=0.2
)

# Check for drift in production data
drift_results = drift_detector.compute_drift(production_data)

# Check if retraining is needed
if drift_detector.should_retrain():
    print("Data drift detected. Model retraining recommended.")
```

### A/B Testing

```python
from mlops_production_system.monitoring.ab_testing import get_ab_test_manager

# Get the global A/B test manager
ab_manager = get_ab_test_manager()

# Create an experiment
ab_manager.create_experiment(
    experiment_id="model_comparison",
    variants={
        "model_v1": model_v1,
        "model_v2": model_v2
    },
    traffic_allocation={"model_v1": 0.5, "model_v2": 0.5},
    description="Comparing model performance",
    auto_start=True
)

# Get a variant for a request
experiment_id, variant_name = ab_manager.get_variant(user_id="user123")
model = ab_manager.get_experiment(experiment_id).variants[variant_name]

# Make prediction with the selected variant
prediction = model.predict(features)

# Record conversion
ab_manager.record_conversion(experiment_id, variant_name)
```

## CI/CD Pipeline

The system includes a CI/CD pipeline implemented with GitHub Actions.

### Pipeline Stages

1. **Test**: Run unit and integration tests
2. **Build**: Build Docker images
3. **Model Validation**: Validate model performance
4. **Deploy**: Deploy to Kubernetes

### Configuring GitHub Actions

The workflow is defined in `.github/workflows/main.yml`. To use it, set up the following GitHub secrets:

- `CODECOV_TOKEN`: For coverage reporting
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_REGION`: AWS region
- `EKS_CLUSTER_NAME`: EKS cluster name
- `MLFLOW_TRACKING_URI`: MLflow tracking server URL

## API Documentation

### API Endpoints

The API provides the following endpoints:

- `GET /health`: Health check
- `GET /model/info`: Get model metadata
- `POST /predict`: Make predictions

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.0, "feature2": "category1"}}'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_probability": 0.85,
  "model_version": "0.1.0",
  "processing_time_ms": 5
}
```

## Troubleshooting

### Common Issues

#### Model Loading Errors

```
Error: Failed to load model from path: models/model.joblib
```

**Solution**: Ensure the model file exists and is accessible. Check the `MODEL_PATH` environment variable.

#### API Connection Issues

```
Error: Connection refused when connecting to API at http://localhost:8000
```

**Solution**: Verify the API service is running and ports are correctly exposed. Check Docker logs with `docker logs <container_id>`.

#### Data Drift Alerts

```
Warning: Data drift detected with score 0.35 (threshold: 0.2)
```

**Action**: Review drift details and consider retraining the model with more recent data.

## Cloud Deployment

The MLOps Production System supports deployment to multiple cloud providers.

### AWS Deployment

1. Set up AWS credentials:

```bash
aws configure
```

2. Deploy infrastructure with Terraform:

```bash
cd infrastructure/terraform
terraform init
terraform apply -var="aws_region=us-east-1" -var="cluster_name=mlops-cluster"
```

3. Deploy the application to EKS:

```bash
aws eks update-kubeconfig --name mlops-cluster --region us-east-1
kubectl apply -f infrastructure/kubernetes/
```

### Azure Deployment

See [Azure Deployment Guide](cloud_deployment_azure.md) for details.

### GCP Deployment

See [GCP Deployment Guide](cloud_deployment_gcp.md) for details.

## Advanced Features

### Distributed Training

The system supports distributed training for large datasets:

```bash
# Deploy distributed training job to Kubernetes
kubectl apply -f infrastructure/kubernetes/distributed-training.yaml
```

### GPU Support

For GPU-accelerated training and inference:

```bash
# Build GPU-enabled Docker image
docker build -f infrastructure/docker/Dockerfile.gpu -t mlops-production-system:gpu .

# Run with GPU support
docker run --gpus all -p 8000:8000 mlops-production-system:gpu
```

### Automated Retraining

Set up automated retraining based on drift detection:

```python
from mlops_production_system.monitoring.drift_detection import AutomaticRetrainingTrigger

# Initialize retraining trigger
retraining_trigger = AutomaticRetrainingTrigger(
    drift_detector=drift_detector,
    retraining_frequency=24,  # hours
    min_samples_required=1000,
    min_drift_score=0.2
)

# Check if retraining is needed
should_retrain, drift_results = retraining_trigger.check_drift(current_data)
if should_retrain:
    # Trigger retraining pipeline
    subprocess.run(["python", "scripts/retrain_model.py"])
```
